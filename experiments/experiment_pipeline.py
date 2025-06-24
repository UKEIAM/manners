import torch
from typing import Dict, Optional
import wandb
import os
import json
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import collections
import petname
import datetime
from sklearn.linear_model import LogisticRegression, LinearRegression
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.model import FullyConvAutoEncoder
from src.utils import set_seeds, generate_synthetic_data
from src.config import DISCRETE_FEATURES, CLASS_FEATURES, WANDB_ARGS, JSON_LOG_DIR, LABEL_PATH, MODELS_DIR
from src.data_loader import HypoDataset
from src.combined_loss import CombinedLoss
from src.metrics import EvalMetrics
from src.data_structures import TargetDataBatch
from src.arg_parser import ExperimentArgumentParser
from src.logger import JSONLogger, DummyLogger

class ExperimentPipeline:
    """
    The experiment pipeline for training and evaluating autoencoder models on the hypotension dataset.
    This class implements a complete pipeline for training variational and non-variational 
    autoencoders, and evaluating them through reconstruction analysis and survival downstream (non-variational),
    and synthetic data generation (variational) experiments.
    Args:
        config_dict (Dict, optional): Configuration dictionary containing all training hyperparameters, model architecture and settings.
        fixed_model_name (Dict, optional): Fixed model name, no timestamp will be added.
        device (str, optional): Device to run computations on ('cpu' or 'cuda').
        seed (int, optional): Random seed for reproducibility, can also be set in `config_dict`.
    """
    def __init__(self, config_dict: Optional[Dict] = None, fixed_model_name: Optional[Dict] = None, device: Optional[str] = None, seed: Optional[int] = None):
        self.config_dict = config_dict
        self.model_name = fixed_model_name
        self.model_dir =  MODELS_DIR / pathlib.Path(self.model_name) if self.model_name is not None else None
        self.device = device
        self.seed = seed
        self.model = None
        self.dataset = None
        self.data_loaders = None
        self.synth_eval_data = None
        self.verbose = self.config_dict['base']['verbose'] if self.config_dict is not None else None

    def run(self):
        pipeline_steps = set(self.config_dict['base']['pipeline'])
        verbose = self.verbose if self.verbose is not None else True
        if 'train' in pipeline_steps:
            if self.config_dict is None:
                raise AttributeError('Config dictionary must be provided when training model')
            variational = self.config_dict['auto encoder']['variational']
            if variational:
                if 'reconstruction' in pipeline_steps or 'downstream' in pipeline_steps:
                    raise AttributeError('Variational autoencoders can only be evaluated with synthetic data experiments')
            else:
                if 'synthesis' in pipeline_steps:
                    raise AttributeError('Un-variational autoencoders can only be evaluated with reconstruction and downstream experiments')
            if self.model_name is None:
                if self.config_dict['base']['name'] is None:
                    self.model_name = petname.Generate()
                else:
                    self.model_name = self.config_dict['base']['name']
                self.model_name += '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                self.model_dir = MODELS_DIR / pathlib.Path(self.model_name)
        else:
            if 'synthesis' in pipeline_steps and ('reconstruction' in pipeline_steps or 'downstream' in pipeline_steps):
                raise AttributeError('Variational autoencoders can only be evaluated with synthetic data experiments, un-variational autoencoders with reconstruction and downstream experiments')
            if self.model_name is None and (self.config_dict is None or self.config_dict['base']['name'] is None):
                raise AttributeError('Model name must be provided when evaluating model')
            if self.model_name is None:
                self.model_name = self.config_dict['base']['name']

        experiments = sorted([step for step in pipeline_steps if step != 'train'])
        exp_funcs = []
        for idx, experiment in enumerate(experiments):
            if experiment == 'downstream':
                exp_funcs.append(self.perform_downstream_tasks)
            elif experiment == 'reconstruction':
                exp_funcs.append(self.perform_reconstruction_analysis)
            elif experiment == 'synthesis':
                exp_funcs.append(self.generate_synthetic_test_data)
            else:
                raise NotImplementedError('Pipeline step "' + experiment + '" not implemented')

        if 'train' in pipeline_steps:
            self.train_model(verbose=verbose)

        for idx, func in enumerate(exp_funcs):
            func(load_model=idx==0, verbose=verbose)

    def load_best_epoch_model(self):
        if self.model_dir is None:
            self.model_dir =  MODELS_DIR / pathlib.Path(self.model_name)
        if not self.model_dir.exists():
            raise AttributeError('Model "' + self.model_name + '" does not exist in weights directory structure')

        models = list(self.model_dir.glob('*.pt'))
        if len(models) != 1:
            raise AttributeError('Directory ' + str(self.model_dir) + ' must contain exactly one weights file')
        
        set_debug = self.config_dict['base']['debug'] if self.config_dict is not None else None
        
        log_json_path = JSON_LOG_DIR / pathlib.Path(self.model_name + '.json')

        if log_json_path.exists():
            with open(log_json_path, 'r') as f:
                log_data = json.load(f)
                self.config_dict = log_data['config']

        else:
            api = wandb.Api(overrides=WANDB_ARGS)
            runs = api.runs(filters={"display_name" : self.model_name})
            if len(runs) == 0:
                raise AttributeError('Model "' + self.model_name + '" not found in wandb API')

            run = next(runs)
            self.config_dict = run.config
        if set_debug is not None:
            self.config_dict['base']['debug'] = set_debug
        ae_config = self.config_dict['auto encoder']
        stack_mode = self.config_dict['train']['stack_mode']
        if self.device is None:
            self.device = self.config_dict['base']['device']
        if self.seed is None:
            self.seed = self.config_dict['base']['seed']

        self.model = FullyConvAutoEncoder(**ae_config, stack_mode=stack_mode).to(self.device)
        model_path = models[0]
        self.model.load_state_dict(torch.load(model_path))

    def train_model(self, verbose: bool = True):
        if self.model is not None:
            raise AttributeError('Model already initialized')
        assert self.config_dict is not None
        debug = self.config_dict['base']['debug']
        iterations = self.config_dict['train']['iterations']
        learnrate = self.config_dict['train']['learnrate']
        weight_decay = self.config_dict['train']['weight_decay']
        stopping_delta = self.config_dict['early stopping']['increase_delta']
        early_stopping_patience = self.config_dict['early stopping']['stopping_patience']
        normalization = self.config_dict['train']['normalization']
        stack_mode = self.config_dict['train']['stack_mode']
        if self.device is None:
            self.device = self.config_dict['base']['device']

        # create logs
        if verbose:
            print('Training model ' + self.model_name)
        
        if not debug:
            os.makedirs(self.model_dir, exist_ok=True)
        
        self.model = FullyConvAutoEncoder(**self.config_dict['auto encoder'], stack_mode=stack_mode).to(self.device)
        opt = torch.optim.AdamW(lr=learnrate, weight_decay=weight_decay, params=self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', threshold_mode='rel', **self.config_dict['learn rate scheduler'])
        loss = CombinedLoss(**self.config_dict['loss'], variational=self.model.variational, normalization=normalization)

        if self.dataset is None:
            self._load_data(verbose=verbose)
            if not debug:
                self.store_dataset_split()

        if self.synth_eval_data is None:
            self._get_validation_synthetic_eval_data()

        step = 0
        best_val_combined = np.inf
        best_epoch = None
        epochs_since_best = 0
        curr_epoch = 0
        if verbose:
            print('Start training')
        with self._get_logger() as logger:
            while step < iterations:
                if verbose:
                    print('BEGIN EPOCH ' + str(curr_epoch))
                self.model.train()
                # train
                to_iter = tqdm(self.data_loaders['train']) if verbose else self.data_loaders['train']
                for (stacked_batch, disc_batch, class_ordinal_batch, missing_batch, __) in to_iter:
                    target_batch = TargetDataBatch(stacked_batch, disc_batch, class_ordinal_batch, missing_batch, device=self.device)
                    result_batch = self.model(target_batch.stacked)
                    loss_val, loss_dict = loss(result_batch, target_batch)
                    loss_val.backward()
                    opt.step()
                    opt.zero_grad(True)
                    train_log_dict = {'train/' + key : val for key,val in loss_dict.items()}
                    train_log_dict.update({'train/train_step' : step})
                    logger.log(train_log_dict)
                    step += 1
                    if step >= iterations:
                        break

                curr_lr = opt.param_groups[0]['lr']

                # evaluate on validation set after epoch
                with torch.no_grad():
                    overall_metrics, featurewise_metrics = EvalMetrics.evaluate_model(self.model, self.data_loaders['validation'],
                                                                                      self.device, self.config_dict['loss']['huber_delta'])
                    if debug and verbose:
                        if self.model.variational:
                            print('KL divergence: '+ str(overall_metrics['vae']['kl_divergence']))
                        print('Huber: ' + str(overall_metrics['discrete']['huber']))
                    val_log_dict = {
                        'parameters/epoch' : curr_epoch,
                        'parameters/learnrate' : curr_lr,
                    }
                    val_log_dict.update({'validation/' + key : val for key, val in overall_metrics.items()})
                    
                    val_log_dict.update({key + '/' : val for key, val in featurewise_metrics.items()})

                    if self.model.variational:
                        overall_synth_metrics, featurewise_synth_metrics = EvalMetrics.eval_synth_data(self.model, self.synth_eval_data, self.device)
                        if verbose:
                            print(overall_synth_metrics)
                        for feature, feature_data in featurewise_synth_metrics.items():
                            val_log_dict[feature + '/'].update(feature_data)
                        val_log_dict.update({'validation/' + key : val for key, val in overall_synth_metrics.items()})

                    logger.log(val_log_dict)

                    disc_err = overall_metrics['discrete']['huber']
                    missing_err = overall_metrics['missing']['cross_entropy']

                    disc_weight = self.config_dict['loss']['discrete_weight']
                    miss_weight = self.config_dict['loss']['missing_weight']

                    if not self.model.variational:
                        val_combined = disc_weight * disc_err + miss_weight * missing_err
                    else:
                        val_combined = overall_synth_metrics['synth_qq_diffs'] + overall_synth_metrics['synth_missing_diffs'] + overall_synth_metrics['synth_frac_diffs']
                    scheduler.step(val_combined)

                if val_combined < best_val_combined:
                    best_val_combined = val_combined
                    best_epoch = curr_epoch
                    if not self.config_dict['base']['debug']:
                        for weight_file in self.model_dir.glob('*.pt'):
                            weight_file.unlink()
                        model_path = self.model_dir / pathlib.Path(self.model_name + '_epoch' + str(curr_epoch) + '.pt')
                        torch.save(self.model.state_dict(), model_path)
                    epochs_since_best = 0
                # must be actual increase from best
                elif val_combined > (best_val_combined + stopping_delta):
                    epochs_since_best += 1
                
                if epochs_since_best > early_stopping_patience:
                    if verbose:
                        print('FINISHED BECAUSE OF INCREASE WITH DELTA ' + str(stopping_delta) + ' IN EPOCH ' + str(curr_epoch) + '. BEST COMBINED VALIDATION METRIC WAS ' + str(best_val_combined) + ' IN EPOCH ' + str(best_epoch))
                    break

                curr_epoch += 1
        if verbose:
            print('Finished training')

    def perform_downstream_tasks(self, load_model: bool = True, verbose: bool = True):
        if verbose:
            print('Downstream experiments')
        if load_model:
            self.load_best_epoch_model()

        if self.model is None:
            raise AttributeError('No model trained or loaded for evaluation')

        if self.model.variational:
            raise AttributeError('Unvarational models must be used here')

        if self.dataset is None:
            self._load_data(verbose=verbose)

        label_df = pd.read_csv(LABEL_PATH)

        survival_dict = dict(zip(label_df['stay_id'], label_df['survival48h']))
        los_dict = dict(zip(label_df['stay_id'], label_df['los']))

        train_data_to_stack = []
        y_train_survival = []
        y_train_los = []
        test_data_to_stack = []
        y_test_survival = []
        y_test_los = []
        test_stay_ids = []

        if verbose:
            print('Extract data for train and test')

        with torch.no_grad():
            for (stacked_batch, disc_batch, class_ordinal_batch, missing_batch, stay_id_batch) in tqdm(self.data_loaders['train']):
                target_batch = TargetDataBatch(stacked_batch, disc_batch, class_ordinal_batch, missing_batch, device=self.device)
                self.model.eval()
                z = self.model.encode(target_batch.stacked)
                train_data_to_stack.append(z)
                for stay_id in stay_id_batch:
                    y_train_survival.append(survival_dict[stay_id.item()])
                    y_train_los.append(los_dict[stay_id.item()])
            for (stacked_batch, disc_batch, class_ordinal_batch, missing_batch, stay_id_batch) in tqdm(self.data_loaders['test']):
                target_batch = TargetDataBatch(stacked_batch, disc_batch, class_ordinal_batch, missing_batch, device=self.device)
                self.model.eval()
                z = self.model.encode(target_batch.stacked)
                test_data_to_stack.append(z)
                for stay_id in stay_id_batch:
                    y_test_survival.append(survival_dict[stay_id.item()])
                    y_test_los.append(los_dict[stay_id.item()])
                    test_stay_ids.append(stay_id.item())

        train_encodings = torch.cat(train_data_to_stack)
        test_encodings = torch.cat(test_data_to_stack)

        X_train = train_encodings.detach().cpu().numpy()
        X_test = test_encodings.detach().cpu().numpy()

        if verbose:
            print('Fit to train data')

        survival_clf = LogisticRegression(random_state=self.seed, max_iter=10000).fit(X_train, y_train_survival)
        los_regression = LinearRegression().fit(X_train, y_train_los)

        if verbose:
            print('Predict on test set')

        y_test_survival_pred = survival_clf.predict(X_test)

        y_test_survival_pred = np.where(y_test_survival_pred >= 0.5, 1, 0)

        y_test_los_pred = los_regression.predict(X_test)

        result_df = pd.DataFrame({'stay_id' : test_stay_ids, 'survival48h': y_test_survival, 'survival_prediction' : y_test_survival_pred, 'los' : y_test_los, 'los_prediction' : y_test_los_pred})

        if not self.config_dict['base']['debug']:
            result_df.to_csv(self.model_dir / pathlib.Path(self.model_name + '_downstream_results.csv'), index=False)

    def perform_reconstruction_analysis(self, load_model: bool = True, verbose: bool = True):
        if verbose:
            print('Reconstruction experiment')
        if load_model:
            self.load_best_epoch_model()

        if self.model is None:
            raise AttributeError('No model trained or loaded for evaluation')

        if self.model.variational:
            raise AttributeError('Unvarational models must be used here')

        if self.dataset is None:
            self._load_data(verbose=verbose)

        to_cat = []

        with torch.no_grad():
            for (stacked_batch, disc_batch, class_ordinal_batch, missing_batch, stay_id_batch) in tqdm(self.data_loaders['test']):
                target_batch = TargetDataBatch(stacked_batch, disc_batch, class_ordinal_batch, missing_batch, device=self.device)
                self.model.eval()
                model_result = self.model(target_batch.stacked)
                result_df = model_result.to_dataframe(stay_ids=stay_id_batch.detach().cpu().numpy(), keep_inactive_values=True)
                to_cat.append(result_df)

        final_df = pd.concat(to_cat)
        if not self.config_dict['base']['debug']:
            final_df.to_csv(self.model_dir / pathlib.Path(self.model_name + '_test_reconstructions.csv'), index=False)

    def generate_synthetic_test_data(self, load_model: bool = True, verbose: bool = True):
        if verbose:
            print('Synthetic data experiment')
        if load_model:
            self.load_best_epoch_model()

        if self.model is None:
            raise AttributeError('No model trained or loaded for evaluation')

        if not self.model.variational:
            raise AttributeError('Varational models must be used here')
        # 5889 is test set size when 20% test set
        if self.dataset is None:
            nof_samples = 5889
        else:
            nof_samples = 0
            for (stacked_batch, __, __, __, __) in self.data_loaders['test']:
                nof_samples += stacked_batch.shape[0]

        synth_df = generate_synthetic_data(self.model, nof_samples=nof_samples, device=self.device, verbose=True, invert_transform=False)

        if not self.config_dict['base']['debug']:
            if verbose:
                print('Store in CSV')
            synth_df.to_csv(self.model_dir / pathlib.Path(self.model_name + '_synthetic_data.csv'), index=False)

    def store_dataset_split(self):
        assert self.dataset is not None
        test_stay_ids = []
        val_stay_ids = []

        for key, to_fill in [('validation', val_stay_ids), ('test', test_stay_ids)]:
            for (__, __, __, __, stay_id_batch) in self.data_loaders[key]:
                to_fill.extend(stay_id_batch.tolist())

        stay_id_df = self.dataset.raw_df[['stay_id']].drop_duplicates()
        stay_id_df['split_member'] = 'train'
        stay_id_df.loc[stay_id_df['stay_id'].isin(test_stay_ids), 'split_member'] = 'test'
        stay_id_df.loc[stay_id_df['stay_id'].isin(val_stay_ids), 'split_member'] = 'validation'
        stay_id_df.to_csv(self.model_dir / pathlib.Path(self.model_name + '_dataset_split.csv'), index=False)
    

    def _load_data(self, verbose: bool = True):
        assert self.config_dict is not None

        batch_size = self.config_dict['train']['batch_size']
        stack_mode = self.config_dict['train']['stack_mode']
        imputation_mode = self.config_dict['train']['imputation_mode']
        if self.seed is None:
            self.seed = self.config_dict['base']['seed']
        generator = set_seeds(self.seed)
        self.dataset = HypoDataset(stack_mode=stack_mode, imputation_mode=imputation_mode, verbose=verbose, generator=generator)
        train_data, val_data, test_data = self.dataset.get_data_loaders(batch_size=batch_size)
        self.data_loaders = {
            'train' : train_data,
            'validation' : val_data,
            'test' : test_data
        }

    def _get_validation_synthetic_eval_data(self):
        val_stay_ids_to_cat = [entry[4] for entry in self.data_loaders['validation']]
        val_stay_ids = torch.cat(val_stay_ids_to_cat).numpy()
        orig_dataset = self.dataset.raw_df
        orig_dataset[~self.dataset.mask_df] = np.nan
        val_orig_df = orig_dataset[np.isin(orig_dataset['stay_id'], val_stay_ids)]
        val_orig_df = val_orig_df[val_orig_df['data_reported'] == 1]
        all_features = DISCRETE_FEATURES + list(CLASS_FEATURES.keys())
        for feature in all_features:
            if not val_orig_df[feature].isnull().any():
                val_orig_df[feature] = val_orig_df[feature].replace(-99.0, np.nan)
        result = collections.defaultdict(dict)

        for disc_feature in DISCRETE_FEATURES:
            percs = np.nanpercentile(val_orig_df[disc_feature], np.arange(1,100))
            result[disc_feature]['data'] = percs

        for class_feature in CLASS_FEATURES.keys():
            fracs = val_orig_df[class_feature].value_counts(normalize=True)
            result[class_feature]['data'] = fracs

        for feature in all_features:
            orig_nan_frac = val_orig_df[feature].isna().mean()
            result[feature]['missing_ratio'] = orig_nan_frac

        self.synth_eval_data = result

    def _get_logger(self):
        if self.config_dict is None:
            raise AttributeError('Can only used logger, if config dict is set')
        if self.config_dict['base']['debug']:
            return DummyLogger()
        logger_type = self.config_dict['train']['logger']
        if logger_type == 'wandb':
            arg_dict = WANDB_ARGS
            arg_dict['mode'] = 'online'
            arg_dict['name'] = self.model_name
            arg_dict['config'] = self.config_dict
            return wandb.init(**arg_dict)
        else:
            return JSONLogger(self.model_name, self.config_dict)


if __name__ == '__main__':
    parser = ExperimentArgumentParser()
    args = parser.parse_args()
    argument_dict = parser.get_param_dict(args)
    pipeline = ExperimentPipeline(argument_dict)
    pipeline.run() 
