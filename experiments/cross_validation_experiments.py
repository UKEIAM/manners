import datetime
import subprocess
import time
import argparse
import multiprocessing
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.arg_parser import ExperimentArgumentParser
from src.config import MODELS_DIR
from experiment_pipeline import ExperimentPipeline

"""Cross validation experiments for training and evaluating autoencoder models.
This script implements functionality to run Monte-Carlo cross-validation experiments to compare autoencoders with and without MANNERS. 
It supports parallel processing of multiple experiments and logging to either Weights & Biases or local JSON files.
Example:
    Running cross validation with 20 splits:
    $ python cross_validation_experiments.py --cv_name test_cv --nof_splits 20 --logger json
"""

def run_experiment_pipeline(pipeline: ExperimentPipeline):
    print('Start Training ' + pipeline.model_name)
    pipeline.train_model(verbose=False)
    print('Start Evaluation ' + pipeline.model_name)
    if pipeline.model.variational:
        pipeline.generate_synthetic_test_data(load_model=True)
    else:
        pipeline.perform_reconstruction_analysis(load_model=True)
        pipeline.perform_downstream_tasks(load_model=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--cv_name', help='Name of cross-validation run, used for all models as stem', required=True)
    requiredNamed.add_argument('--nof_splits', type=int, help='Number of cross validation splits to run on', required=True)
    requiredNamed.add_argument('--logger', type=str, choices=['wandb', 'json'], help='Log to WandB or to local JSON file', required=True)
    
    parser.add_argument('--overwrite', action='store_true', help='Flag to overwrite existing results with the same cv_name and split')

    args = parser.parse_args()

    cv_name = args.cv_name
    nof_splits = args.nof_splits
    logger = args.logger
    overwrite = args.overwrite

    if not overwrite:
        models_with_cv_name = list(MODELS_DIR.glob(cv_name + '*'))
    else:
        models_with_cv_name = []

    pipelines = []
    for idx in range(nof_splits):
        seed = idx
        stem = cv_name + '_' + str(seed) + '_'
        for imputation_mode in ['fixed', 'mice-linear', 'missforest']:
            for normalization in ['manners-macro', 'manners-micro', 'vanilla']:
                for model_type in ['discriminative', 'generative']:
                    for encode_mask in [True, False]:
                        parser = ExperimentArgumentParser()
                        args = parser.parse_args(['--pipeline', 'train'])
                        config = parser.get_param_dict(args)
                        config['base']['seed'] = seed
                        model_config_name = stem + model_type + '_' + normalization + '_' + imputation_mode + ('_mask' if encode_mask else '_nomask')
                        run_with_config = True
                        if not overwrite and len(models_with_cv_name) > 0:
                            for existing_model in models_with_cv_name:
                                if model_config_name in str(existing_model):
                                    print(f'Model {model_config_name} with config and split already exists, skipping')
                                    run_with_config = False
                                    break
                        if not run_with_config:
                            continue
                        model_name = model_config_name + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                        config['base']['name'] = model_name
                        config['loss']['normalization'] = normalization
                        config['train']['logger'] = logger
                        config['train']['imputation_mode'] = imputation_mode
                        config['train']['encode_mask'] = encode_mask
                        config['base']['model_type'] = model_type

                        pipeline = ExperimentPipeline(fixed_model_name=model_name, config_dict=config)
                        pipelines.append(pipeline)

    with multiprocessing.get_context('spawn').Pool(processes=3) as pool:
        pool.map(run_experiment_pipeline, pipelines, chunksize=1)
    


    