import torch
from scipy import stats
from torcheval.metrics.functional import multiclass_f1_score
import numpy as np
import collections
from typing import Iterable
from src.utils import generate_synthetic_data, mean_list_dict
from src.config import DISCRETE_FEATURES, CLASS_FEATURES
from src.model import FullyConvAutoEncoder
from src.data_structures import ResultBatch, TargetDataBatch
from src.manners import MANNERS
    
class EvalMetrics:
    @staticmethod
    def evaluate_model(model : FullyConvAutoEncoder, dataloader, device = 'cuda', huber_delta: float = 1.0, normalization: str = 'manners'):
        """
        Evaluates the performance of an autoencodder on a given dataset (always perfomred on validation set in pipeline). 
        Is called after each epoch. Calculates various metrics including Huber error, MAE and MSE, missing classification accuracy, and ordinal classification 
        accuracy/Kendall's tau per feature and meaned for all suitable features.

        Parameters:
        model (FullyConvAutoEncoder): The model to be evaluated.
        dataloader (DataLoader): DataLoader providing the dataset for evaluation.
        device (str, optional): The device to run the evaluation on. Default is 'cuda'. Must be the same as the one used for training.
        huber_delta (float, optional): The delta value for Huber error calculation. Default is 1.0.
        normalization (str, optional): The loss normalization method to be used. Default is 'manners'.

        Returns:
        tuple: A tuple containing two dictionaries:
            - overall_metrics (dict): Dictionary containing overall evaluation metrics.
            - featurewise_metrics (dict): Dictionary containing feature-wise evaluation metrics.
        """
        overall_metrics_lists = collections.defaultdict(lambda : collections.defaultdict(list))
        featurewise_metrics_lists = collections.defaultdict(lambda : collections.defaultdict(list))

        for stacked_batch, disc_batch, class_ordinal_batch, missing_batch, __ in dataloader:
            target = TargetDataBatch(stacked_batch, disc_batch, class_ordinal_batch, missing_batch, device=device)
            model.eval()
            model_result = model(target.stacked)
            if model.variational:
                overall_metrics_lists['vae']['kl_divergence'].append(EvalMetrics.calc_kl_divergence_loss(model_result).item())

            # eval missing classification
            overall_metrics_lists['missing']['cross_entropy'].append(EvalMetrics.calc_missing_bce_loss(model_result, target).item())
            correct = model_result.missing_mask == target.missing_bool
            true_pos_featurewise = (correct & target.missing_bool).sum(dim=(0, 2))
            all_pos_featurewise = target.missing_bool.sum(dim=(0, 2))
            true_neg_featurewise = (correct & ~target.missing_bool).sum(dim=(0, 2))
            all_neg_featurewise = target.missing_bool.shape[0] * target.missing_bool.shape[2] - all_pos_featurewise

            global_sensitivity = true_pos_featurewise.sum() / all_pos_featurewise.sum()
            global_specificity = true_neg_featurewise.sum() / all_neg_featurewise.sum()
            global_accuracy = (true_pos_featurewise.sum() + true_neg_featurewise.sum()) / (torch.numel(target.missing_bool))

            overall_metrics_lists['missing']['accuracy'].append(global_accuracy.item())
            overall_metrics_lists['missing']['sensitivity'].append(global_sensitivity.item())
            overall_metrics_lists['missing']['specificity'].append(global_specificity.item())

            ## eval missing clf on feature level
            all_features = DISCRETE_FEATURES + list(CLASS_FEATURES.keys())
            for idx, feature in enumerate(all_features):
                feature_sensitivity = true_pos_featurewise[idx] / all_pos_featurewise[idx]
                feature_specificity = true_neg_featurewise[idx] / all_neg_featurewise[idx]
                feature_accuracy = (true_pos_featurewise[idx] + true_neg_featurewise[idx]) / (target.missing_bool.shape[0] * target.missing_bool.shape[2])
                featurewise_metrics_lists[feature]['missing_accuracy'].append(feature_accuracy.item())
                featurewise_metrics_lists[feature]['missing_sensitivity'].append(feature_sensitivity.item())
                featurewise_metrics_lists[feature]['missing_specificity'].append(feature_specificity.item())

            # eval discrete features

            ## calculate overall errors
            disc_missing_mask = target.missing_bool[...,:len(DISCRETE_FEATURES),:]
            feature_non_missing_sum = disc_missing_mask.sum(dim=-1)
            feature_has_non_missing = feature_non_missing_sum.to(bool)
            total_non_missing_sum = feature_has_non_missing.sum()

            disc_diff_elewise = (model_result.discrete_vals - target.disc_data) * disc_missing_mask
            mse_error_featurewise = (disc_diff_elewise**2).sum(dim=-1)
            mae_error_featurewise = disc_diff_elewise.abs().sum(dim=-1)
            for key, error_featurewise in [('mse', mse_error_featurewise), ('mae', mae_error_featurewise)]:
                error_featurewise[feature_has_non_missing] = error_featurewise[feature_has_non_missing] / feature_non_missing_sum[feature_has_non_missing]
                total_error = error_featurewise.sum() / total_non_missing_sum
                overall_metrics_lists['discrete'][key].append(total_error.item())
            overall_metrics_lists['discrete']['huber'].append(EvalMetrics.calc_huber_loss(model_result, target, huber_delta, normalization).item())

            ##  calculate mae/mse per feature
            feature_non_missing_batch_sum = feature_has_non_missing.sum(dim=0)
            feature_has_non_missing_batch = feature_non_missing_batch_sum.to(bool)
            for key, error_featurewise in [('mse', mse_error_featurewise), ('mae', mae_error_featurewise)]:
                error_feature_batchwise = error_featurewise.sum(dim=0)
                error_feature_batchwise[feature_has_non_missing_batch] = error_feature_batchwise[feature_has_non_missing_batch] / feature_non_missing_batch_sum[feature_has_non_missing_batch]
                for idx, feature in enumerate(DISCRETE_FEATURES):
                    featurewise_metrics_lists[feature][key].append(error_feature_batchwise[idx].item())

            ## calculate pearson coefficient between output und target per feature
            for idx, feature in enumerate(DISCRETE_FEATURES):
                feature_mask = target.missing_bool[:,idx,:]
                feature_out = model_result.discrete_vals[:,idx,:][feature_mask]
                feature_target = target.disc_data[:,idx,:][feature_mask]
                corr_input = torch.stack((feature_out, feature_target))
                corr_matrix = torch.corrcoef(corr_input)
                featurewise_metrics_lists[feature]['pearson'].append(corr_matrix[0][1].item())

            # eval class features
            class_pred_indices = model_result.get_class_indices()
            ordinal_losses = []
            for idx, (feature, class_info) in enumerate(CLASS_FEATURES.items()):
                feature_ordinal_loss_elementwise = EvalMetrics.calc_ordinal_ce_loss(model_result, target, idx, normalization)
                ordinal_losses.append(feature_ordinal_loss_elementwise)
                feature_missing = target.missing_bool[:,len(DISCRETE_FEATURES)+idx,:]
                manners = MANNERS()
                feature_ordinal_loss = manners(feature_ordinal_loss_elementwise, feature_missing)
                featurewise_metrics_lists[feature]['value_cross_entropy'].append(feature_ordinal_loss.item())
                feature_pred = class_pred_indices[:,idx,:][feature_missing]
                feature_target_indices = (target.class_data[idx].sum(dim=-1))[feature_missing]
                res = stats.kendalltau(feature_pred.cpu().numpy(), feature_target_indices.cpu().numpy())
                featurewise_metrics_lists[feature]['value_kendall_tau'].append(res.statistic)
                featurewise_metrics_lists[feature]['value_f1_score'].append(multiclass_f1_score(feature_pred, feature_target_indices, average='micro', num_classes=class_info['nof_classes']).item())

            combined_ordinal_error = EvalMetrics.combine_ordinal_losses(ordinal_losses, target, normalization)
            overall_metrics_lists['class']['ordinal_cross_entropy'].append(combined_ordinal_error.item())

        overall_metrics = {key : mean_list_dict(val) for key, val in overall_metrics_lists.items()}
        featurewise_metrics = {key : mean_list_dict(val) for key, val in featurewise_metrics_lists.items()}

        return overall_metrics, featurewise_metrics
    
    @staticmethod
    def eval_synth_data(model : FullyConvAutoEncoder, synth_eval_data, device = 'cuda', verbose=False):
        """
        Evaluate synthetic data generated by an autoencoder compared to a validation set.

        Parameters:
        model (FullyConvAutoEncoder): The model used to generate synthetic data.
        synth_eval_data (dict): Dictionary containing the evaluation data for synthetic data (extracted from a validation set).
        device (str, optional): The device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.
        verbose (bool, optional): If True, prints additional information during execution. Default is False.

        Returns:
        tuple: A tuple containing:
            - overall_results (dict): Dictionary with overall evaluation metrics:
                - 'synth_qq_diffs' (float): Mean quantile-quantile differences for discrete features.
                - 'synth_frac_diffs' (float): Mean fractional differences for class features.
                - 'synth_missing_diffs' (float): Mean differences in missing data ratios.
            - featurewise_result (dict): Dictionary with feature-wise evaluation metrics:
                - For each discrete feature:
                    - 'synth_qq_diff' (float): Quantile-quantile difference.
                - For each class feature:
                    - 'synth_frac_diff' (float): Fractional difference.
                - For each feature:
                    - 'synth_missing_diff' (float): Difference in missing data ratio.
        """
        # large number to get reliable quantiles
        nof_samples = 30000
        synth_data = generate_synthetic_data(model, nof_samples, device, verbose, False)
        synth_data = synth_data[synth_data['data_reported'] == 1]
        all_features = DISCRETE_FEATURES + list(CLASS_FEATURES.keys())
        featurewise_result = collections.defaultdict(dict)
        qq_diffs = []
        frac_diffs = []
        missing_diffs = []

        for disc_feature in DISCRETE_FEATURES:
            orig_percs = synth_eval_data[disc_feature]['data']
            synth_percs = np.nanpercentile(synth_data[disc_feature], np.arange(1,100))
            qq_diff = round(np.mean(np.abs(orig_percs - synth_percs)), 3)
            featurewise_result[disc_feature]['synth_qq_diff'] = qq_diff
            qq_diffs.append(qq_diff)

        for class_feature in CLASS_FEATURES.keys():
            orig_fracs = synth_eval_data[class_feature]['data']
            synth_fracs = synth_data[class_feature].value_counts(normalize=True)
            diff = orig_fracs - synth_fracs
            diff[np.isnan(diff)] = orig_fracs[np.isnan(diff)]
            frac_diff = round(diff.abs().mean(), 3)
            featurewise_result[class_feature]['synth_frac_diff'] = frac_diff
            frac_diffs.append(frac_diff)

        for feature in all_features:
            orig_nan_frac = synth_eval_data[feature]['missing_ratio']
            synth_nan_frac = synth_data[feature].isna().mean()
            missing_diff = round(np.absolute(orig_nan_frac - synth_nan_frac), 3)
            featurewise_result[feature]['synth_missing_diff'] = missing_diff
            missing_diffs.append(missing_diff)

        overall_results = {
            'synth_qq_diffs' : round(np.mean(qq_diffs), 3),
            'synth_frac_diffs' : round(np.mean(frac_diffs), 3),
            'synth_missing_diffs' : round(np.mean(missing_diffs), 3)
            }
        return overall_results, featurewise_result

    @staticmethod
    def calc_missing_bce_loss(model_result: ResultBatch, target: TargetDataBatch) -> torch.Tensor:
        ce_loss_operator = torch.nn.BCEWithLogitsLoss(reduction='none')
        ce_loss_elementwise = ce_loss_operator(model_result.missing_logits, target.missing_mask)
        nof_time_steps = 48
        weights = torch.ones_like(ce_loss_elementwise)
        non_missing_sums = torch.empty_like(ce_loss_elementwise)
        non_missing_sums[...,:] = target.missing_mask.sum(dim=-1, keepdim=True)
        feature_has_both_classes = (0 < non_missing_sums) & (non_missing_sums < nof_time_steps)
        weights[feature_has_both_classes & target.missing_bool] = nof_time_steps / (2 * non_missing_sums[feature_has_both_classes & target.missing_bool])
        weights[feature_has_both_classes & ~target.missing_bool] = nof_time_steps / (2 * (nof_time_steps - non_missing_sums[feature_has_both_classes & ~target.missing_bool]))
        return (ce_loss_elementwise * weights).mean(dim=-1).mean()
    
    @staticmethod
    def calc_huber_loss(model_result: ResultBatch, target: TargetDataBatch, huber_delta: float = 1.0, normalization: str = 'manners') -> torch.Tensor:
        if normalization not in ['vanilla', 'manners']:
            raise AttributeError('Normalization mode must be "manners" or "vanilla"')
        huber_loss_operator = torch.nn.HuberLoss(reduction='none', delta=huber_delta)
        huber_loss = huber_loss_operator(model_result.discrete_vals, target.disc_data)
        if normalization == 'vanilla':
            return huber_loss.mean()
        disc_missing_mask = target.missing_bool[...,:len(DISCRETE_FEATURES), :]
        manners = MANNERS()
        return manners(huber_loss, disc_missing_mask)
        
    @staticmethod
    def calc_ordinal_ce_loss(model_result: ResultBatch, target: TargetDataBatch, feature_idx: int, normalization: str = 'manners') -> torch.Tensor:
        if normalization not in ['vanilla', 'manners']:
            raise AttributeError('Normalization mode must be "manners" or "vanilla"')
        nof_classes = list(CLASS_FEATURES.values())[feature_idx]['nof_classes']
        out_logits = model_result.ordinal_logits[feature_idx]
        target_one_hot= target.class_data[feature_idx]
        assert out_logits.shape[-1] == nof_classes -1
        assert target_one_hot.shape[-1] == nof_classes - 1
        target_missing_mask = target.missing_bool[:,len(DISCRETE_FEATURES)+feature_idx,:]
        target_missing_inflated = target_missing_mask[...,:,None]
        target_one_hot_mask = target_one_hot.to(bool)
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        ordinal_loss = loss_func(out_logits, target_one_hot)
        if normalization == 'manners':
            cases_to_consider = target_one_hot * target_missing_inflated
        else:
            cases_to_consider = target_one_hot

        nof_pos_cases_per_sample_and_class = cases_to_consider.sum(dim=-2, keepdim=True).to(ordinal_loss.dtype)
        nof_pos_cases_inflated = nof_pos_cases_per_sample_and_class.expand_as(ordinal_loss)

        if normalization == 'manners':
            non_missing_sums = target_missing_mask.sum(dim=-1).to(ordinal_loss.dtype)
            non_missing_sums_inflated = non_missing_sums[:,None,None]
            non_missing_sums_inflated = non_missing_sums_inflated.expand_as(ordinal_loss)
            class_has_both_cases = (0 < nof_pos_cases_inflated) & (nof_pos_cases_inflated < non_missing_sums_inflated)
            pos_lookup = target_missing_inflated & class_has_both_cases & target_one_hot_mask
            neg_lookup = target_missing_inflated & class_has_both_cases & ~target_one_hot_mask
            weights = torch.ones_like(ordinal_loss)
            weights[pos_lookup] = non_missing_sums_inflated[pos_lookup] / (2*nof_pos_cases_inflated[pos_lookup])
            weights[neg_lookup] = non_missing_sums_inflated[neg_lookup] / (2*(non_missing_sums_inflated[neg_lookup] - nof_pos_cases_inflated[neg_lookup]))
        else:
            time_steps = target_missing_mask.shape[-1]
            class_has_both_cases = (0 < nof_pos_cases_inflated) & (nof_pos_cases_inflated < time_steps)
            pos_lookup = class_has_both_cases & target_one_hot_mask
            neg_lookup = class_has_both_cases & ~target_one_hot_mask
            weights = torch.ones_like(ordinal_loss)
            weights[pos_lookup] = time_steps / (2*nof_pos_cases_inflated[pos_lookup])
            weights[neg_lookup] = time_steps / (2*(time_steps - nof_pos_cases_inflated[neg_lookup]))
        ordinal_loss = ordinal_loss * weights
        result = ordinal_loss.mean(dim=-1)        
        return result
    
    @staticmethod
    def combine_ordinal_losses(elementwise_ordinal_losses: Iterable[torch.Tensor], target: TargetDataBatch, normalization: str = 'manners') -> torch.Tensor:
        if normalization not in ['vanilla', 'manners']:
            raise AttributeError('Normalization mode must be "manners" or "vanilla"')
        stacked_ordinal_loss = torch.stack(elementwise_ordinal_losses, dim=1)
        if normalization == 'vanilla':
            return stacked_ordinal_loss.mean()

        ordinal_missing = target.missing_bool[:,len(DISCRETE_FEATURES):, :]
        manners = MANNERS()
        return manners(stacked_ordinal_loss, ordinal_missing)
            
    @staticmethod
    def calc_kl_divergence_loss(model_result: ResultBatch, normalize: bool = True) -> torch.Tensor:
        assert model_result.z_mean is not None
        assert model_result.z_logged_var is not None
        result_elementwise = -0.5 * (1 + model_result.z_logged_var - model_result.z_mean.pow(2) - model_result.z_logged_var.exp())
        func = torch.mean if normalize else torch.sum
        result_samplewise = func(result_elementwise, dim=1)
        result_batchwise = result_samplewise.mean()
        return result_batchwise
