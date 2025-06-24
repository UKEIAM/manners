import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from src.config import DISCRETE_FEATURES, CLASS_FEATURES

class ResultBatch:
    """A class to handle and process batches of model results in PyTorch.
    This class processes and stores model outputs including missing value logits,
    discrete values, ordinal logits and optional latent space variables. It provides
    methods to convert these outputs into class indices and pandas DataFrames.
    Args:
        missing_logits (torch.Tensor): Logits predicting whether values are missing.
        discrete_vals (torch.Tensor): Tensor containing discrete values.
        ordinal_logits (List[torch.Tensor]): List of tensors containing ordinal logits.
        z (Optional[torch.Tensor], optional): Latent space representation. Defaults to None.
        z_mean (Optional[torch.Tensor], optional): Mean of latent space distribution. Defaults to None.
        z_logged_var (Optional[torch.Tensor], optional): Log variance of latent space distribution. Defaults to None.
    """

    def __init__(self, missing_logits: torch.Tensor, discrete_vals: torch.Tensor, ordinal_logits: List[torch.Tensor], z: Optional[torch.Tensor]= None, z_mean: Optional[torch.Tensor]= None,
                 z_logged_var: Optional[torch.Tensor]=None):
        self.missing_logits = missing_logits
        self.discrete_vals = discrete_vals
        self.ordinal_logits = ordinal_logits
        self.z = z
        self.z_mean = z_mean
        self.z_logged_var = z_logged_var
        missing_pred = torch.sigmoid(self.missing_logits)
        self.missing_mask = torch.where(missing_pred < 0.5, False, True)
        self.batch_size = self.missing_logits.shape[0]

    def get_class_indices(self, threshold: float = 0.5):
        to_stack = []
        for idx in range(len(self.ordinal_logits)):
            ordinal_preds = torch.sigmoid(self.ordinal_logits[idx])
            threshold_passed = ordinal_preds >= threshold
            feature_result = threshold_passed.sum(dim=-1).to(dtype=ordinal_preds.dtype)
            to_stack.append(feature_result)
        return torch.stack(to_stack, dim=1)
    
    def to_dataframe(self, start_stay_id: int = 0, stay_ids: Optional[np.ndarray] = None, keep_inactive_values: bool = False) ->pd.DataFrame:
        if stay_ids is not None:
            assert len(stay_ids.shape) == 1
            assert stay_ids.shape[0] == self.batch_size
        if keep_inactive_values:
            torch_vals = torch.cat((self.discrete_vals, self.get_class_indices(), self.missing_mask.to(int)), dim=1)
        else:
            torch_vals = torch.cat((self.discrete_vals, self.get_class_indices()), dim=1)
            torch_vals[~self.missing_mask] = np.nan
        np_vals = torch_vals.cpu().detach().numpy()
        result_array = np.empty((self.batch_size * 48, np_vals.shape[1] + 2))
        if stay_ids is None:
            result_array[:, 0] = np.repeat(np.arange(start_stay_id, start_stay_id + self.batch_size), 48)
        else:
            result_array[:, 0] = np.repeat(stay_ids, 48)
        result_array[:, 1] = np.tile(np.arange(48), self.batch_size)
        for feature_idx in range(np_vals.shape[1]):
            result_array[:,feature_idx + 2] = np_vals[:,feature_idx,:].flatten()
        feature_cols = DISCRETE_FEATURES + list(CLASS_FEATURES.keys())
        if keep_inactive_values:
            all_cols = ['stay_id', 'hour'] + feature_cols + [feature + '_active' for feature in feature_cols]
            int_cols = ['stay_id', 'hour'] + [feature + '_active' for feature in feature_cols] + list(CLASS_FEATURES.keys())
        else:
            all_cols = ['stay_id', 'hour'] + feature_cols
            int_cols = ['stay_id', 'hour'] + list(CLASS_FEATURES.keys())
        result = pd.DataFrame(result_array, columns=all_cols)
        for col in int_cols:
            result[col] = result[col].astype('Int64')
        result['data_reported'] = (result[feature_cols].isnull().sum(axis=1) < len(feature_cols)).astype(int)
        return result

class TargetDataBatch:
    """A class wrapping target data (to be reconstructed) in batches for training.
    Args:
        stacked (torch.Tensor): The stacked tensor containing all target data.
        disc_data (torch.Tensor): Tensor containing discrete target data.
        class_data (List[torch.Tensor]): List of tensors containing ordinal class indices.
        missing_mask (torch.Tensor): Tensor indicating missing values in the data (1 for present, 0 for missing).
        device (str, optional): Device to store the tensors on. Defaults to 'cuda'.
    """

    def __init__(self, stacked: torch.Tensor, disc_data: torch.Tensor, class_data: List[torch.Tensor], missing_mask: torch.Tensor, device: str= 'cuda'):
        self.stacked = stacked.to(device, dtype=torch.float)
        self.disc_data = disc_data.to(device, dtype=torch.float)
        self.class_data = [feature_batch.to(device, dtype=torch.float) for feature_batch in class_data]
        self.missing_mask = missing_mask.to(device, dtype=torch.float)
        self.missing_bool = self.missing_mask.to(bool)
        