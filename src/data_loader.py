from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.experimental import enable_iterative_imputer   
from sklearn.impute import IterativeImputer
from src.config import DISCRETE_FEATURES, CLASS_FEATURES, NORMALIZED_DATASET_PATH

class HypoDataset(Dataset):
    """HypoDataset class loads the hypotension dataset for model training. Missing mask is added and stacked onto variable data. 
    Oridnal class data is standardized to increase model conversion time.
    Args:
        generator (torch.Generator): Random generator for reproducibility.
        stack_mode (str, optional): Format for stacking the missing mask onto the data. '1D' as separate channles, '2D' as new dimension. 
            Defaults to '2D'.
        imputation_mode (str, optional): Strategy for imputing missing values. 'Fixed' to fill discrete data with mean and class data with median, 
            'random' for filling with random values from standard gaussian. Defaults to 'fixed'.
        split_fractions (List[float], optional): Fraction of train, validation and test set, must sum to 1. Defaults to [0.7, 0.1, 0.2].
        mice (bool, optional): Whether to use the MICE imputation technique. Will be fitted to train set and applied to train, validation and test set. Defaults to False.
        verbose (bool, optional): Whether to print progress messages during initialization.
            Defaults to True.
    """

    def __init__(self, generator: torch.Generator, stack_mode: str = '2D', imputation_mode: str = 'fixed', split_fractions: List[float] = [0.7, 0.1, 0.2], verbose: bool = True) -> None:
        super().__init__()
        self.generator = generator
        stack_modes = ['1D', '2D']
        if stack_mode not in stack_modes:
            raise AttributeError('Stack mode must be one of "' + '", "'.join(stack_modes))
        imputation_modes = ['fixed', 'random', 'mice']
        if imputation_mode not in imputation_modes:
            raise AttributeError('Fill missing mode must be one of "' + '", "'.join(imputation_modes))
        self.stack_mode = stack_mode
        self.imputation_mode = imputation_mode
        if verbose:
            print('Read data from CSV')
        self.raw_df = pd.read_csv(NORMALIZED_DATASET_PATH)
        self.mask_df = ~np.isnan(self.raw_df)
        stay_ids = self.raw_df['stay_id'].unique()
        self.stay_ids = {idx : stay_ids[idx] for idx in range(len(stay_ids))}
        if verbose:
            print('Generate data split')
        self.subsets = self._generate_split_indices(split_fractions=split_fractions)
        self.missing_mask = self.mask_df[DISCRETE_FEATURES + list(CLASS_FEATURES.keys())].to_numpy().reshape((len(self.stay_ids), 48, -1)).swapaxes(1, 2).astype(int)
        if self.imputation_mode == 'mice':
            if verbose:
                print('Impute missing values with MICE')
            self._perform_mice_imputation(verbose=verbose)
        if verbose:
            print('Transform discrete data')
        self.disc_data = self._prepare_discrete_data()
        if verbose:
            print('Transform class data')
        ordinal_one_hot, class_data = self._prepare_class_data()
        self.ordinal_one_hot = ordinal_one_hot
        self.stacked = self._stack_data(self.disc_data, class_data)
        if verbose:
            print('Finished data loading')

    def __len__(self) -> int:
        return len(self.stay_ids)
    
    def __getitem__(self, index) -> Tuple[np.array, np.array, List[np.array], np.array, int]:
        return (self.stacked[index], self.disc_data[index], [ordinal[index] for ordinal in self.ordinal_one_hot],
                self.missing_mask[index], self.stay_ids[index])
    
    def _generate_split_indices(self, split_fractions: List[float]) -> List[torch.utils.data.Subset]:
        train_size = int(np.floor(len(self) * split_fractions[0]))
        val_size = int(np.floor(len(self) * split_fractions[1]))
        lengths = [train_size, val_size, len(self) - train_size - val_size]
        return random_split(self, lengths, generator=self.generator)

    def _prepare_discrete_data(self) -> Tuple[np.array, np.array]:
        data = self.raw_df[DISCRETE_FEATURES].to_numpy().reshape((len(self.stay_ids), 48, len(DISCRETE_FEATURES)))
        data = data.swapaxes(1, 2)
        disc_missing = self.missing_mask[:, :len(DISCRETE_FEATURES), :]
        missing_bool = disc_missing.astype(bool)
        if self.imputation_mode == 'fixed':
            data[~missing_bool] = 0.0
        elif self.imputation_mode == 'random':
            nof_missings = (~missing_bool).sum()
            data[~missing_bool] = np.random.normal(size=nof_missings)
        return data
    
    def _prepare_class_data(self) -> Tuple[List[np.ndarray], np.array, np.array]:
        class_indices = self.raw_df[CLASS_FEATURES].to_numpy().reshape((len(self.stay_ids), 48, len(CLASS_FEATURES)))
        class_indices = class_indices.swapaxes(1, 2)
        class_missing = self.missing_mask[:, len(DISCRETE_FEATURES):, :]
        missing_bool = class_missing.astype(bool)
        class_biases = np.empty([1, class_indices.shape[1], 1])
        for idx, class_info in enumerate(CLASS_FEATURES.values()):
            class_biases[:,idx,:] = class_info['mean']
            if self.imputation_mode == 'fixed':
                class_indices[:,idx,:][~missing_bool[:,idx,:]] = class_info['median']
            elif self.imputation_mode == 'random':
                nof_missings = (~missing_bool[:,idx,:]).sum()
                class_indices[:,idx,:][~missing_bool[:,idx,:]] = np.random.randint(class_info['nof_classes'], size=nof_missings)

        # NaNs are filled at this point
        class_indices = class_indices.astype(int)
        ordinal_one_hot = []
        class_normalizer = np.empty([1, class_indices.shape[1], 1])
        for idx, class_info in enumerate(CLASS_FEATURES.values()):
            nof_classes = class_info['nof_classes']
            class_normalizer[:,idx,:] = class_info['average distance']
            feature_indices = class_indices[:,idx,:]
            feature_ordinal_one_hot = np.empty([feature_indices.shape[0], feature_indices.shape[1], nof_classes-1])
            for class_idx in range(nof_classes-1):
                feature_ordinal_one_hot[...,class_idx] = (feature_indices > class_idx).astype(int)
            ordinal_one_hot.append(feature_ordinal_one_hot)

        class_normalized = (class_indices - class_biases) / class_normalizer
        return ordinal_one_hot, class_normalized
    
    def _stack_data(self, discrete_data: np.array, class_data: np.array, missing_class : int = -1)-> np.array:
        mask_to_stack = np.where(self.missing_mask == 1, 1, missing_class)
        vals_concat = np.concatenate((discrete_data, class_data), axis=1)
        assert mask_to_stack.shape == vals_concat.shape
        if self.stack_mode == '2D':
            return np.stack((vals_concat, mask_to_stack), axis=2)
        else:
            return np.concatenate((vals_concat, mask_to_stack), axis=1)
        
    def _perform_mice_imputation(self, verbose: bool = True) -> None:
        remaining_stay_ids = set([self.stay_ids[idx] for idx in self.subsets[1].indices + self.subsets[2].indices])
        train_df = self.raw_df[~self.raw_df['stay_id'].isin(remaining_stay_ids)]
        remaining_df = self.raw_df[self.raw_df['stay_id'].isin(remaining_stay_ids)]
        train_to_impute = train_df[DISCRETE_FEATURES + list(CLASS_FEATURES.keys())].to_numpy()
        remaining_to_impute = remaining_df[DISCRETE_FEATURES + list(CLASS_FEATURES.keys())].to_numpy()
        imputer = IterativeImputer(max_iter=10, random_state=self.generator.initial_seed(), verbose=2 if verbose else 0)
        train_imputed = imputer.fit_transform(train_to_impute)
        remaining_imputed = imputer.transform(remaining_to_impute)
        train_df[DISCRETE_FEATURES + list(CLASS_FEATURES.keys())] = train_imputed
        remaining_df[DISCRETE_FEATURES + list(CLASS_FEATURES.keys())] = remaining_imputed
        self.raw_df[~self.raw_df['stay_id'].isin(remaining_stay_ids)] = train_df
        self.raw_df[self.raw_df['stay_id'].isin(remaining_stay_ids)] = remaining_df
    
    def get_data_loaders(self, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get data loaders for train, validation and test set

        Args:
            batch_size (int, optional): Batch size used for training. Defaults to 64.

        Returns:
            Tuple[DataLoader, DataLoader]: Returns triple of dataloaders for train, validation and test set
        """
        
        return (DataLoader(subset, shuffle=True, batch_size=batch_size, generator=self.generator) for subset in self.subsets)
