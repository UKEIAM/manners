from torch.utils.data import Dataset, DataLoader, random_split
import torch
import time
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.experimental import enable_iterative_imputer   
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from src.config import DISCRETE_FEATURES, CLASS_FEATURES, NORMALIZED_DATASET_PATH

class HypoDataset(Dataset):
    """HypoDataset class loads the hypotension dataset for model training. Missing mask is added and stacked onto variable data. 
    Oridnal class data is standardized to increase model conversion time.
    Args:
        generator (torch.Generator): Random generator for reproducibility.
        imputation_mode (str, optional): Strategy for imputing missing values. 'Fixed' to fill discrete data with mean and class data with median, 
            'mice-linear' for MICE with linear bayesian ridge regression, or 'missforest' for MICE with random forest. Defaults to 'fixed'.
        split_fractions (List[float], optional): Fraction of train, validation and test set, must sum to 1. Defaults to [0.7, 0.1, 0.2].
        mice (bool, optional): Whether to use the MICE imputation technique. Will be fitted to train set and applied to train, validation and test set. Defaults to False.
        verbose (bool, optional): Whether to print progress messages during initialization.
            Defaults to True.
    """

    def __init__(self, generator: torch.Generator, imputation_mode: str = 'fixed', split_fractions: List[float] = [0.7, 0.1, 0.2], verbose: bool = True) -> None:
        super().__init__()
        self.generator = generator
        self.impute_time = None
        imputation_modes = ['fixed', 'mice-linear', 'missforest']
        if imputation_mode not in imputation_modes:
            raise AttributeError('Fill missing mode must be one of "' + '", "'.join(imputation_modes) + '"')
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
        if verbose:
            print(f'Impute missing values with mode {self.imputation_mode}')
        self._impute_missing_values(verbose=verbose)
        if verbose:
            print(f'Imputation took {self.impute_time:.0f} milliseconds')
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
        return data
    
    def _prepare_class_data(self) -> Tuple[List[np.ndarray], np.array, np.array]:
        class_indices = self.raw_df[CLASS_FEATURES].to_numpy().reshape((len(self.stay_ids), 48, len(CLASS_FEATURES)))
        class_indices = class_indices.swapaxes(1, 2)
        
        class_biases = np.empty([1, class_indices.shape[1], 1])
        for idx, class_info in enumerate(CLASS_FEATURES.values()):
            class_biases[:,idx,:] = class_info['mean']

        # missing values are imputed and filled at this point
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
        return np.stack((vals_concat, mask_to_stack), axis=2)
        
    def _impute_missing_values(self, verbose: bool = True) -> None:
        start_time = time.time()
        remaining_stay_ids = set([self.stay_ids[idx] for idx in self.subsets[1].indices + self.subsets[2].indices])
        train_df = self.raw_df[~self.raw_df['stay_id'].isin(remaining_stay_ids)]
        remaining_df = self.raw_df[self.raw_df['stay_id'].isin(remaining_stay_ids)]
        if self.imputation_mode == 'fixed':
            for disc_feature in DISCRETE_FEATURES:
                mean_val = np.round(train_df[disc_feature].mean(), 4)
                train_df[disc_feature].fillna(mean_val, inplace=True)
                remaining_df[disc_feature].fillna(mean_val, inplace=True)
            for class_feature in CLASS_FEATURES.keys():
                median_val = train_df[class_feature].median()
                train_df[class_feature].fillna(median_val, inplace=True)
                remaining_df[class_feature].fillna(median_val, inplace=True)
        elif self.imputation_mode in ['mice-linear', 'missforest']:
            if self.imputation_mode == 'mice-linear':
                estimator = BayesianRidge()
            else:
                estimator = RandomForestRegressor(n_estimators=10, random_state=self.generator.initial_seed())
            train_to_impute = train_df[DISCRETE_FEATURES + list(CLASS_FEATURES.keys())].to_numpy()
            remaining_to_impute = remaining_df[DISCRETE_FEATURES + list(CLASS_FEATURES.keys())].to_numpy()
            imputer = IterativeImputer(estimator=estimator, max_iter=10, random_state=self.generator.initial_seed(), verbose=2 if verbose else 0)
            train_imputed = imputer.fit_transform(train_to_impute)
            remaining_imputed = imputer.transform(remaining_to_impute)
            train_df[DISCRETE_FEATURES + list(CLASS_FEATURES.keys())] = train_imputed
            remaining_df[DISCRETE_FEATURES + list(CLASS_FEATURES.keys())] = remaining_imputed
        else:
            raise NotImplementedError('Imputation mode ' + self.imputation_mode + ' not implemented')
        self.raw_df[~self.raw_df['stay_id'].isin(remaining_stay_ids)] = train_df
        self.raw_df[self.raw_df['stay_id'].isin(remaining_stay_ids)] = remaining_df
        end_time = time.time()
        self.impute_time = (end_time - start_time) * 1000

    def get_data_loaders(self, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get data loaders for train, validation and test set

        Args:
            batch_size (int, optional): Batch size used for training. Defaults to 64.

        Returns:
            Tuple[DataLoader, DataLoader]: Returns triple of dataloaders for train, validation and test set
        """
        
        return (DataLoader(subset, shuffle=True, batch_size=batch_size, generator=self.generator) for subset in self.subsets)
