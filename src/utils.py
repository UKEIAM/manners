import numpy as np
from typing import Dict, List, Any
import torch
import random
from tqdm import tqdm
import pandas as pd
from src.config import DISCRETE_FEATURES, NORMALIZATION_PARAMS_PATH

def set_seeds(seed: int=42):
    generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return generator

def mean_list_dict(list_dict_to_mean: Dict[Any, List]):
    return {key : np.mean(val) for key, val in list_dict_to_mean.items()}

def generate_synthetic_data(model, nof_samples, device = 'cuda', verbose = True, invert_transform = True):
    """Generate synthetic data using a trained variational autoencoder.

    Args:
        model (torch.nn.Module): Trained variational autoencoder model used for generating synthetic data.
        nof_samples (int): Number of synthetic samples to generate.
        device (str, optional): Device to run generation on. Defaults to 'cuda'.
        verbose (bool, optional): Whether to show progress bars and print statements. Defaults to True.
        invert_transform (bool, optional): Whether to invert the normalization transformation 
            on the generated data. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame containing the generated synthetic patient data. If invert_transform, standardization
        and Box-Cox transformation are inverted
    """
    assert model.variational
    batch_size = 64
    chunks = nof_samples // batch_size * [batch_size] + [nof_samples % batch_size]
    to_cat = []
    curr_nof_samples = 0
    if verbose:
        print('Generating synthetic data')
    for chunk in (tqdm(chunks) if verbose else chunks):
        z = torch.randn((chunk, model.bottleneck_dim)).to(device)
        model_result = model.decode(z)
        chunk_df = model_result.to_dataframe(start_stay_id=curr_nof_samples)
        curr_nof_samples += chunk
        to_cat.append(chunk_df)
    final_df = pd.concat(to_cat)
    if invert_transform:
        normalization_df = pd.read_csv(NORMALIZATION_PARAMS_PATH, index_col=0)
        if verbose:
            print('Invert transformation')
        for feature in DISCRETE_FEATURES:
            mean = normalization_df[feature]['mean']
            std = normalization_df[feature]['std']
            lmbd = normalization_df[feature]['boxcox_lambda']
            # unnormalize and revert boxcox transform
            final_df[feature] = (final_df[feature] * std) + mean
            final_df[feature] = np.power((final_df[feature] * lmbd) + 1, (1/lmbd))
        final_df = final_df.round(2)
        for whole_number_feature in ['MAP', 'DBP', 'SBP', 'urine']:
            final_df[whole_number_feature] = final_df[whole_number_feature].round(decimals=0)
    return final_df





