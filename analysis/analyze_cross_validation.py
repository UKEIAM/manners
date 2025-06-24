import pathlib
import pandas as pd
import numpy as np
import time
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import wasserstein_distance, pearsonr, kendalltau, wilcoxon
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import collections
from typing import Dict, Tuple, List
import re
import umap
import itertools
import wandb
import argparse
import torch
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import ANALYSIS_PATH, MODELS_DIR, DISCRETE_FEATURES, CLASS_FEATURES, NORMALIZED_DATASET_PATH, LABEL_PATH, TREATMENT_FEATURES, WANDB_ARGS
from src.utils import set_seeds
from src.data_loader import HypoDataset
from src.model import FullyConvAutoEncoder
from src.data_structures import TargetDataBatch

"""Analyze cross-validation results for autoencoders with and without MANNERS.
This script analyzes the performance of autoencoders with MANNERS by evaluating survival classification accuracy, 
test set reconstruction quality, and synthetic data generation capabilities. 
The analysis includes calculating metrics and generating visualization plots.
The script handles three main analysis tasks:
1. Downstream survival classification and length-of-stay regression evaluation using balanced accuracy and mean absolute error across different missing rate bins
2. Test set reconstruction evaluation using MAE and correlation coefficients
3. Synthetic data evaluation using Wasserstein distance and mean class ratio differences
Example:
    $ python analyze_cross_validation.py --cv_name test_cv --to_compare manners,mice, manners,fixed, vanilla,mice, vanilla,fixed
"""

CLASS_FEATURE_LIST = list(CLASS_FEATURES.keys())

DISPLAY_NAMES = {feature: (feature if feature not in ['urine', 'serum_creatinine'] else 'urine output' if feature == 'urine' else 'creatinine') for feature in DISCRETE_FEATURES + CLASS_FEATURE_LIST}

COLOR_MAP = {
    'manners' : {
        'mice' : 'blue',
        'fixed' : 'turquoise',
    },
    'vanilla' : {
        'mice' : 'red',
        'fixed' : 'orange',
    }
}

### Utils and data loading

def to_percent_string(x: float) ->str:
    return f'{x * 100:.1f}%'

def get_evaluated_models(full_hypo_df: pd.DataFrame, cv_name: str) -> Tuple[Dict, Dict]:
    variational_models = {}
    ae_models = {}
    for model_dir in tqdm(list(MODELS_DIR.glob(cv_name + '*'))):
        model_name = model_dir.stem
        variational = 'vae' in model_name
        variant = 'vanilla' if 'vanilla' in model_name else 'manners'
        imputation_mode = 'mice' if 'mice' in model_name else 'fixed'
        # split comes right after the cv_name in the model name
        split = re.search(r'(\d+)', model_name.replace(cv_name + '_', '')).group(0)
        if variational:
            synth_results = model_dir / pathlib.Path(model_name + '_synthetic_data.csv')
            if not synth_results.exists():
                continue
            model_data = {
                'synth_data': pd.read_csv(synth_results),
                'variant': variant,
                'imputation_mode': imputation_mode,
                'split': split
            }
            to_add_to = variational_models
        else:
            downstream_results = model_dir / pathlib.Path(model_name + '_downstream_results.csv')
            if not downstream_results.exists():
                continue
            recon_results = model_dir / pathlib.Path(model_name + '_test_reconstructions.csv')
            if not recon_results.exists():
                continue
            model_data = {
                'downstream_data': pd.read_csv(downstream_results),
                'recon_data' : pd.read_csv(recon_results),
                'variant': variant,
                'imputation_mode': imputation_mode,
                'split': split
            }
            to_add_to = ae_models
        split_df = pd.read_csv(model_dir / pathlib.Path(model_name + '_dataset_split.csv'))
        test_df = split_df[split_df['split_member'] == 'test']
        test_df = test_df[['stay_id']].merge(full_hypo_df, on='stay_id')
        model_data['test_data'] = test_df
        to_add_to[model_name] = model_data
    print('Got ' + str(len(ae_models)) + ' evaluated autoencoder models')
    print('Got ' + str(len(variational_models)) + ' evaluated variational autoencoder models')
    return ae_models, variational_models

### Downstream experiments

def generate_per_model_downstream_performance_data(model_results: Dict, stay_mr_df: pd.DataFrame, nof_bins: int) -> pd.DataFrame:
    df_data = collections.defaultdict(list)
    for model_name, model_data in tqdm(model_results.items()):
        all_downstream_data = model_data['downstream_data']
        for bin in range(-1,nof_bins):
            df_data['model_name'].append(model_name)
            df_data['variant'].append(model_data['variant'])
            df_data['imputation_mode'].append(model_data['imputation_mode'])
            df_data['split'].append(model_data['split'])
            df_data['bin'].append(bin)
            if bin > -1:
                bin_ids = stay_mr_df[stay_mr_df['treatment_mr_bin'] == bin]['stay_id']
                downstream_data = all_downstream_data[all_downstream_data['stay_id'].isin(bin_ids)]
            else:
                downstream_data = all_downstream_data
            b_acc = balanced_accuracy_score(downstream_data['survival48h'], downstream_data['survival_prediction'])
            mae = np.mean(np.abs(downstream_data['los_prediction'] - downstream_data['los']))
            df_data['survival_balanced_accuracy'].append(b_acc)
            df_data['los_mae'].append(mae)
    return pd.DataFrame(df_data)

def assign_missing_bin(val, bin_edges):
    # last bin value is 1.0
    for idx, edge in enumerate(bin_edges):
        if val <= edge:
            return idx
        
def calculate_treatment_missing_rates_per_stay(full_hypo_df: pd.DataFrame) -> pd.DataFrame:
    label_df = pd.read_csv(LABEL_PATH)
    grouped_by_stay = full_hypo_df.groupby(['stay_id'])
    missing_rates = grouped_by_stay[TREATMENT_FEATURES].agg(lambda x: x.isna().mean()).mean(axis=1).round(4)
    stay_mr_df = pd.DataFrame(missing_rates).reset_index()
    stay_mr_df.columns = ['stay_id', 'treatment_mr']
    stay_mr_df = stay_mr_df.merge(label_df[['stay_id', 'survival48h', 'los']], on='stay_id')
    return stay_mr_df

def get_downstream_metadata(stay_mr_df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float, List[float], List[float], List[str]]:
    bin_edges = [0.87, 0.97, 1.0]
    intervals = ["TMR <= 87%", "87% < TMR <= 97%", "TMR > 97%"]        
    stay_mr_df['treatment_mr_bin'] = stay_mr_df['treatment_mr'].apply(lambda x: assign_missing_bin(x, bin_edges))
    stay_mr_df.sort_values(by='stay_id', inplace=True)
    total_survival_rate = stay_mr_df['survival48h'].mean()
    total_mean_los = stay_mr_df['los'].mean()

    bin_survival_rates = []
    bin_mean_los = []
    for idx in range(len(bin_edges)):
        bin_df = stay_mr_df[stay_mr_df['treatment_mr_bin'] == idx]
        bin_survival_rates.append(bin_df['survival48h'].mean())
        bin_mean_los.append(bin_df['los'].mean())
    return stay_mr_df, total_survival_rate, total_mean_los, bin_survival_rates, bin_mean_los, intervals


def get_downstream_tasks_summary_table(per_model_df: pd.DataFrame, to_compare: List[Tuple[str, str]], intervals: List[str]) -> pd.DataFrame:
    split_model_counts = collections.defaultdict(int)
    compare_grouped = per_model_df.groupby(['variant', 'imputation_mode'])
    for comp_pair in to_compare:
        if comp_pair not in compare_grouped.groups:
            print(f"Pair {comp_pair} not found in downstream data, not generating summary table")
            return pd.DataFrame()
        for split in compare_grouped.get_group(comp_pair)['split'].unique():
            split_model_counts[split] += 1
    finished_splits = [split for split, count in split_model_counts.items() if count == len(to_compare)]
    if len(finished_splits) == 0:
        print('No splits finished, not generating summary table')
        return pd.DataFrame()
    print(f"Generating downstream summary for {len(finished_splits)} splits")
    per_model_df = per_model_df[per_model_df['split'].isin(finished_splits)]

    nof_bins = len(intervals)
    total_results_data = collections.defaultdict(list)
    for bin in range(-1, nof_bins):
        bin_data = per_model_df[per_model_df['bin'] == bin]
        total_results_data['bin'].append(bin)
        total_results_data['interval'].append('all' if bin == -1 else intervals[bin])
        total_results_data['survival_rate'].append(total_survival_rate if bin == -1 else bin_survival_rates[bin])
        total_results_data['los'].append(total_mean_los if bin == -1 else bin_mean_los[bin])
        binned_groups = bin_data.groupby(['variant', 'imputation_mode'])
        metric_means = collections.defaultdict(dict)
        for metric_str, metric_key in [('los', 'los_mae'), ('survival', 'survival_balanced_accuracy')]:
            for variant, imputation_mode in to_compare:
                metric_mean = np.round(binned_groups.get_group((variant, imputation_mode))[metric_key].mean(), 3)
                total_results_data[f'{metric_str}_{variant}_{imputation_mode}'].append(metric_mean)
                metric_means[(metric_str, metric_key)][(variant, imputation_mode)] = metric_mean
        for (metric_str, metric_key), mean_data in metric_means.items():
            for first_pair, second_pair in sorted(list(itertools.combinations(mean_data.keys(), 2))):
                first_mean = mean_data[first_pair]
                second_mean = mean_data[second_pair]
                if first_mean >= second_mean:
                    larger = first_pair
                    smaller = second_pair
                else:
                    larger = second_pair
                    smaller = first_pair
                better = larger if metric_str == 'survival' else smaller
                worse = smaller if metric_str == 'survival' else larger
                better_scores = binned_groups.get_group(better)[['split', metric_key]]
                worse_scores = binned_groups.get_group(worse)[['split', metric_key]]
                merged_scores = pd.merge(better_scores, worse_scores, on='split', suffixes=('_better', '_worse'))
                diff = merged_scores[metric_key + '_better'] - merged_scores[metric_key + '_worse']
                alternative = 'greater' if metric_str == 'survival' else 'less'
                vs_str = f"{'-'.join(first_pair)}_vs_{'-'.join(second_pair)}"
                wilcoxon_res = wilcoxon(x=diff, alternative=alternative)
                total_results_data[f'{metric_str}_{vs_str}_better'].append('-'.join(better))
                total_results_data[f'{metric_str}_{vs_str}_wilcoxon_stat'].append(wilcoxon_res.statistic)
                total_results_data[f'{metric_str}_{vs_str}_p_value'].append(wilcoxon_res.pvalue)
    result = pd.DataFrame(total_results_data)
    result = result.round(4)
    return result

def calculate_umap_embeddings(per_model_df: pd.DataFrame, stay_mr_df: pd.DataFrame) -> pd.DataFrame:
    best_manners = per_model_df[(per_model_df['variant'] == 'manners') & (per_model_df['bin'] == -1)].sort_values('survival_balanced_accuracy', ascending=False).iloc[0]
    best_vanilla = per_model_df[(per_model_df['variant'] == 'vanilla') & (per_model_df['bin'] == -1)].sort_values('survival_balanced_accuracy', ascending=False).iloc[0]
    print(best_manners)
    print(best_vanilla)
    dfs_to_cat = []
    for model in [best_manners, best_vanilla]:
        imputation_mode = model['imputation_mode']
        seed = model['split']
        generator = set_seeds(seed=seed)
        dataset = HypoDataset(generator=generator, imputation_mode=imputation_mode, verbose=False)
        __, __, test = dataset.get_data_loaders(batch_size=64)
        variant = model['variant']
        model_name = model['model_name']
        to_cat = []
        stay_ids_cat = []
        model_dir =  MODELS_DIR / pathlib.Path(model_name)
        models = list(model_dir.glob('*.pt'))
        if len(models) != 1:
            raise AttributeError('Directory ' + str(model_dir) + ' must contain exactly one weights file')
        api = wandb.Api(overrides=WANDB_ARGS)
        runs = api.runs(filters={"display_name" : model_name})
        if len(runs) == 0:
            raise AttributeError('Model "' + model_name + '" not found in wandb API')

        run = next(runs)
        config_dict = run.config
        ae_config = config_dict['auto encoder']
        stack_mode = config_dict['train']['stack_mode']
        device='cuda'
        model = FullyConvAutoEncoder(**ae_config, stack_mode=stack_mode).to(device)
        model_path = models[0]
        model.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            for (stacked_batch, disc_batch, class_ordinal_batch, missing_batch, stay_id_batch) in tqdm(test):
                target_batch = TargetDataBatch(stacked_batch, disc_batch, class_ordinal_batch, missing_batch, device=device)
                model.eval()
                z = model.encode(target_batch.stacked)
                to_cat.append(z)
                stay_ids_cat.append(stay_id_batch)
        encodings = torch.cat(to_cat, dim=0).cpu().numpy()
        stay_ids = torch.cat(stay_ids_cat, dim=0).cpu().numpy()
        proj = umap.UMAP(n_neighbors=5, random_state=0).fit_transform(encodings)
        df = pd.DataFrame(proj, columns=['x', 'y'])
        df['stay_id'] = stay_ids
        df['variant'] = variant
        df['imputation_mode'] = imputation_mode
        df['model_name'] = model_name
        df['split'] = seed
        df = df.merge(stay_mr_df[['stay_id', 'treatment_mr', 'survival48h', 'los']], on='stay_id')
        dfs_to_cat.append(df)
    all_df = pd.concat(dfs_to_cat, axis=0)
    return all_df

def plot_umap_results(umap_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=2, column_titles=["MANNERS", "Vanilla"], horizontal_spacing=0.05, vertical_spacing=0.01)
    # Set column title font size
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=20)
    los_colorbar = dict(lenmode='fraction', len=0.25, y=0.83, tickfont=dict(size=14))
    mr_colorbar = dict(lenmode='fraction', len=0.25, y=0.16, tickfont=dict(size=14))
    for idx, variant in enumerate(['manners', 'vanilla']):
        filtered_df = umap_df[umap_df['variant'] == variant]
        fig.add_trace(go.Scatter(x=filtered_df['x'], y=filtered_df['y'], mode='markers', marker=dict(color=filtered_df['los'], colorscale='Viridis', colorbar=los_colorbar if idx==0 else None, cmax=5.0, cmin=0.2), showlegend=False, opacity=0.7), row=1, col=idx + 1)
        survived_df = filtered_df[filtered_df['survival48h'] == 1]
        not_survived_df = filtered_df[filtered_df['survival48h'] == 0]
        fig.add_trace(go.Scatter(x=survived_df['x'], y=survived_df['y'], mode='markers', marker=dict(color="#bddf26"), name="True", showlegend=idx==0, opacity=0.7), row=2, col=idx + 1)
        fig.add_trace(go.Scatter(x=not_survived_df['x'], y=not_survived_df['y'], mode='markers', marker=dict(color="#2e6f8e"), name="False", showlegend=idx==0, opacity=0.7), row=2, col=idx + 1)
        fig.add_trace(go.Scatter(x=filtered_df['x'], y=filtered_df['y'], mode='markers', marker=dict(color=filtered_df['treatment_mr'], colorscale='Viridis', colorbar=mr_colorbar if idx==0 else None), showlegend=False, opacity=0.7), row=3, col=idx + 1)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # Add row titles as y-axis labels on the left
    fig.update_yaxes(title_text="Length of stay", title_standoff=5, row=1, col=1, titlefont=dict(size=20))
    fig.update_yaxes(title_text="Survival", title_standoff=5, row=2, col=1, titlefont=dict(size=20))  
    fig.update_yaxes(title_text="Treatment missing rate", title_standoff=5, row=3, col=1, titlefont=dict(size=20))
    fig.update_layout(width=1600, height=2000, legend=dict(title="Survival", orientation="v", x=1.0, y=0.645, font=dict(size=15)), margin=dict(l=35, r=5, t=25, b=10))
    return fig

### Test set reconstruction

def evaluate_reconstruction_per_model(model_results: Dict) -> pd.DataFrame:
    result_data = collections.defaultdict(list)
    for model_name, model_data in tqdm(model_results.items()):
        all_features = DISCRETE_FEATURES + CLASS_FEATURE_LIST
        cols_to_merge = ['stay_id', 'hour'] + all_features + [feature + '_active' for feature in all_features]
        recon_data = model_data['recon_data'][cols_to_merge]
        test_data = model_data['test_data']
        renamed_cols = {feature: feature + '_recon' for feature in all_features}
        merged = recon_data.rename(columns=renamed_cols).merge(test_data, on=['stay_id', 'hour'])
        for idx, feature in enumerate(all_features):
            feature_active_df = merged[~np.isnan(merged[feature])]
            recon_target = feature_active_df[feature]
            recon_pred = feature_active_df[feature + '_recon']
            mae = np.abs(recon_target - recon_pred).mean()
            if idx < len(DISCRETE_FEATURES):
                corr_coeff = pearsonr(recon_target, recon_pred).statistic
            else:
                corr_coeff = kendalltau(recon_target, recon_pred).statistic
            mae = np.round(mae, 4)
            corr_coeff = np.round(corr_coeff, 4)
            result_data['model_name'].append(model_name)
            result_data['variant'].append(model_data['variant'])
            result_data['imputation_mode'].append(model_data['imputation_mode'])
            result_data['feature'].append(feature)
            result_data['missing_rate'].append(np.isnan(merged[feature]).mean().round(4))
            result_data['mae'].append(mae)
            result_data['corr_coeff'].append(corr_coeff)
            orig_active = ~np.isnan(merged[feature])
            recon_active = merged[feature + '_active']
            missing_acc = balanced_accuracy_score(orig_active, recon_active)
            result_data['missing_accuracy'].append(missing_acc)
    return pd.DataFrame(result_data)

def get_plot_name_str(variant: str, imputation_mode: str) -> str:
    imput_str = 'MICE' if imputation_mode == 'mice' else 'fixed'
    if variant == 'manners':
        return f'MANNERS ({imput_str})'
    elif variant == 'vanilla':
        return f'Vanilla ({imput_str})'
    else:
        raise ValueError(f"Unknown variant: {variant}")

def format_split_plot(fig: go.Figure, missing_rates: Dict) -> go.Figure:
    discrete_missing_order = sorted(DISCRETE_FEATURES, key=lambda x: missing_rates[x])
    discrete_xticks = [f'{DISPLAY_NAMES[feature]}<br>{np.round((missing_rates[feature])*100, 1)}%' for feature in discrete_missing_order]
    class_missing_order = sorted(CLASS_FEATURE_LIST, key=lambda x: missing_rates[x])
    class_xticks = [f'{DISPLAY_NAMES[feature]}<br>{np.round((missing_rates[feature])*100, 1)}%' for feature in class_missing_order]
    fig.update_traces(boxpoints=False)
    fig.update_layout(boxmode='group', boxgap=0.15, boxgroupgap=0.1)
    fig.update_xaxes(categoryorder='array', categoryarray=discrete_missing_order, ticktext=discrete_xticks, tickvals=np.arange(len(discrete_xticks)), row=1, col=1)
    fig.update_xaxes(categoryorder='array', categoryarray=class_missing_order, ticktext=class_xticks, tickvals=np.arange(len(class_xticks)), row=1, col=2)
    fig.update_layout(width=1600, height=550, showlegend=True, legend=dict(orientation="h", x=0.25, y=-0.20, font=dict(size=18)))
    for col in range(1, 3):
        fig.update_yaxes(tickfont=dict(size=18), title_font=dict(size=20), title_standoff = 5, row=1, col=col)
        fig.update_xaxes(tickfont=dict(size=18), row=1, col=col)
    fig.layout.annotations[0].update(x=-0.03)
    fig.layout.annotations[1].update(x=0.675)
    fig.add_annotation(text="Variable with missing rate",x=0.5,y=-0.2,xref="paper",yref="paper",showarrow=False)
    fig.update_annotations(font_size=20)
    fig.update_layout(margin=dict(l=60, r=10, t=25, b=0))
    return fig

def format_mask_plot(fig: go.Figure, missing_rates: Dict) -> go.Figure:
    missing_order = sorted(list(missing_rates.keys()), key=lambda x: missing_rates[x])
    xticks = [f'{DISPLAY_NAMES[feature]}<br>{np.round((missing_rates[feature])*100, 1)}%' for feature in missing_order]
    fig.update_traces(boxpoints=False)
    fig.update_layout(boxmode='group', boxgap=0.15, boxgroupgap=0.1)
    fig.update_xaxes(categoryorder='array', categoryarray=missing_order, ticktext=xticks, tickvals=np.arange(len(xticks)), tickfont=dict(size=16), title_text="Variable with missing rate", title_font=dict(size=20), title_standoff = 15)
    fig.update_layout(width=1600, height=550, showlegend=True, legend=dict(orientation="h", x=0.25, y=-0.2, font=dict(size=18), title_text=''), margin=dict(l=70, r=5, t=5, b=0))
    return fig

def plot_mask_reconstruction(recon_df_per_model: pd.DataFrame, missing_rates: Dict, to_compare: List[Tuple[str, str]])-> go.Figure:
    fig = go.Figure()
    for variant, imputation_mode in to_compare:
        filtered_df = recon_df_per_model[(recon_df_per_model['variant'] == variant) & (recon_df_per_model['imputation_mode'] == imputation_mode)]
        name_str = get_plot_name_str(variant, imputation_mode)
        fig.add_trace(go.Box(x=filtered_df['feature'], y=filtered_df['missing_accuracy'], name=name_str, legendgroup=name_str, marker_color=COLOR_MAP[variant][imputation_mode], offsetgroup=name_str, showlegend=True))
    fig.update_yaxes(title_text="Balanced accuracy", tickfont=dict(size=18), title_font=dict(size=20), title_standoff = 5)
    fig = format_mask_plot(fig, missing_rates)
    return fig

def plot_reconstruction_results(recon_df_per_model: pd.DataFrame, missing_rates: Dict, to_compare: List[Tuple[str, str]])-> go.Figure:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], subplot_titles=("(a)", "(b)"), horizontal_spacing=0.05, vertical_spacing=0.01)
    for variant, imputation_mode in to_compare:
        name_str = get_plot_name_str(variant, imputation_mode)
        filtered_df = recon_df_per_model[(recon_df_per_model['variant'] == variant) & (recon_df_per_model['imputation_mode'] == imputation_mode)]
        discrete_results = filtered_df[filtered_df['feature'].isin(DISCRETE_FEATURES)]
        class_results = filtered_df[filtered_df['feature'].isin(CLASS_FEATURE_LIST)]
        fig.add_trace(go.Box(x=discrete_results['feature'], y=discrete_results['mae'], name=name_str, legendgroup=name_str, marker_color=COLOR_MAP[variant][imputation_mode], offsetgroup=name_str, showlegend=True), row=1, col=1)
        fig.add_trace(go.Box(x=class_results['feature'], y=class_results['corr_coeff'], name=name_str, legendgroup=name_str, marker_color=COLOR_MAP[variant][imputation_mode], offsetgroup=name_str, showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="Mean absolute error", row=1, col=1)

    fig.update_yaxes(title_text="Kendall's Tau", row=1, col=2)

    fig = format_split_plot(fig, missing_rates)

    return fig

## Synthetic Data Evaluation

def evaluate_synthetic_data_per_model(model_results: Dict) -> pd.DataFrame:
    df_data = collections.defaultdict(list)
    for model_name, model_data in tqdm(model_results.items()):
        synth_data = model_data['synth_data']
        test_data = model_data['test_data']
        for feature in DISCRETE_FEATURES + CLASS_FEATURE_LIST:
            df_data['model_name'].append(model_name)
            df_data['variant'].append(model_data['variant'])
            df_data['imputation_mode'].append(model_data['imputation_mode'])
            df_data['variable'].append(feature)
            orig_feature = test_data[feature].dropna()
            synth_feature = synth_data[feature].dropna()
            if feature in DISCRETE_FEATURES:
                ws_dist = wasserstein_distance(orig_feature, synth_feature)
                df_data['metric'].append('wasserstein_distance')
                df_data['value'].append(ws_dist)
            else:
                orig_fracs = orig_feature.value_counts(normalize=True).to_frame(name='orig')
                synth_fracs = synth_feature.value_counts(normalize=True).to_frame(name='synth')
                fracs_merged = pd.merge(orig_fracs, synth_fracs, left_index=True, right_index=True, how='outer')
                fracs_merged.fillna(0, inplace=True)
                fracs_merged['abs_diff'] = np.abs(fracs_merged['orig'] - fracs_merged['synth'])
                mean_frac_diff = fracs_merged['abs_diff'].mean()
                df_data['metric'].append('mean_fraction_difference')
                df_data['value'].append(mean_frac_diff)

            orig_missing_rate = test_data[feature].isna().mean().round(4)
            synth_missing_rate = synth_data[feature].isna().mean().round(4)
            df_data['original_missing_rate'].append(orig_missing_rate)
            df_data['synthetic_missing_rate'].append(synth_missing_rate)
            df_data['missing_rate_difference'].append(np.round(np.abs(orig_missing_rate - synth_missing_rate), 4))

    return pd.DataFrame(df_data)

def calculate_synthetic_correlations(model_results: Dict, variant: str, imputation_mode: str) -> pd.DataFrame:
    to_cat = []
    for model_name, model_data in model_results.items():
        if model_data['variant'] != variant or model_data['imputation_mode'] != imputation_mode:
            continue
        synth_data = model_data['synth_data']
        to_cat.append(synth_data)
    all_synth_data = pd.concat(to_cat, axis=0)
    all_synth_data = all_synth_data[DISCRETE_FEATURES + CLASS_FEATURE_LIST]
    return all_synth_data.corr(method='pearson')

def plot_correlation_heatmap(orig_correlations: pd.DataFrame, synth_correlations: Dict) -> go.Figure:
    nof_rows = int(np.ceil((len(synth_correlations)) / 2))
    fig = make_subplots(rows=nof_rows, cols=2, subplot_titles=[get_plot_name_str(variant, imputation_mode) for variant, imputation_mode in synth_correlations.keys()],
                         horizontal_spacing=0.07, vertical_spacing=0.07)
    orig_correlations = orig_correlations.rename(columns=DISPLAY_NAMES, index=DISPLAY_NAMES)
    for idx, synth_correlation in enumerate(synth_correlations.values()):
        synth_correlation = synth_correlation.rename(columns=DISPLAY_NAMES, index=DISPLAY_NAMES)
        diff = synth_correlation - orig_correlations
        fig.add_trace(go.Heatmap(z=diff.values, x=diff.columns, y=diff.columns, coloraxis='coloraxis', text=diff.values, texttemplate="%{text:.2f}"), row=int(idx / 2) + 1, col=(idx % 2) + 1)
    
    fig.update_layout(width=1600, height=1300, margin=dict(l=15, r=0, t=25, b=15),
                       coloraxis=dict(colorscale='RdBu_r', colorbar=dict(title="Pearson's R difference", titleside="right", titlefont=dict(size=16), tickfont=dict(size=14), len=0.5, xpad=12), cmid=0.0, cmin=-1.0, cmax=1.0))
    return fig

def plot_synthetic_data_results(synth_df_per_model: pd.DataFrame, missing_rates: Dict, to_compare: List[Tuple[str, str]])-> go.Figure:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], subplot_titles=("(a)", "(b)"),
                        horizontal_spacing=0.05, vertical_spacing=0.01)
    for variant, imputation_mode in to_compare:
        name_str = get_plot_name_str(variant, imputation_mode)
        filtered_df = synth_df_per_model[(synth_df_per_model['variant'] == variant) & (synth_df_per_model['imputation_mode'] == imputation_mode)]
        discrete_results = filtered_df[filtered_df['variable'].isin(DISCRETE_FEATURES)]
        class_results = filtered_df[filtered_df['variable'].isin(CLASS_FEATURE_LIST)]
        fig.add_trace(go.Box(x=discrete_results['variable'], y=discrete_results['value'], name=name_str, legendgroup=name_str,
                              marker_color=COLOR_MAP[variant][imputation_mode], offsetgroup=name_str, showlegend=True), row=1, col=1)
        fig.add_trace(go.Box(x=class_results['variable'], y=class_results['value'], name=name_str, legendgroup=name_str,
                              marker_color=COLOR_MAP[variant][imputation_mode], offsetgroup=name_str, showlegend=False), row=1, col=2)

    fig.update_yaxes(title_text="Wasserstein distance", row=1, col=1)

    fig.update_yaxes(title_text="Mean absolute class ratio difference", row=1, col=2)

    fig = format_split_plot(fig, missing_rates)
    return fig

def plot_synthetic_mask(synth_df_per_model: pd.DataFrame, missing_rates: Dict, to_compare: List[Tuple[str, str]])-> go.Figure:
    fig = go.Figure()
    for variant, imputation_mode in to_compare:
        filtered_df = synth_df_per_model[(synth_df_per_model['variant'] == variant) & (synth_df_per_model['imputation_mode'] == imputation_mode)]
        name_str = get_plot_name_str(variant, imputation_mode)
        fig.add_trace(go.Box(x=filtered_df['variable'], y=filtered_df['missing_rate_difference'], name=name_str, legendgroup=name_str, marker_color=COLOR_MAP[variant][imputation_mode], offsetgroup=name_str, showlegend=True))
    fig.update_yaxes(title_text="Absolute difference", tickfont=dict(size=18), title_font=dict(size=20), title_standoff=5)
    fig = format_mask_plot(fig, missing_rates)
    return fig

def config_pair(arg: str) -> Tuple[str, str]:
    pair = arg.split(',')
    if len(pair) != 2:
        raise argparse.ArgumentTypeError(f"Argument {arg} is not a valid pair of values")
    normalization = pair[0]
    imputation_mode = pair[1]
    if normalization not in ['manners', 'vanilla']:
        raise argparse.ArgumentTypeError(f"Normalization {normalization} is not valid, must be 'manners' or 'vanilla'")
    if imputation_mode not in ['mice', 'fixed']:
        raise argparse.ArgumentTypeError(f"Imputation mode {imputation_mode} is not valid, must be 'mice' or 'fixed'")
    return normalization, imputation_mode

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--cv_name', help='Name of cross-validation run, used for all models as stem', required=True)
    requiredNamed.add_argument('--to_compare', type=config_pair, nargs='+',
                               help='Normalization and imputation mode to plot against, e.g., "manners,mice" or "vanilla,fixed". Multiple pairs can be provided, e.g., "manners,mice vanilla,fixed".',
                               required=True)
    parser.add_argument('--overwrite', action='store_true', help='Flag to overwrite existing results with the same cv_name and split')

    args = parser.parse_args()

    cv_name = args.cv_name
    overwrite = args.overwrite
    to_compare = args.to_compare

    result_dir = ANALYSIS_PATH / pathlib.Path(cv_name)
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)
    print('Analysis results will be saved to ' + str(result_dir))

    correlations_dir = result_dir / pathlib.Path('correlations')
    if not correlations_dir.exists():
        correlations_dir.mkdir(parents=True, exist_ok=True)

    tmr_csv_path = result_dir / pathlib.Path('treatment_missing_rate.csv')
    downstream_csv_path = result_dir / pathlib.Path('downstream_results_per_model.csv')
    recon_csv_path = result_dir / pathlib.Path('reconstructions_results_per_model.csv')
    synth_csv_path = result_dir / pathlib.Path('synthetic_data_results_per_model.csv')
    umap_csv_path = result_dir / pathlib.Path('umap_embeddings.csv')

    if not overwrite and downstream_csv_path.exists() and recon_csv_path.exists() and synth_csv_path.exists() and umap_csv_path.exists():
        print('Analysis CSV files already exist, no recalculation needed. Only perform significance tests and generate new plots.')
        calculate_data = False
    else:
        calculate_data = True

    for variant, imputation_mode in to_compare:
        synth_correlation_path = correlations_dir / pathlib.Path(f'synthetic_correlations_{variant}_{imputation_mode}.csv')
        if not synth_correlation_path.exists() and not calculate_data:
            calculate_data = True
            break

    print('Loading original dataset')
    hypo_df = pd.read_csv(NORMALIZED_DATASET_PATH)

    orig_missing_rates = {}
    for feature in DISCRETE_FEATURES + CLASS_FEATURE_LIST:
        orig_missing_rates[feature] = np.round(np.isnan(hypo_df[feature]).mean(), 4)

    # write empty figure, to resolve pdf writing bug
    fig = go.Figure()
    temp_fig_path = result_dir / pathlib.Path('temp.pdf')
    fig.write_image(temp_fig_path)
    temp_fig_path.unlink()
    time.sleep(1.0)

    orig_correlations_csv_path = correlations_dir / pathlib.Path('original_correlations.csv')
    if not orig_correlations_csv_path.exists() or overwrite:
        print('Calculating original correlations')
        orig_corr_df = hypo_df[DISCRETE_FEATURES + CLASS_FEATURE_LIST].corr(method='pearson')
        orig_corr_df.to_csv(orig_correlations_csv_path, index=True)
    else:
        orig_corr_df = pd.read_csv(orig_correlations_csv_path, index_col=0)

    print('Calculating metadata for downstream analysis')
    if tmr_csv_path.exists() and not overwrite:
        stay_mr_df = pd.read_csv(tmr_csv_path)
    else:
        stay_mr_df = calculate_treatment_missing_rates_per_stay(hypo_df)
    stay_mr_df, total_survival_rate, total_mean_los, bin_survival_rates, bin_mean_los, intervals = get_downstream_metadata(stay_mr_df)
    if not tmr_csv_path.exists() or overwrite:
        stay_mr_df.to_csv(tmr_csv_path, index=False)
        print('Saved treatment missing rate data to ' + str(tmr_csv_path))

    if calculate_data:
        print('Loading model data')

        ae_models, variational_models = get_evaluated_models(hypo_df, cv_name)

        print('Calculating downstream performance')
        downstream_df_per_model = generate_per_model_downstream_performance_data(ae_models, stay_mr_df, len(intervals))
        downstream_df_per_model.to_csv(downstream_csv_path, index=False)

        print('Calculating UMAP embeddings')

        umap_df = calculate_umap_embeddings(downstream_df_per_model, stay_mr_df)
        umap_df.to_csv(umap_csv_path, index=False)

        print('Analyzing test reconstruction')

        recon_df_per_model = evaluate_reconstruction_per_model(ae_models)
        recon_df_per_model.to_csv(recon_csv_path, index=False)

        print('Analyzing synthetic data')

        synth_df_per_model = evaluate_synthetic_data_per_model(variational_models)
        synth_df_per_model.to_csv(synth_csv_path, index=False)

        print('Calculating synthetic data correlations')
        synth_correlations = {}
        for variant, imputation_mode in tqdm(to_compare):
            synth_correlation_path = correlations_dir / pathlib.Path(f'synthetic_correlations_{variant}_{imputation_mode}.csv')
            if not synth_correlation_path.exists() or overwrite:
                synth_corr_df = calculate_synthetic_correlations(variational_models, variant, imputation_mode)
                synth_corr_df.to_csv(synth_correlation_path, index=True)
                synth_correlations[(variant, imputation_mode)] = synth_corr_df
    else:
        downstream_df_per_model = pd.read_csv(downstream_csv_path)
        recon_df_per_model = pd.read_csv(recon_csv_path)
        synth_df_per_model = pd.read_csv(synth_csv_path)
        umap_df = pd.read_csv(umap_csv_path)
        synth_correlations = {}
        for variant, imputation_mode in to_compare:
            synth_correlation_path = correlations_dir / pathlib.Path(f'synthetic_correlations_{variant}_{imputation_mode}.csv')
            synth_correlations[(variant, imputation_mode)] = pd.read_csv(synth_correlation_path, index_col=0)

    compare_dir = result_dir / pathlib.Path(f"compare_{'_'.join([f'{x[0]}-{x[1]}' for x in to_compare])}")
    if not compare_dir.exists():
        compare_dir.mkdir(parents=True, exist_ok=True)
    print('Saving comparison results to ' + str(compare_dir))

    print('Calculating summary table for downstream tasks')
    downstream_df = get_downstream_tasks_summary_table(downstream_df_per_model, to_compare, intervals)
    if not downstream_df.empty:
        downstream_df.to_csv(compare_dir / pathlib.Path('downstream_results.csv'), index=False)

    umap_fig = plot_umap_results(umap_df)
    umap_fig.write_image(compare_dir / pathlib.Path('umap_results.pdf'))

    recon_mask_fig = plot_mask_reconstruction(recon_df_per_model, orig_missing_rates, to_compare)
    recon_mask_fig.write_image(compare_dir / pathlib.Path('reconstruction_mask_results.pdf'))
    recon_fig = plot_reconstruction_results(recon_df_per_model, orig_missing_rates, to_compare)
    recon_fig.write_image(compare_dir / pathlib.Path('reconstruction_results.pdf'))
    
    synth_fig = plot_synthetic_data_results(synth_df_per_model, orig_missing_rates, to_compare)
    synth_fig.write_image(compare_dir / pathlib.Path('synthetic_data_results.pdf'))
    synth_mask_fig = plot_synthetic_mask(synth_df_per_model, orig_missing_rates, to_compare)
    synth_mask_fig.write_image(compare_dir / pathlib.Path('synthetic_data_mask_results.pdf'))
    corr_fig = plot_correlation_heatmap(orig_corr_df, synth_correlations)
    corr_fig.write_image(compare_dir / pathlib.Path('correlation_heatmaps.pdf'))

    
