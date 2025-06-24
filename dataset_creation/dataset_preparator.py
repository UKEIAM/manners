import pandas as pd
import numpy as np
import pathlib
import collections
from tqdm import tqdm
from scipy.stats import boxcox
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import RAW_DATA_DIR, LABEL_PATH, UNNORMALIZED_DATASET_PATH, DISCRETE_FEATURES, CLASS_FEATURES, NORMALIZATION_PARAMS_PATH, NORMALIZED_DATASET_PATH

class DataPreparator:
    """
    A class used to prepare the Hypotension dataset of Gottesman et al. (https://doi.org/10.48550/arXiv.2002.03478) extracted from the MIMIC-IV database. 
    Events are quantized into hours, class data is binned. Discrete data is normalized using Box-Cox transformation and then standardized.
    """

    def __init__(self):
        self.label_df = pd.read_csv(LABEL_PATH)
        for time_col in ['icu_intime', 'icu_outtime', 'hosp_discharge_time', 'hosp_death_time']:
            self.label_df[time_col] = pd.to_datetime(self.label_df[time_col])
        self.intime_df = self.label_df[['stay_id', 'icu_intime']]

    @staticmethod
    def prepared_data_exists() -> bool:
        return pathlib.Path(UNNORMALIZED_DATASET_PATH).exists() and pathlib.Path(NORMALIZED_DATASET_PATH).exists()
    
    def prepare_2_day_survival(self):
        self.label_df['icu_intime_discharge_diff'] = (self.label_df['hosp_discharge_time'] - self.label_df['icu_intime']).dt.total_seconds() / (24* 3600)
        self.label_df['survival48h'] = ((self.label_df['survival'] == 1) | (self.label_df['icu_intime_discharge_diff'] > 2)).astype(int)
        self.label_df.to_csv(LABEL_PATH, index=False)

    def prepare_vasopressors(self):
        print('Reading raw vasopressor data')
        raw_df = pd.read_csv(RAW_DATA_DIR / pathlib.Path('vasopressor_data.csv'))
        raw_df = raw_df.merge(self.intime_df, on='stay_id', how='inner')
        raw_df['starttime'] = pd.to_datetime(raw_df['starttime'])
        raw_df['endtime'] = pd.to_datetime(raw_df['endtime'])
        raw_df['start_hour'] = np.floor((raw_df['starttime'] - raw_df['icu_intime'])/pd.Timedelta(hours=1)).astype('int')
        raw_df['end_hour'] = np.floor((raw_df['endtime'] - raw_df['icu_intime'])/pd.Timedelta(hours=1)).astype('int')
        print('Spliting vasopressor data to hours')
        split_df_data = collections.defaultdict(list)
        for idx, row in tqdm(raw_df.iterrows(), total=raw_df.shape[0]):
            for hour in range(row['start_hour'], min(row['end_hour'] +1, 48)):
                # max between start time and current hour since admission
                # start time could be some value within hour interval
                start_time_in_hour = max(row['starttime'], row['icu_intime'] + pd.DateOffset(hours=hour))
                # same for end time, but min
                end_time_in_hour = min(row['endtime'], row['icu_intime'] + pd.DateOffset(hours=hour+1))
                duration = (end_time_in_hour - start_time_in_hour) / pd.Timedelta(minutes=1)
                # rate was given per minute, move to hour
                hour_value = row['adjusted_rate'] * duration
                split_df_data['stay_id'].append(row['stay_id'])
                split_df_data['hour'].append(hour)
                split_df_data['value'].append(hour_value)
        split_df = pd.DataFrame(split_df_data)
        split_df = split_df[split_df['hour'] >= -1]
        split_df['hour'] = split_df['hour'].clip(lower=0)
        grouped = split_df.groupby(['stay_id', 'hour'])
        result = grouped['value'].sum().reset_index().sort_values(by=['stay_id', 'hour'])
        print('Generate binned classes')
        discrete_vals = result['value']
        class_cases = [discrete_vals < 8.4, discrete_vals < 20.28, discrete_vals >= 20.28]
        result['value'] = np.select(class_cases, [0, 1, 2], default=-1)
        result['value'] = result['value'].astype(int)
        return result
    
    def prepare_charted_data(self) -> pd.DataFrame:
        print('Reading raw charted data')
        raw_df = pd.read_csv(RAW_DATA_DIR / pathlib.Path('charted_data.csv'))
        orig_len = len(raw_df)
        print('Clean up')
        # remove blood pressure of > 250
        raw_df = raw_df[~((raw_df['item'].isin(['MAP', 'DBP', 'SBP'])) & (raw_df['value'] > 250))]
        # move FiO2 to fraction and remove implausible values above 100%
        raw_df.loc[(raw_df['item'] == 'FiO2') & (raw_df['value'] > 1), 'value'] = raw_df['value'] / 100
        raw_df = raw_df[~((raw_df['item'] == 'FiO2') & (raw_df['value'] > 1))]
        gcs_df = raw_df[raw_df['item'].str.startswith('GCS')]
        all_three_gcs_events = gcs_df.groupby(['stay_id', 'charttime'])['item'].transform('nunique') == 3
        summed_gcs_df = gcs_df.loc[all_three_gcs_events].groupby(['stay_id', 'charttime'])['value'].sum().reset_index()
        summed_gcs_df['item'] = 'GCS'
        summed_gcs_df['value'] = summed_gcs_df['value'].astype(int)
        non_gcs_df = raw_df[~raw_df['item'].str.startswith('GCS')]
        clean_thresholds = {
            'MAP' : 200,
            'DBP' : 180,
            'SBP' : 220,
            'pO2' : 500,
            'serum_creatinine' : 8,
            'ALT' : 750,
            'AST' : 1000,
            'lactate' : 15,
            'urine' : 750
        }
        for key, value in clean_thresholds.items():
            non_gcs_df = non_gcs_df[~((non_gcs_df['item'] == key) & (non_gcs_df['value'] > value))]  
        result = pd.concat([non_gcs_df, summed_gcs_df])
        cleaned_len = len(result)
        print('Cleaned ' + str(orig_len - cleaned_len) + ' entries from the charted data')
        return result
    
    def derive_values_per_hour(self, charted_df) ->pd.DataFrame:
        raw_bolus_df = pd.read_csv(RAW_DATA_DIR / pathlib.Path('bolus_data.csv'))
        raw_bolus_df['item'] = 'bolus'
        raw_outputevents_df = pd.read_csv(RAW_DATA_DIR / pathlib.Path('outputevents_data.csv'))
        raw_combined_df = pd.concat([charted_df, raw_bolus_df, raw_outputevents_df])
        raw_combined_df = raw_combined_df.merge(self.intime_df, on='stay_id', how='inner')
        raw_combined_df['charttime'] = pd.to_datetime(raw_combined_df['charttime'])
        raw_combined_df['hour'] = np.floor((raw_combined_df['charttime'] - raw_combined_df['icu_intime'])/pd.Timedelta(hours=1)).astype('int')
        # remove everything more than one hour from the valid time interval away, clip hour
        raw_combined_df = raw_combined_df[(raw_combined_df['hour'] >= -1) & (raw_combined_df['hour'] <= 48)]
        raw_combined_df['hour'] = raw_combined_df['hour'].clip(lower=0, upper=47)
        bp_df = raw_combined_df[raw_combined_df['item'].isin(['MAP', 'DBP', 'SBP'])].pivot_table(index=['stay_id', 'hour'], columns='item', values='value', aggfunc='min')
        summed_df = raw_combined_df[raw_combined_df['item'].isin(['bolus', 'urine'])].pivot_table(index=['stay_id', 'hour'], columns='item', values='value', aggfunc='sum').replace({0.0: np.nan})
        summed_df['urine'] = summed_df['urine'].clip(upper=1000)
        meaned_df = raw_combined_df[raw_combined_df['item'].isin(['GCS', 'serum_creatinine', 'FiO2', 'lactate', 'pO2', 'ALT', 'AST'])].pivot_table(index=['stay_id', 'hour'], columns='item', values='value', aggfunc='mean')
        meaned_df['GCS'] = meaned_df['GCS'].round()
        result = pd.concat([bp_df, summed_df, meaned_df], axis=1).reset_index().rename_axis(None, axis=1)
        return result
    
    def generate_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        # Decrement GCS by 3 to obtain class index
        df['GCS'] = df['GCS'] - 3
        # Bin bolus treatments
        bolus_cases = [df['bolus'] < 500.0, df['bolus'] < 1000.0, df['bolus'] >= 1000.0]
        df['bolus'] = np.select(bolus_cases, [0, 1, 2], default=np.nan)
        # move FiO2 to class indices
        fio2_cases = [df['FiO2'] < 0.2, df['FiO2'] < 0.3, df['FiO2'] < 0.4, df['FiO2'] < 0.5, df['FiO2'] < 0.6, df['FiO2'] < 0.7, df['FiO2'] < 0.8, df['FiO2'] < 0.9, df['FiO2'] < 1.0, df['FiO2'] == 1.0]
        df['FiO2'] = np.select(fio2_cases, range(10), default=np.nan)
        return df
    
    def add_reporting_channel(self, df: pd.DataFrame) -> pd.DataFrame:
        df['data_reported'] = 1
        not_reported_rows = []
        unique_stay_ids = df['stay_id'].unique()
        other_cols = [col for col in df.columns if col not in ['stay_id', 'hour', 'data_reported']]
        for hour in tqdm(range(0, 48)):
            hour_stay_ids = set(df.loc[df['hour'] == hour]['stay_id'])
            for stay_id in unique_stay_ids:
                if stay_id not in hour_stay_ids:
                    row = {'stay_id' : stay_id, 'hour' : hour, 'data_reported' : 0}
                    row.update({col : np.nan for col in other_cols})
                    not_reported_rows.append(row)

        return pd.concat([df, pd.DataFrame.from_records(not_reported_rows)], ignore_index=True)
    
    def normalize_disrete_features(self, df: pd.DataFrame) -> pd.DataFrame:
        normalization_data = [['mean'], ['std'], ['boxcox_lambda']]
        columns = ['index']
        for var in DISCRETE_FEATURES:
            columns.append(var)
            non_missing = df[~np.isnan(df[var])][var]
            trans_data, lmbd = boxcox(non_missing)
            df.loc[~np.isnan(df[var]), var] = trans_data
            mean = trans_data.mean()
            std = trans_data.std()
            df[var] = (df[var] - mean)/((std))
            normalization_data[0].append(mean)
            normalization_data[1].append(std)
            normalization_data[2].append(lmbd)

        normalization_df = pd.DataFrame(normalization_data, columns=columns)
        normalization_df.index = normalization_df['index']
        del normalization_df['index']
        normalization_df.to_csv(NORMALIZATION_PARAMS_PATH)
        return df

    def prepare_dataset(self):
        self.prepare_2_day_survival()
        vasopressor_df = self.prepare_vasopressors()
        charted_df = self.prepare_charted_data()
        print('Quantize data to hours')
        hourly_df = self.derive_values_per_hour(charted_df)
        vasopressor_df = vasopressor_df.rename(columns={'value': 'vasopressor'})
        hourly_df = hourly_df.merge(vasopressor_df, on=['stay_id', 'hour'], how='outer')
        result = self.generate_classes(hourly_df)
        print('Fill missing hours')
        result = self.add_reporting_channel(result)
        result = result.round(2)
        for col in ['GCS', 'bolus', 'vasopressor', 'FiO2']:
            result[col] = result[col].astype('Int64')
        result.sort_values(by=['stay_id', 'hour'], inplace=True, ignore_index=True)
        result = result[['stay_id', 'hour', 'data_reported'] + DISCRETE_FEATURES + list(CLASS_FEATURES.keys())]
        result.to_csv(UNNORMALIZED_DATASET_PATH, index=False)
        print('Normalizing discrete features')
        result_normalized = self.normalize_disrete_features(result)
        result_normalized = result_normalized.round(4)
        result_normalized.to_csv(NORMALIZED_DATASET_PATH, index=False)