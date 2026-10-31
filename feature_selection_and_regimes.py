import pandas as pd
import numpy as np
from BorutaShap import BorutaShap
from lightgbm import LGBMRegressor
import ppscore as pps
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# ==========================================================================
# FEATURE CONSOLIDATION (Appendix L)
# ==========================================================================
def execute_boruta_shap_ablation(df_train: pd.DataFrame, target_col: str = 'target') -> dict:
    drop_cols = [target_col, 'timestamp', 'item_id']
    drop_cols = [c for c in drop_cols if c in df_train.columns]

    X_train = df_train.drop(columns=drop_cols)
    y_train = df_train[target_col]

    if X_train.isna().sum().sum() > 0:
        raise ValueError("Fatal Integrity Error: NaNs detected in training matrix.")

    proxy_model = LGBMRegressor(
        n_estimators=200, 
        max_depth=5, 
        random_state=42, 
        n_jobs=-1, 
        verbose=-1, 
        learning_rate=0.05
    )

    feature_selector = BorutaShap(
        model=proxy_model,
        importance_measure='shap',
        classification=False
    )

    feature_selector.fit(X=X_train, y=y_train, n_trials=50, random_state=42, verbose=False)
    
    return {
        'accepted': list(feature_selector.accepted),
        'rejected': list(feature_selector.rejected),
        'tentative': list(feature_selector.tentative)
    }

def calculate_multivariate_diagnostics(df_features: pd.DataFrame, target_col: str = 'target') -> dict:
    X = df_features.drop(columns=[target_col, 'item_id', 'timestamp'], errors='ignore')
    y = df_features[target_col]

    base_discrete_vars = [
        'is_pre_holiday', 'is_post_holiday', 'is_bridge_day', 'is_day', 
        'day_of_week', 'is_weekend', 'is_holiday', 'weather_code'
    ]
    discrete_vars = [col for col in X.columns if any(base in col for base in base_discrete_vars)]

    # 1. Mutual Information
    np.random.seed(42)
    mi_results = []
    
    y_mi = y.astype(np.float64).values 
    y_noise = np.linspace(-1e-3, 1e-3, len(y_mi))
    np.random.shuffle(y_noise)
    y_mi += y_noise

    for col in X.columns:
        is_discrete = col in discrete_vars
        x_raw = X[col].astype(np.float64).values.reshape(-1, 1)
        
        if not is_discrete:
            x_noise = np.linspace(-1e-3, 1e-3, len(x_raw)).reshape(-1, 1)
            np.random.shuffle(x_noise)
            x_raw += x_noise
            
        score = mutual_info_regression(
            x_raw, y_mi, discrete_features=[is_discrete], n_neighbors=5, random_state=42
        )[0]
        mi_results.append({'Feature': col, 'MI_Score': score})
        
    mi_df = pd.DataFrame(mi_results).sort_values(by='MI_Score', ascending=False)

    # 2. Predictive Power Score
    pps_results = []
    for col in X.columns:
        score = pps.score(df_features, x=col, y=target_col, cross_validation=5, sample=None, random_seed=42)['ppscore']
        pps_results.append({'Feature': col, 'PPS': score})

    pps_df = pd.DataFrame(pps_results).sort_values(by='PPS', ascending=False)

    # 3. Hierarchical Cluster Analysis
    spearman_corr = X.corr(method='spearman').fillna(0)
    distance_matrix = 1 - np.abs(spearman_corr)

    dist_array = distance_matrix.values
    dist_array = np.tril(dist_array, -1) + np.tril(dist_array, -1).T
    np.fill_diagonal(dist_array, 0.0)
    
    dist_linkage = hierarchy.average(squareform(dist_array))
    redundancy_threshold = 0.3
    cluster_labels = hierarchy.fcluster(dist_linkage, t=redundancy_threshold, criterion='distance')
    
    hca_clusters = pd.DataFrame({
        'Feature': X.columns,
        'Cluster_ID': cluster_labels
    }).sort_values(by='Cluster_ID')

    return {'MI': mi_df, 'PPS': pps_df, 'HCA_Clusters': hca_clusters}

# ==========================================================================
# REGIME SELECTION (Appendix M)
# ==========================================================================
def execute_wasserstein_regime_scan(df: pd.DataFrame) -> pd.DataFrame:
    weather_cols = ['temp', 'humidity', 'solar_ghi', 'wind_speed', 'clouds_total', 'pressure', 'precip_total']
    features = ['target'] + weather_cols
    behavioral_flags = ['is_holiday', 'is_pre_holiday', 'is_post_holiday', 'is_bridge_day']

    df_clean_unscaled = df[features + behavioral_flags].dropna()

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_clean_unscaled[features]), 
        columns=features, 
        index=df_clean_unscaled.index
    )

    window_size = 2016 
    step_size = 288 
    global_dist = df_scaled 
    results = []

    for i in range(window_size, len(df_scaled) - window_size + 1, step_size):
        prev_window = df_scaled.iloc[i - window_size : i]
        curr_window = df_scaled.iloc[i : i + window_size]
        curr_window_unscaled = df_clean_unscaled.iloc[i : i + window_size]

        start_date = curr_window.index[0].strftime('%Y-%m-%d')
        end_date = curr_window.index[-1].strftime('%Y-%m-%d')

        res_dict = {'Start_Date': start_date, 'End_Date': end_date}
        
        global_total, seq_total, seq_weather, seq_temporal = 0.0, 0.0, 0.0, 0.0
        global_weather = 0.0
        
        for col in features:
            g_wd = wasserstein_distance(global_dist[col], curr_window[col])
            s_wd = wasserstein_distance(prev_window[col], curr_window[col])
            
            res_dict[f'Global_WD_{col}'] = g_wd
            res_dict[f'Seq_WD_{col}'] = s_wd
            
            global_total += g_wd
            seq_total += s_wd
            
            if col == 'target': 
                seq_temporal += s_wd
            else: 
                seq_weather += s_wd
                global_weather += g_wd

        res_dict['Physical_Mean_Temp'] = curr_window_unscaled['temp'].mean()
        res_dict['Physical_Mean_Solar'] = curr_window_unscaled['solar_ghi'].mean()
        res_dict['Physical_Sum_Precip'] = curr_window_unscaled['precip_total'].sum()
        res_dict['Physical_Mean_Clouds'] = curr_window_unscaled['clouds_total'].mean()

        res_dict['Behavioral_Anomaly_Count'] = curr_window_unscaled[behavioral_flags].sum().sum()
        res_dict['Load_Variance'] = curr_window_unscaled['target'].var()
        
        dataset_start = pd.to_datetime('2023-01-01')
        current_start = pd.to_datetime(start_date)
        res_dict['Training_Days_Available'] = (current_start - dataset_start).days

        res_dict['Global_WD_Total_System'] = global_total
        res_dict['Global_WD_Weather_Only'] = global_weather
        res_dict['Seq_WD_Total_Concept_Drift'] = seq_total
        
        results.append(res_dict)

    return pd.DataFrame(results)

def extract_non_overlapping_regimes(df_res: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    top_n = []
    for _, row in df_res.iterrows():
        start_dt = pd.to_datetime(row['Start_Date'])
        overlap = any(abs((start_dt - pd.to_datetime(s['Start_Date'])).days) < 7 for s in top_n)
        if not overlap:
            top_n.append(row)
        if len(top_n) == n:
            break
    return pd.DataFrame(top_n)
