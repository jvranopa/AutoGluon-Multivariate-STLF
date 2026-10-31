import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

# ==========================================================================
# LOAD RECONSTRUCTION AND DIAGNOSTICS (Appendix I)
# ==========================================================================
def discover_harmonic_lags(series: pd.Series, max_lag: int = 4320, top_n: int = 15) -> list:
    """
    Scans the load signal using FFT-accelerated autocorrelation to mathematically 
    discover the natural diurnal and weekly inertia of the specific grid.
    """
    clean_series = series.interpolate(method='pchip').bfill().ffill()
    acf_values = acf(clean_series, nlags=max_lag, fft=True)
    peaks, properties = find_peaks(acf_values, height=0.2)
    top_lags_data = sorted(zip(peaks, properties['peak_heights']), key=lambda x: x[1], reverse=True)[:top_n]
    return [int(lag) for lag, corr in top_lags_data]

def execute_load_reconstruction(df_target: pd.DataFrame, df_temporal: pd.DataFrame, target_col: str) -> dict:
    """
    Executes XGBoost imputation and Rolling MAD anomaly detection, returning
    the reconstructed dataframe alongside its mathematical performance diagnostics.
    """
    df = pd.merge(df_temporal, df_target[['Timestamp', target_col]], on='Timestamp', how='left')
    df['is_missing'] = df[target_col].isna()
    
    seasonal_lags = discover_harmonic_lags(df[target_col])
    immediate_lags = [1, 2, 3, 4, 5, 6, 12, 24, 48, 60, 288, 2016]
    final_lags = sorted(list(set(immediate_lags + seasonal_lags)))
    
    lag_cols = []
    for lag in final_lags:
        col_name = f'lag_{lag}'
        df[col_name] = df[target_col].shift(lag)
        lag_cols.append(col_name)
        
    df[lag_cols] = df[lag_cols].bfill().astype('float32')
    
    feature_cols = [c for c in df.columns if c not in ['Timestamp', target_col, 'is_missing']]
    train_mask = ~df['is_missing']
    
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    
    model = xgb.XGBRegressor(
        objective='reg:absoluteerror', 
        n_estimators=300, 
        max_depth=5, 
        learning_rate=0.05, 
        n_jobs=-1, 
        random_state=42,
        enable_categorical=True
    )
    
    model.fit(X_train, y_train)
    df['model_forecast'] = model.predict(df[feature_cols])
    
    df['residuals'] = np.nan
    df.loc[~df['is_missing'], 'residuals'] = np.abs(
        df.loc[~df['is_missing'], target_col] - df.loc[~df['is_missing'], 'model_forecast']
    )
    
    window_size = 288 
    rolling_med_res = df['residuals'].rolling(window=window_size, center=True, min_periods=144).median()
    rolling_mad = (df['residuals'] - rolling_med_res).abs().rolling(window=window_size, center=True, min_periods=144).median()
    
    threshold = 6 * (1.4826 * rolling_mad)
    
    df['is_anomaly'] = False
    df['is_anomaly'] = (df['residuals'] > threshold) & (~df['is_missing'])
    
    conditions = [df['is_missing'], df['is_anomaly']]
    choices = [df['model_forecast'], df['model_forecast']]
    df['cleaned_load'] = np.select(conditions, choices, default=df[target_col])
    
    export_numerics = [target_col, 'cleaned_load', 'model_forecast']
    df[export_numerics] = df[export_numerics].astype('float32')
    
    total_rows = len(df)
    missing_orig = df['is_missing'].sum()
    anomalies_detected = df['is_anomaly'].sum()
    
    valid_mask = ~df['is_missing'] & ~df['is_anomaly']
    y_true = df.loc[valid_mask, target_col]
    y_pred = df.loc[valid_mask, 'model_forecast']

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wmape = (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100
    
    mask_mape = y_true > 1.0
    mape = np.mean(np.abs((y_true[mask_mape] - y_pred[mask_mape]) / y_true[mask_mape])) * 100

    diagnostics = {
        'Total_Intervals': total_rows,
        'Missing_Imputed': int(missing_orig),
        'Anomalies_Corrected': int(anomalies_detected),
        'MAE_MW': float(mae),
        'RMSE_MW': float(rmse),
        'WMAPE_Percent': float(wmape),
        'MAPE_Percent': float(mape)
    }
    
    return {'reconstructed_dataframe': df, 'diagnostics': diagnostics}

# ==========================================================================
# PHYSICS-AWARE TEMPORAL DISAGGREGATION (Appendix J)
# ==========================================================================
def execute_physics_aware_disaggregation(df_hourly: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    continuous_cols = ['temp', 'humidity', 'solar_ghi', 'wind_speed', 'clouds_total', 'pressure']
    discrete_cols = ['precip_total', 'weather_code']
    
    df_hourly['Timestamp'] = pd.to_datetime(df_hourly['Timestamp'])
    df_hourly.set_index('Timestamp', inplace=True)
    
    full_5min_index = pd.date_range(start=start_date, end=end_date, freq='5min')
    df_phys = df_hourly.reindex(full_5min_index)
    
    # A. Mass Conservation (Accumulated Flux Variables)
    df_phys['precip_total'] = (df_phys['precip_total'].ffill() / 12.0).astype('float32')
    
    # B. Zero-Order Hold (Discrete Categorical States)
    df_phys['weather_code'] = df_phys['weather_code'].ffill().astype(int)
    
    # C. Continuous Thermodynamics (PCHIP Interpolation)
    df_phys[continuous_cols] = df_phys[continuous_cols].interpolate(method='pchip', limit_area='inside')
    
    # D. Trailing Boundary Protection
    df_phys[continuous_cols] = df_phys[continuous_cols].ffill()
    
    for col in continuous_cols:
        df_phys[col] = df_phys[col].astype('float32')
        
    df_phys['weather_code'] = df_phys['weather_code'].astype('category')
    df_final = df_phys.reset_index().rename(columns={'index': 'Timestamp'})
    
    return df_final

# ==========================================================================
# SPEARMAN LAG OPTIMIZATION (Appendix K)
# ==========================================================================
def precalculate_spearman_corrs(df: pd.DataFrame, target_col: str, feat_col: str, max_steps: int) -> list:
    y = df[target_col].values
    x = df[feat_col].values
    corrs = []
    
    for lag in range(0, max_steps + 1):
        x_shifted = pd.Series(x).shift(lag) if lag > 0 else pd.Series(x)
        mask = ~x_shifted.isna()
        
        if mask.sum() > 100:
            rho, _ = spearmanr(y[mask], x_shifted[mask])
            corrs.append(abs(rho))
        else:
            corrs.append(0)
    return corrs

def get_optimal_lag_for_window(corrs: list, window_lags: int) -> tuple:
    sub_corrs = corrs[:window_lags + 1]
    if not sub_corrs or max(sub_corrs) < 0.01:
        return 0, 0.0
    best_lag = int(np.argmax(sub_corrs))
    best_corr = sub_corrs[best_lag]
    return best_lag, best_corr

def run_visual_audit_and_get_lags(df_train: pd.DataFrame) -> dict:
    weather_vars = ['temp', 'humidity', 'solar_ghi', 'wind_speed', 'pressure', 'precip_total', 'clouds_total']
    search_lag_limit = 72
    checkpoints = [12, 24, 36, 48, 60, 72]
    final_lags_dict = {}
    
    for var in weather_vars:
        if var not in df_train.columns:
            continue
            
        corrs = precalculate_spearman_corrs(df_train, 'target', var, max_steps=search_lag_limit)
        row_lags = []
        
        for cp in checkpoints:
            best_lag, _ = get_optimal_lag_for_window(corrs, cp)
            row_lags.append(best_lag)
            
        final_lag = row_lags[-1]
        prev_lag = row_lags[-2]
        
        is_exact_staircase = (row_lags == checkpoints)
        hit_ceiling_no_plateau = (final_lag == search_lag_limit) and (final_lag != prev_lag)
        
        if final_lag == 0 or is_exact_staircase or hit_ceiling_no_plateau:
            final_lags_dict[var] = []
        else:
            final_lags_dict[var] = [final_lag]
            
    return final_lags_dict

def generate_dam_matrices(df_source: pd.DataFrame, target_week_start: str, num_days: int = 7) -> None:
    start_date = pd.to_datetime(target_week_start)
    train_end_w1 = start_date - timedelta(minutes=5)
    
    df_anchor_train = df_source[df_source['timestamp'] <= train_end_w1].copy()
    universal_physical_lags = run_visual_audit_and_get_lags(df_anchor_train)
    
    max_lag = 0
    for lags in universal_physical_lags.values():
        if lags:
            max_lag = max(max_lag, max(lags))
            
    df_train_master = df_source.copy()
    for var, lags in universal_physical_lags.items():
        for lag in lags:
            df_train_master[f"{var}_lag_{lag}"] = df_train_master[var].shift(lag)
            
    df_train_master.bfill(inplace=True)

    for i in range(num_days):
        current_test_day = start_date + timedelta(days=i)
        train_end = current_test_day - timedelta(minutes=5)
        test_start = current_test_day
        test_end = current_test_day + timedelta(hours=23, minutes=55)
        
        train_mask = df_train_master['timestamp'] <= train_end
        df_train = df_train_master.loc[train_mask].copy()
        
        history_prepend = df_source[df_source['timestamp'] <= train_end].tail(max_lag)
        future_window = df_source[(df_source['timestamp'] >= test_start) & (df_source['timestamp'] <= test_end)]
        
        hybrid = pd.concat([history_prepend, future_window]).reset_index(drop=True)
        for var, lags in universal_physical_lags.items():
            for lag in lags:
                hybrid[f"{var}_lag_{lag}"] = hybrid[var].shift(lag)
                
        df_test = hybrid[hybrid['timestamp'] >= test_start].copy()
