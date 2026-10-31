import os
import gc
import torch
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# ==========================================================================
# AUTOGLUON TIME SERIES PREDICTOR (Appendix N)
# ==========================================================================
def apply_pytorch_backend_tuning():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)

apply_pytorch_backend_tuning()

def apply_scenario_filter(df: pd.DataFrame, scenario: str, target_col: str = 'target'):
    all_temporal_vars = [
        'is_pre_holiday', 'is_post_holiday', 'is_bridge_day', 'is_day', 
        'is_weekend', 'is_holiday', 'day_of_week',
        'hour_sin', 'hour_cos', 'day_year_sin', 'day_year_cos'
    ]
    
    active_temporal = [c for c in df.columns if c in all_temporal_vars]
    weather_vars = [c for c in df.columns if c not in active_temporal and c not in ['timestamp', 'item_id', target_col]]
    
    keep_cols = ['timestamp', 'item_id']
    if target_col in df.columns: 
        keep_cols.append(target_col)
        
    if scenario == "TEMPORAL_ONLY" or scenario == "FULL":
        keep_cols.extend(active_temporal)
    if scenario == "WEATHER_ONLY" or scenario == "FULL":
        keep_cols.extend(weather_vars)

    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols: 
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

def enforce_tensor_dtypes(df_dict: dict, target_col: str = 'target'):
    categorical_vars = ['day_of_week', 'weather_code']
    df_train = df_dict['train']
    
    for name, df in df_dict.items():
        if target_col in df.columns:
            df[target_col] = df[target_col].astype('float32')
            
        for col in df.columns:
            if col not in categorical_vars and col not in ['timestamp', 'item_id', target_col]:
                df[col] = df[col].astype('float32')

    for col in categorical_vars:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype('category')
            master_categories = df_train[col].cat.categories
            
            for name, df in df_dict.items():
                if col in df.columns and name != 'train':
                    df[col] = pd.Categorical(df[col], categories=master_categories, ordered=False)

def execute_autogluon_batch(df_train: pd.DataFrame, df_truth: pd.DataFrame, df_inference: pd.DataFrame, scenario: str):
    target_col = 'target'
    prediction_length = 288
    time_limit = 3600
    
    train_cols = df_train.columns
    infer_cols = [c for c in train_cols if c != target_col]
    df_truth = df_truth.reindex(columns=train_cols)
    df_inference = df_inference.reindex(columns=infer_cols)
    
    df_combined = pd.concat([df_train, df_truth], ignore_index=True)

    dfs = {'train': df_train, 'truth': df_truth, 'inference': df_inference, 'combined': df_combined}
    for name, df in dfs.items():
        apply_scenario_filter(df, scenario)
        
    enforce_tensor_dtypes(dfs)
    
    if target_col in dfs['inference'].columns:
        raise ValueError("CRITICAL ERROR: Target column leaked into Inference Dataset!")

    train_ts = TimeSeriesDataFrame.from_data_frame(dfs['train'], id_column="item_id", timestamp_column="timestamp")
    tuning_ts = TimeSeriesDataFrame.from_data_frame(dfs['combined'], id_column="item_id", timestamp_column="timestamp")
    inference_ts = TimeSeriesDataFrame.from_data_frame(dfs['inference'], id_column="item_id", timestamp_column="timestamp")
    
    known_covariates = list(inference_ts.columns) if scenario != "TARGET_ONLY" else None

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target=target_col,
        eval_metric="MAPE",
        known_covariates_names=known_covariates,
        quantile_levels=[0.1, 0.5, 0.9]
    )
    
    predictor.fit(
        train_data=train_ts,
        tuning_data=tuning_ts, 
        presets="best_quality",
        time_limit=time_limit,
        random_seed=42,
        excluded_model_types=["NPTS", "RecursiveTabular"]
    )
    
    pred_covariates = inference_ts if scenario != "TARGET_ONLY" else None
    predictions = predictor.predict(train_ts, known_covariates=pred_covariates)
    
    y_true = dfs['truth'].set_index(['item_id', 'timestamp'])[target_col]
    y_pred_mean = predictions['mean']
    y_true = y_true.reindex(y_pred_mean.index) 

    mape = np.mean(np.abs((y_true - y_pred_mean) / y_true)) * 100
    wape = np.sum(np.abs(y_true - y_pred_mean)) / np.sum(np.abs(y_true)) 
    mae = np.mean(np.abs(y_true - y_pred_mean))
    rmse = np.sqrt(np.mean((y_true - y_pred_mean) ** 2))
    
    del predictor, train_ts, tuning_ts, inference_ts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        'predictions': predictions,
        'metrics': {'MAPE': mape, 'WAPE': wape, 'MAE': mae, 'RMSE': rmse}
    }

# ==========================================================================
# ECONOMETRIC VALIDATION (Appendix O)
# ==========================================================================
def calculate_gw_test_with_nw_hac(loss_a: pd.Series, loss_b: pd.Series, lag_window: int = 24) -> tuple:
    d_t = loss_a - loss_b
    
    X = np.ones(len(d_t))
    model = sm.OLS(d_t, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag_window})
    
    mean_diff = d_t.mean()
    t_stat = results.tvalues.iloc[0]
    p_val = results.pvalues.iloc[0]
    
    return mean_diff, t_stat, p_val

def execute_regime_significance_testing(df_regime: pd.DataFrame, regime_name: str) -> pd.DataFrame:
    comparisons = [
        ("LO vs MT", df_regime['loss_TARGET_ONLY'], df_regime['loss_FULL']),
        ("TE vs MT", df_regime['loss_TEMPORAL_ONLY'], df_regime['loss_FULL']),
        ("LO vs TE", df_regime['loss_TARGET_ONLY'], df_regime['loss_TEMPORAL_ONLY'])
    ]
    
    results_list = []
    for comp_name, loss_a, loss_b in comparisons:
        mean_diff, t_stat, p_val = calculate_gw_test_with_nw_hac(loss_a, loss_b, lag_window=24)
        is_significant = "Yes" if p_val < 0.05 else "No"
        
        results_list.append({
            "Regime": regime_name,
            "Configuration_Comparison": comp_name,
            "Mean_Loss_Difference": mean_diff,
            "T_Statistic": t_stat,
            "P_Value": p_val,
            "Significant_at_5%": is_significant
        })
        
    return pd.DataFrame(results_list)
