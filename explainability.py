import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================================================
# POST-HOC EXPLAINABILITY (Appendix P)
# ==========================================================================
def calculate_surrogate_fidelity(y_actual: np.ndarray, y_surrogate: np.ndarray) -> dict:
    r2 = r2_score(y_actual, y_surrogate)
    spearman, _ = spearmanr(y_actual, y_surrogate)
    mape = np.mean(np.abs((y_actual - y_surrogate) / np.where(y_actual == 0, 1e-8, y_actual))) * 100
    
    return {
        'R2_Score': r2, 
        'Spearman_Rank_Correlation': spearman, 
        'Fidelity_MAPE': mape
    }

def execute_surrogate_shap_pipeline(df_features: pd.DataFrame, y_autogluon_preds: pd.Series) -> dict:
    # 1. Train the Interpretable Surrogate Model
    surrogate = lgb.LGBMRegressor(
        n_estimators=200, 
        max_depth=8, 
        learning_rate=0.05, 
        random_state=42, 
        verbose=-1
    )
    
    surrogate.fit(df_features, y_autogluon_preds)
    y_surrogate_preds = surrogate.predict(df_features)
    
    # 2. Mathematical Fidelity Validation
    fidelity_metrics = calculate_surrogate_fidelity(y_autogluon_preds.values, y_surrogate_preds)
    
    if fidelity_metrics['R2_Score'] < 0.90:
        warnings.warn("Surrogate fidelity is critically low. SHAP values may not accurately reflect the primary ensemble.")

    # 3. Post-Hoc Explainability (TreeSHAP Extraction)
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(df_features)
    
    # A. Format Local (High-Frequency) SHAP Matrix
    feature_names = df_features.columns.tolist()
    df_raw_shap = pd.DataFrame(shap_values, columns=[f"SHAP_{c}" for c in feature_names])
    
    df_val = pd.DataFrame(df_features.values, columns=[f"VAL_{c}" for c in feature_names])
    df_local_autopsy = pd.concat([df_val, df_raw_shap], axis=1)
    
    # B. Calculate Global SHAP (Mean Absolute Attribution)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    global_shap_list = []
    for feat, importance in zip(feature_names, mean_abs_shap):
        global_shap_list.append({
            'Feature': feat,
            'Mean_Absolute_SHAP_Impact': importance
        })
        
    df_global_shap = pd.DataFrame(global_shap_list).sort_values(by='Mean_Absolute_SHAP_Impact', ascending=False)
    
    return {
        'Fidelity': fidelity_metrics,
        'Global_Attribution': df_global_shap,
        'Local_Autopsy_Matrix': df_local_autopsy
    }
