import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

def cross_validated_survival_predictions(
    df,                      
    y,                       
    all_var,                 
    preprocessor,            
    model_params,            
    survival_timepoints,     
    tau,                     
    n_splits=5,              
    random_state=42,         
    verbose=True             
):
    """
    Perform cross-validated survival analysis with GradientBoostingSurvivalAnalysis.
    
    Returns risk scores and survival probabilities at specified timepoints for each patient.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    y : np.array
        Structured array with survival outcomes (event, time)
    all_var : list
        List of feature column names to use
    preprocessor : sklearn transformer
        Preprocessing pipeline (e.g., ColumnTransformer)
    n_splits : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed for reproducibility
    model_params : dict
        Parameters for GradientBoostingSurvivalAnalysis. 
    survival_timepoints : tuple
        Timepoints in days for survival probability predictions 
    tau : int
        Maximum time for evaluation grid
    verbose : bool, default=True
        Whether to print summary statistics
        
    Returns:
    --------
    pd.DataFrame : Original df with added columns:
        - 'risk_score': predicted risk scores
        - 'psurv_{timepoint}': survival probabilities at each timepoint
    """
    # Ensure PatientID is the index
    if 'PatientID' in df.columns:
        df = df.set_index('PatientID')
    
    # Initialize storage
    all_patient_ids = df.index
    risk_score_preds = []
    psurv_preds = {tp: [] for tp in survival_timepoints}
    
    # Time grid for survival function evaluation
    t_grid = np.linspace(0.0, tau, tau + 1)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df, df['event'])):
        if verbose:
            print(f"Processing fold {fold_idx + 1}/{n_splits}")
        
        # Split data
        patient_ids_train = all_patient_ids[train_idx]
        patient_ids_test = all_patient_ids[test_idx]
        
        train_df_fold = df.loc[patient_ids_train]
        test_df_fold = df.loc[patient_ids_test]
        train_y_fold = y[train_idx]
        
        # Build and train model
        model = GradientBoostingSurvivalAnalysis(**model_params)
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        full_pipeline.fit(train_df_fold[all_var], train_y_fold)
        
        # Get risk score predictions
        test_predictions = full_pipeline.predict(test_df_fold[all_var])
        fold_predictions = list(zip(patient_ids_test, test_predictions))
        risk_score_preds.extend(fold_predictions)
        
        # Get survival function predictions
        X_test_tx = full_pipeline.named_steps['preprocessor'].transform(test_df_fold[all_var])
        surv_fns = full_pipeline.named_steps['model'].predict_survival_function(X_test_tx)
        
        # Evaluate at specified timepoints
        for pid, fn in zip(patient_ids_test, surv_fns):
            S_grid = fn(t_grid)
            for tp in survival_timepoints:
                psurv_preds[tp].append((pid, float(S_grid[tp])))
    
    # Add predictions to dataframe
    df = df.copy()
    df['risk_score'] = df.index.map(dict(risk_score_preds))
    
    for tp in survival_timepoints:
        col_name = f'psurv_{tp}'
        df[col_name] = df.index.map(dict(psurv_preds[tp]))
    
    # Print summary
    if verbose:
        print(f"\nNumber of patients in df: {df.shape[0]}")
        print(f"risk_scores computed: {df['risk_score'].notna().sum()}")
        for tp in survival_timepoints:
            col_name = f'psurv_{tp}'
            print(f"{col_name} computed: {df[col_name].notna().sum()}")
    
    return df