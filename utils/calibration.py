# utils/calibration.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sksurv.metrics import brier_score
from lifelines import KaplanMeierFitter

from utils.pseudo_obs import pseudo_observations_psurv

def calibrate_survival_predictions(
    df,
    y,
    prediction_col: str,
    tau: int,
    n_splits=5,
    random_state=42,
    calculate_brier=True,
    verbose=True
):
    """
    Calibrate survival probability predictions using isotonic regression with cross-validation.
    
    Uses pseudo-observations from training folds to fit isotonic regression models,
    then applies calibration to test fold predictions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with PatientID, event, duration, and uncalibrated prediction columns
    y : np.array
        Structured array with survival outcomes (event, time)
    prediction_col : str
        Column name containing uncalibrated predictions (e.g., 'psurv_180')
    tau : int
        Timepoint in days corresponding to `prediction_col`.
    n_splits : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed for reproducibility
    calculate_brier : bool, default=True
        Whether to calculate Brier score
    verbose : bool, default=True
        Whether to print progress and results
        
    Returns:
    --------
    tuple : (df_with_calibrated, brier_scores)
        - df_with_calibrated: DataFrame with new calibrated prediction column
        - brier_scores: List of Brier scores per fold (empty list if calculate_brier=False)
    """    
    # Input validation 
    if 'PatientID' in df.columns:
        df = df.set_index('PatientID')
    if prediction_col not in df.columns:
        raise KeyError(f"'{prediction_col}' not found in df columns.")
    if not {'event', 'duration'}.issubset(df.columns):
        raise KeyError("df must contain 'event' and 'duration' columns.")


    all_patient_ids = df.index
    
    # Initialize storage for calibrated predictions
    calibrated_preds = {}
    brier_scores = []
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df, df['event'])):
        if verbose:
            print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")
        
        # Split data
        patient_ids_train = all_patient_ids[train_idx]
        patient_ids_test = all_patient_ids[test_idx]
        
        train_df_fold = df.loc[patient_ids_train]
        test_df_fold = df.loc[patient_ids_test]
        
        train_y_fold = y[train_idx]
        test_y_fold = y[test_idx]
        
        # Uncalibrated predictions
        pred_train = train_df_fold[prediction_col].values
        pred_test = test_df_fold[prediction_col].values
            
        # Calculate pseudo-observations on training fold
        pseudo_obs_train = pseudo_observations_psurv(
            train_df_fold['duration'].values,
            train_df_fold['event'].values,
            tau=float(tau)
        )
            
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        iso_reg.fit(pred_train, pseudo_obs_train)

        # Calibrated predictions for test fold
        pred_cal = iso_reg.predict(pred_test)
            
        # Store calibrated predictions with patient IDs
        for pid, p_cal in zip(patient_ids_test, pred_cal):
            calibrated_preds[pid] = float(p_cal)
        
        # Calculate Brier score if requested
        if calculate_brier:

            # Filter test set to only include times within training range
            max_train_time = train_df_fold['duration'].max()
            brier_mask = (test_df_fold['duration'] <= max_train_time).values
            
            if brier_mask.sum() > 0:  
                times_eval = np.array([float(tau)])
                surv_probs_fold = pred_cal[brier_mask].reshape(-1, 1)
                
                # Calculate Brier score
                _, brier_at_tau = brier_score(train_y_fold, test_y_fold[brier_mask], 
                                              surv_probs_fold, times_eval)
                brier_scores.append(float(brier_at_tau[0]))
                if verbose:
                    print(f"Brier score at {tau} days (calibrated): {brier_at_tau[0]:.4f}")
    
    # Add calibrated predictions to dataframe
    df = df.copy()
    calibrated_col = f"{prediction_col}_calibrated"
    df[calibrated_col] = df.index.map(calibrated_preds)
    
    # Reset index to make PatientID a column again
    df = df.reset_index()

    # Print summary
    if verbose:
        print(f"\n=== Summary ===")
        print(f"{calibrated_col} computed: {df[calibrated_col].notna().sum()}")
        if brier_scores:
            print(f"\n=== Brier Scores at {tau} days (calibrated) ===")
            print(f"Mean: {np.mean(brier_scores):.4f} ± {np.std(brier_scores):.4f}")
    
    return df, brier_scores

def calculate_calibration_curve(
    df,
    prediction_col,
    timepoint,
    duration_col='duration',
    event_col='event',
    n_bins=10,
    method='quantile'
):
    """
    Calculate calibration curve by comparing predicted vs observed survival probabilities.
    
    Divides predictions into bins (deciles by default) and compares mean predicted
    survival to Kaplan-Meier observed survival at the specified timepoint.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with predictions and survival outcomes
    prediction_col : str
        Column name with survival probability predictions
    timepoint : int
        Time in days to evaluate survival (e.g., 180 for 6 months)
    duration_col : str, default='duration'
        Column name with survival/censoring times
    event_col : str, default='event'
        Column name with event indicators
    n_bins : int, default=10
        Number of bins to divide predictions into
    method : str, default='quantile'
        Binning method: 'quantile' for equal-sized bins or 'uniform' for equal-width bins
        
    Returns:
    --------
    pd.DataFrame : Calibration data with columns:
        - bin: Bin interval or label
        - predicted: Mean predicted survival probability
        - observed: Observed survival probability (from Kaplan-Meier)
        - n: Number of patients in bin
        - obs_lo, obs_hi: 95% CI for observed survival (Greenwood exponential formula)
    """
    # Create bins based on method
    if method == 'quantile':
        bins = pd.qcut(df[prediction_col], q=n_bins, duplicates='drop')
    elif method == 'uniform':
        bins = pd.cut(df[prediction_col], bins=n_bins, duplicates='drop')
    else:
        raise ValueError("method must be 'quantile' or 'uniform'")
    
    bin_categories = bins.cat.categories
    calibration_data = []
    
    for bin_interval in bin_categories:
        mask = (bins == bin_interval)
        
        # Skip if no patients in this bin
        if mask.sum() == 0:
            continue
        
        # Fit Kaplan-Meier to get observed survival
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, duration_col], df.loc[mask, event_col])
        observed_surv = kmf.predict(timepoint)

        # Calcualte 95% CI for observed survival 
        obs_lo = float(np.interp(timepoint, kmf.confidence_interval_.index, kmf.confidence_interval_['KM_estimate_lower_0.95']))
        obs_hi = float(np.interp(timepoint, kmf.confidence_interval_.index, kmf.confidence_interval_['KM_estimate_upper_0.95']))
        
        # Calculate mean predicted survival
        predicted_surv = df.loc[mask, prediction_col].mean()
        
        calibration_data.append({
            'bin': str(bin_interval),
            'predicted': predicted_surv,
            'observed': observed_surv,
            'n': mask.sum(),
            'obs_lo': obs_lo,
            'obs_hi': obs_hi
        })
    
    cal_df = pd.DataFrame(calibration_data)
    return cal_df