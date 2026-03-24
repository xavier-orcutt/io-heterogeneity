import pandas as pd
import numpy as np

def calculate_treatment_effect_curve_rmst(
    model,
    baseline_var,
    baseline_values,
    treatment_var='treatment'
):
    """
    Calculate treatment effect for RMST across a range of baseline values from interaction model.
    
    Computes treatment effect, standard errors using delta method, and confidence intervals
    for models with treatment-by-covariate interaction.
    
    Parameters:
    -----------
    model : statsmodels regression result
        Fitted model with treatment and treatment:covariate interaction
    baseline_var : str
        Name of the baseline variable in the interaction
    baseline_values : array-like
        Values at which to evaluate treatment effect
    treatment_var : str, default='treatment'
        Name of treatment variable
        
    Returns:
    --------
    pd.DataFrame with columns:
        - baseline: baseline variable values
        - treatment_effect: predicted treatment effect
        - se: standard error of treatment effect
        - ci_lower: lower CI bound
        - ci_upper: upper CI bound
        
    Example:
    --------
    >>> effect_df = calculate_treatment_effect_curve(
    ...     model=wls_model,
    ...     baseline_var='psurv_180_calibrated',
    ...     baseline_values=np.linspace(0.3, 0.9, 100)
    ... )
    """
    baseline_values = np.asarray(baseline_values)
    
    # Predict for treatment=1
    pred_treat = model.predict(pd.DataFrame({
        treatment_var: 1,
        baseline_var: baseline_values
    }))
    
    # Predict for treatment=0
    pred_control = model.predict(pd.DataFrame({
        treatment_var: 0,
        baseline_var: baseline_values
    }))
    
    # Treatment effect
    treatment_effect = pred_treat - pred_control
    
    # Delta method variance calculation
    interaction_term = f'{treatment_var}:{baseline_var}'
    
    var_treatment = model.cov_params().loc[treatment_var, treatment_var]
    var_interaction = model.cov_params().loc[interaction_term, interaction_term]
    cov_treat_int = model.cov_params().loc[treatment_var, interaction_term]
    
    # Var(β₁ + β₃*X) = Var(β₁) + X²*Var(β₃) + 2X*Cov(β₁,β₃)
    var_effect = (var_treatment + 
                  (baseline_values**2) * var_interaction + 
                  2 * baseline_values * cov_treat_int)
    se_effect = np.sqrt(var_effect)
    
    # Confidence intervals
    ci_lower = treatment_effect - 1.96 * se_effect
    ci_upper = treatment_effect + 1.96 * se_effect
    
    return pd.DataFrame({
        'baseline': baseline_values,
        'treatment_effect': treatment_effect,
        'se': se_effect,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })


def calculate_treatment_effect_curve_hr(
    cph_model,
    interaction_term,
    baseline_values,
    treatment_var='treatment'
):
    """
    Calculate hazard ratio treatment effect across a range of baseline values from Cox interaction model.
    
    Computes HR, standard errors using delta method, and confidence intervals
    for Cox models with treatment-by-covariate interaction.
    
    Parameters:
    -----------
    cph_model : lifelines CoxPHFitter result
        Fitted Cox model with treatment and treatment_x_risk interaction
    interaction_term : str
        Name of the interaction_term (e.g., 'treatment_x_risk')
    baseline_values : array-like
        Values at which to evaluate HR
    treatment_var : str, default='treatment'
        Name of treatment variable
        
    Returns:
    --------
    pd.DataFrame with columns:
        - baseline: baseline variable values
        - hr: hazard ratio
        - log_hr: log hazard ratio
        - se_log_hr: standard error of log HR
        - hr_lower: lower CI bound for HR
        - hr_upper: upper CI bound for HR
        
    Example:
    --------
    >>> hr_df = calculate_treatment_effect_curve_hr(
    ...     cph_model=cph,
    ...     interaction_term='treatment_x_risk',
    ...     baseline_values=np.linspace(0.3, 0.9, 100)
    ... )
    """
    baseline_values = np.asarray(baseline_values)
    
    # Get coefficients
    beta_treatment = cph_model.params_[treatment_var]
    beta_interaction = cph_model.params_[interaction_term]
    
    # Get variance-covariance matrix
    vcov = cph_model.variance_matrix_
    
    # Initialize output arrays
    log_hr_values = []
    se_log_hr_values = []
    hr_values = []
    hr_lower = []
    hr_upper = []
    
    for risk in baseline_values:
        # Log-HR: β₁ + β₃*risk
        log_hr = beta_treatment + beta_interaction * risk
        
        # Variance using delta method
        # Var(β₁ + β₃*X) = Var(β₁) + X²*Var(β₃) + 2X*Cov(β₁,β₃)
        var_log_hr = (
            vcov.loc[treatment_var, treatment_var] +
            risk**2 * vcov.loc[interaction_term, interaction_term] +
            2 * risk * vcov.loc[treatment_var, interaction_term]
        )
        
        se_log_hr = np.sqrt(var_log_hr)
        
        # Calculate HR and CI on HR scale
        hr = np.exp(log_hr)
        hr_low = np.exp(log_hr - 1.96 * se_log_hr)
        hr_high = np.exp(log_hr + 1.96 * se_log_hr)
        
        # Store results
        log_hr_values.append(log_hr)
        se_log_hr_values.append(se_log_hr)
        hr_values.append(hr)
        hr_lower.append(hr_low)
        hr_upper.append(hr_high)
    
    return pd.DataFrame({
        'baseline': baseline_values,
        'log_hr': log_hr_values,
        'se_log_hr': se_log_hr_values,
        'hr': hr_values,
        'hr_lower': hr_lower,
        'hr_upper': hr_upper
    })

