import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from joblib import Parallel, delayed

def pseudo_observation_rmst_single(i, time, event, tau, rmst_full, n):
    """
    Calculate pseudo-observation for a single patient (patient i)
    
    Parameters:
    -----------
    i : int
        Index of patient to leave out
    time : np.array
        Array of survival times
    event : np.array
        Array of event indicators (1=event, 0=censored)
    tau : float
        Restriction time for RMST
    rmst_full : float
        RMST calculated from full dataset
    n : int
        Total number of patients
        
    Returns:
    --------
    float : Pseudo-observation for patient i
    """
    # Create leave-one-out dataset (exclude patient i)
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    
    time_loo = time[mask]
    event_loo = event[mask]
    
    # Fit Kaplan-Meier on leave-one-out dataset
    kmf_loo = KaplanMeierFitter()
    kmf_loo.fit(time_loo, event_loo)
    
    # Calculate RMST without patient i
    rmst_loo = restricted_mean_survival_time(kmf_loo, t=tau)
    
    # Jackknife formula for pseudo-observation
    pseudo_obs = n * rmst_full - (n - 1) * rmst_loo
    
    return pseudo_obs

def pseudo_observations_rmst(time, event, tau):
    """
    Calculate pseudo-observations for RMST using jackknife method in parallel. 
    
    This creates individual-level RMST estimates that account for censoring
    and can be used as outcomes in regression models.
    
    Parameters:
    -----------
    time : array-like
        Observed survival or censoring times
    event : array-like
        Event indicator (1 = event observed, 0 = censored)
    tau : float
        Restriction time for RMST (e.g., 1095 for 36 months)
        
    Returns:
    --------
    np.array : Pseudo-observations (one per patient)
    
    Example:
    --------
    >>> pseudo_rmst = calculate_pseudo_observations(
    ...     df['time'].values,
    ...     df['event'].values,
    ...     tau=1095,
    ... )
    >>> df['rmst_pseudo'] = pseudo_rmst
    """
    # Convert to numpy arrays
    time = np.asarray(time)
    event = np.asarray(event)
    n = len(time)
    
    # Step 1: Calculate RMST from full dataset    
    kmf_full = KaplanMeierFitter()
    kmf_full.fit(time, event)
    rmst_full = restricted_mean_survival_time(kmf_full, t=tau)
    
    # Step 2: Calculate leave-one-out pseudo-observations
    pseudo_obs = Parallel(n_jobs = -1)(
        delayed(pseudo_observation_rmst_single)(i, time, event, tau, rmst_full, n)
        for i in range(n)
    )
    pseudo_obs = np.array(pseudo_obs)
    
    return pseudo_obs

def pseudo_observation_psurv_single(i, time, event, tau, S_t_full, n):
    """
    Calculate pseudo-observation for a single patient (patient i)
    
    Parameters:
    -----------
    i : int
        Index of patient to leave out
    time : np.array
        Array of survival times
    event : np.array
        Array of event indicators (1=event, 0=censored)
    tau : float
        Restriction time for probability of survival
    S_t_full : float
        Probability of survival calculated from full dataset
    n : int
        Total number of patients
        
    Returns:
    --------
    float : Pseudo-observation for patient i
    """
    # Create leave-one-out dataset (exclude patient i)
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    
    time_loo = time[mask]
    event_loo = event[mask]
    
    # Fit Kaplan-Meier on leave-one-out dataset
    kmf_loo = KaplanMeierFitter()
    kmf_loo.fit(time_loo, event_loo)
    
    # Calculate survival probability without patient i
    S_t_loo = kmf_loo.predict(tau)
    
    # Jackknife formula for pseudo-observation
    pseudo_obs = n * S_t_full - (n - 1) * S_t_loo
    
    return pseudo_obs

def pseudo_observations_psurv(time, event, tau):
    """
    Calculate pseudo-observations for probability of survival using jackknife method in parallel. 
    
    This creates individual-level probability of survival estimates that account for censoring
    and can be used as outcomes in regression models.
    
    Parameters:
    -----------
    time : array-like
        Observed survival or censoring times
    event : array-like
        Event indicator (1 = event observed, 0 = censored)
    tau : float
        Time for probability of survival (e.g., 1095 for 36 months)
        
    Returns:
    --------
    np.array : Pseudo-observations (one per patient)
    
    Example:
    --------
    >>> pseudo_psurv = pseudo_observations_psurv(
    ...     df['time'].values,
    ...     df['event'].values,
    ...     tau=1095,
    ... )
    >>> df['psurv_pseudo'] = psurv_pseudo
    """
    # Convert to numpy arrays
    time = np.asarray(time)
    event = np.asarray(event)
    n = len(time)
    
    # Step 1: Calculate survival probability from full dataset    
    kmf_full = KaplanMeierFitter()
    kmf_full.fit(time, event)
    S_t_full = kmf_full.predict(tau)
    
    # Step 2: Calculate leave-one-out pseudo-observations
    pseudo_obs = Parallel(n_jobs = -1)(
        delayed(pseudo_observation_psurv_single)(i, time, event, tau, S_t_full, n)
        for i in range(n)
    )
    pseudo_obs = np.array(pseudo_obs)
    
    return pseudo_obs