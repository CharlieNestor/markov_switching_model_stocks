import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Tuple, Dict
import arviz as az
from statsmodels.tsa.stattools import acf, pacf


### FUNCTIONS FOR THE ANALYSIS OF THE MARKOV SWITCHING MODEL


# DIAGNOSTIC CHECKS


def plot_trace(trace: az.InferenceData, 
            var_names: list[str], 
            burn_in: bool = False, 
            vector: bool = False) -> plt.figure:
    """
    Custom function to plot traces and posterior distributions
    : param trace: InferenceData object including the MCMC trace
    : param var_names: list of variable names to plot (or single variable name if vector=True)
    : param burn_in: boolean, if True, include warmup (burn-in) samples in the trace plot
    : param vector: boolean, if True, treat the input as a vector parameter
    : return: matplotlib figure
    """
    if vector:
        # Handle single vector parameter
        var_name = var_names if isinstance(var_names, str) else var_names[0]
        samples = trace.posterior[var_name].values
        vector_dim = samples.shape[-1]
        fig, axes = plt.subplots(vector_dim, 2, figsize=(15, 4*vector_dim))
        
        # Make axes 2D if it's not already
        if vector_dim == 1:
            axes = axes.reshape(1, -1)
        
        for dim in range(vector_dim):
            # Extract samples for this dimension
            posterior_samples = samples[..., dim]
            
            # Plot trace
            ax_trace = axes[dim, 0]
            
            if burn_in:
                warmup_samples = trace.warmup_posterior[var_name].values[..., dim]
                for chain in range(warmup_samples.shape[0]):
                    # Burnin period
                    ax_trace.plot(warmup_samples[chain], 
                                alpha=0.3, color='gray', 
                                label='Burnin' if chain==0 else None)
                    
                    # Posterior samples
                    ax_trace.plot(range(warmup_samples.shape[1], 
                                      warmup_samples.shape[1] + posterior_samples.shape[1]),
                                posterior_samples[chain], 
                                alpha=0.7,
                                label=f'Chain {chain+1}')
            else:
                for chain in range(posterior_samples.shape[0]):
                    ax_trace.plot(posterior_samples[chain], 
                                alpha=0.7,
                                label=f'Chain {chain+1}')
            
            ax_trace.set_title(f'Trace Plot - {var_name}[{dim}]')
            ax_trace.set_xlabel('Iteration')
            ax_trace.set_ylabel('Value')
            ax_trace.legend()
            
            # Posterior distribution plot
            ax_hist = axes[dim, 1]
            post_samples = posterior_samples.flatten()
            
            # Plot histogram and statistics
            ax_hist.hist(post_samples, bins=100, density=True, alpha=0.6, color='skyblue')
            
            mean = np.mean(post_samples)
            median = np.median(post_samples)
            hdi = az.hdi(post_samples, hdi_prob=0.95)
            
            ax_hist.axvline(mean, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean:.4f}')
            ax_hist.axvline(median, color='green', linestyle='--', alpha=0.8, label=f'Median: {median:.4f}')
            ax_hist.axvspan(hdi[0], hdi[1], alpha=0.2, color='gray', 
                          label=f'95% HDI: [{hdi[0]:.4f}, {hdi[1]:.4f}]')
            
            ax_hist.set_title(f'Posterior Distribution - {var_name}[{dim}]')
            ax_hist.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    else:
        # Handling a list of non-vector parameters
        n_vars = len(var_names)
        fig, axes = plt.subplots(n_vars, 2, figsize=(15, 4*n_vars))
        
        # Make axes 2D if it's not already
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        for idx, var in enumerate(var_names):
            posterior_samples = trace.posterior[var].values
            
            # Plot trace
            ax_trace = axes[idx, 0]
            
            if burn_in:
                warmup_samples = trace.warmup_posterior[var].values
                for chain in range(warmup_samples.shape[0]):
                    ax_trace.plot(warmup_samples[chain], 
                                alpha=0.3, color='gray', 
                                label='Burnin' if chain==0 else None)
                    ax_trace.plot(range(warmup_samples.shape[1], 
                                      warmup_samples.shape[1] + posterior_samples.shape[1]),
                                posterior_samples[chain], 
                                alpha=0.7,
                                label=f'Chain {chain+1}')
            else:
                for chain in range(posterior_samples.shape[0]):
                    ax_trace.plot(posterior_samples[chain], 
                                alpha=0.7,
                                label=f'Chain {chain+1}')
            
            ax_trace.set_title(f'Trace Plot - {var}')
            ax_trace.set_xlabel('Iteration')
            ax_trace.set_ylabel('Value')
            ax_trace.legend()
            
            # Posterior distribution plot
            ax_hist = axes[idx, 1]
            post_samples = posterior_samples.flatten()
            
            ax_hist.hist(post_samples, bins=100, density=True, alpha=0.6, color='skyblue')
            
            mean = np.mean(post_samples)
            median = np.median(post_samples)
            hdi = az.hdi(post_samples, hdi_prob=0.95)
            
            ax_hist.axvline(mean, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean:.4f}')
            ax_hist.axvline(median, color='green', linestyle='--', alpha=0.8, label=f'Median: {median:.4f}')
            ax_hist.axvspan(hdi[0], hdi[1], alpha=0.2, color='gray', 
                          label=f'95% HDI: [{hdi[0]:.4f}, {hdi[1]:.4f}]')
            
            ax_hist.set_title(f'Posterior Distribution - {var}')
            ax_hist.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig


def perform_diagnostic_checks(trace: az.InferenceData, 
                            verbose: bool = False,
                            exclude_vars: list = None) -> pd.DataFrame:
    """
    Performs diagnostic checks and generates summary statistics for all parameters.
    : param trace: ArviZ InferenceData object containing the MCMC trace
    : param verbose: boolean, if True, print the list of parameters
    : param exclude_vars: list of variables to ignore from the analysis
    : return: pandas DataFrame with summary statistics for all parameters
    """
    parameters = []
    n_params = 0
    exclude_vars = exclude_vars or []
        
    # Get all parameter names from the trace
    for var in trace.posterior.variables:
        if (var not in ['chain', 'draw'] and \
           'state' not in var and \
           'initial' not in var and \
           'probs' not in var and \
           var not in exclude_vars):
            if len(trace.posterior[var].shape) == 2:
                n_params += 1
                parameters.append(var)
            elif len(trace.posterior[var].shape) > 2:
                n_params += trace.posterior[var].shape[-1]
                parameters.append(var)
    if verbose:
        print(f"Number of parameters: {n_params}")
        print(f"Parameters: {parameters}")

    # Generate summary statistics
    summary = az.summary(trace, 
                        var_names=parameters,
                        hdi_prob=0.95,
                        round_to=6,
                        stat_funcs={'mode': lambda x: float(pd.Series(x).mode()[0]),
                                  'median': lambda x: float(pd.Series(x).median())})
    
    print("\nPosterior Distribution Statistics:")
    print("="*80)
    print(summary)
    
    return summary


# INFORMATION CRITERIA


def calculate_information_criteria(trace: az.InferenceData, 
                                n_samples: int, 
                                verbose: bool = False,
                                exclude_vars: list = None) -> dict:
    """
    Calculate AIC, BIC, DIC, and WAIC for the model

    :param trace: ArviZ InferenceData object
    :param n_samples: number of observations
    :return: dictionary with AIC, BIC, DIC, and WAIC and number of parameters
    """
    # Get number of parameters (excluding states)
    exclude_vars = exclude_vars or []
    params_list = []
    n_params = 0
    for var in trace.posterior.variables:
        if var != 'chain' and \
            var != 'draw' and \
            'state' not in var and \
            'initial' not in var and \
            var not in exclude_vars:
            if len(trace.posterior[var].shape) == 2:
                n_params +=1
                params_list.append(var)
            else:
                if len(trace.posterior[var].shape) >2:
                    n_params += trace.posterior[var].shape[-1]
                    params_list.append(var)
    if verbose:
        print(params_list)
    
    # Get log likelihood values
    log_lik = trace.log_likelihood.returns_obs      # shape: (chains, draws, time points)

    # Calculate mean log likelihood for each time point
    mean_log_lik = log_lik.mean(dim=("chain", "draw")).sum()
    
    # Calculate AIC and BIC
    aic = -2 * mean_log_lik + 2 * n_params
    bic = -2 * mean_log_lik + np.log(n_samples) * n_params

    # CALCULATE DIC
    # First sum over time points to get deviance per draw
    # This gives us the total deviance for each MCMC sample
    deviance_samples = -2 * log_lik.sum(dim="returns_obs_dim_0")
    # Then take the mean across MCMC samples
    mean_deviance = deviance_samples.mean(dim=("chain", "draw"))
    # Approximate deviance at the posterior mean using the mean log likelihood
    deviance_at_means = -2* mean_log_lik
    # Calculate effective number of parameters (pD)
    # pD = 2 * (log p(y|θ_mean) - mean(log p(y|θ)))
    p_d = 0.5 * (mean_deviance - deviance_at_means)
    
    # DIC = deviance_mean + 2 * pD
    dic = deviance_at_means + 2 * p_d

    # Calculate WAIC
    # First, compute point-wise log likelihood
    # This is the log-likelihood for each observation, chain, and draw
    point_log_lik = log_lik
    
    # Compute mean log predictive density (lppd)
    # exp(log_lik) gives us the likelihood, then we take the mean across chains and draws
    lppd = np.log(np.exp(point_log_lik).mean(dim=("chain", "draw"))).sum()
    
    # Compute effective number of parameters (pWAIC)
    # pWAIC is the sum of the variances of the log-likelihood for each observation
    var_log_lik = point_log_lik.var(dim=("chain", "draw"))
    p_waic = var_log_lik.sum()
    
    waic = -2 * (lppd - p_waic)
    
    return {
        'n_params': n_params,
        'AIC': round(float(aic), 3),
        'BIC': round(float(bic), 3),
        'DIC': round(float(dic), 3),
        'WAIC': round(float(waic), 3)
    }


# REGIME PERSISTENCE

def calculate_aggregate_metrics(values):
    """
    Calculate distribution statistics across chains with uncertainty quantification.

    :param values: List of values from different chains 
    :return: Dictionary of distribution statistics
    """
    values = np.array(values)
    return {
        'mean': round(np.mean(values), 6),
        'median': round(np.median(values), 6),
        'std': round(np.std(values), 6),
        'percentile_95': (round(float(np.percentile(values, 2.5)), 6), 
                            round(float(np.percentile(values, 97.5)), 6)),
    }


def analyze_regime_persistence(trace: az.InferenceData, n_regimes: int = 3) -> dict:
    """
    Analyze regime persistence for either 2-regime or 3-regime models
    :param trace: ArviZ InferenceData object containing the MCMC trace
    :param n_regimes: int, optional (default=3)
        Number of regimes (2 or 3)
    :return: dict containing persistence metrics with uncertainty quantification
    """
    if n_regimes not in [2, 3]:
        raise ValueError("Only 2 or 3 regimes are supported")

    # Extract states
    states = trace.posterior['states'].values  # Shape: (chains, draws, time)
    
    # Extract transition probabilities based on regime count
    if n_regimes == 2:
        # For 2 regimes: scalar p00, p11
        p00_samples = trace.posterior['p00'].values
        p11_samples = trace.posterior['p11'].values
        transition_probs = {'p00': p00_samples, 'p11': p11_samples}
    else:
        # For 3 regimes: vector p0, p1, p2
        p0_samples = trace.posterior['p0'].values
        p1_samples = trace.posterior['p1'].values
        p2_samples = trace.posterior['p2'].values
        transition_probs = {
            'p00': p0_samples[..., 0],
            'p11': p1_samples[..., 1],
            'p22': p2_samples[..., 2]
        }
    
    chain_metrics = []
    
    for chain in range(states.shape[0]):    # for each chain
        chain_result = {
            'expected_durations': [],
            'empirical_durations': [],
            'transition_probs': []
        }
        
        for draw in range(states.shape[1]):    # for each draw
            draw_states = states[chain, draw]
            
            # Get transition probabilities for this draw
            if n_regimes == 2:
                p00 = p00_samples[chain, draw]
                p11 = p11_samples[chain, draw]
                probs = (p00, p11)
            else:
                p00 = p0_samples[chain, draw, 0]
                p11 = p1_samples[chain, draw, 1]
                p22 = p2_samples[chain, draw, 2]
                probs = (p00, p11, p22)
            
            # Calculate expected durations (from transition probabilities)
            expected_durations = tuple(1 / (1 - p) for p in probs)  # Formula for expected duration
            chain_result['expected_durations'].append(expected_durations)
            chain_result['transition_probs'].append(probs)
            
            # Calculate empirical durations (from states)
            regime_changes = np.diff(draw_states)  # Difference between consecutive states
            change_points = np.where(regime_changes != 0)[0] + 1  # Index of regime changes
            regime_durations = np.split(draw_states, change_points)  # Split states into regimes
            
            # Calculate durations for each regime (from states)
            empirical_durations = []
            for regime in range(n_regimes):
                # get the consecutive durations of the regime (how many consecutive states are in the regime)
                durations = [len(r) for r in regime_durations if r[0] == regime]
                # calculate the mean duration
                mean_duration = np.mean(durations) if durations else np.nan
                empirical_durations.append(mean_duration)
            
            chain_result['empirical_durations'].append(tuple(empirical_durations))
        
        chain_metrics.append(chain_result)
    
    # Aggregate results
    results = {}
    
    # Process each regime
    for regime in range(n_regimes):
        regime_key = f'regime_{regime}'
        p_key = f'p{regime}{regime}'

        # For empirical durations: combine all chains and draws
        empirical_values = [m['empirical_durations'][i][regime] 
                          for m in chain_metrics 
                          for i in range(len(m['empirical_durations']))]
        
        # For expected durations: use transition probabilities
        p_samples = transition_probs[p_key].flatten()
        expected_values = 1/(1-p_samples)
        
        results[regime_key] = {
            'empirical': calculate_aggregate_metrics(empirical_values),
            'expected': calculate_aggregate_metrics(expected_values),
            'transition_probs': calculate_aggregate_metrics(p_samples)
        }
    
    return results


def calculate_regime_classification(states_posterior, n_regimes: int = 3):
    """
    Calculate regime classification measure (RCM) for 3 regimes with uncertainty across chains.
    A higher RCM (closer to 100) indicates better regime classification.
    """
    chain_rcms = []
    
    for chain in range(states_posterior.shape[0]):
        # Step 1: Calculate regime probabilities for each time point
        state_probs = np.zeros((n_regimes, states_posterior.shape[2]))  # n_regimes x time
        for t in range(states_posterior.shape[2]):
            for s in range(n_regimes):
                # Calculate probability of being in regime s at time t
                state_probs[s, t] = np.mean(states_posterior[chain, :, t] == s)
        
        # Step 2: Calculate pairwise products sum
        pairwise_sum = 0
        for i in range(n_regimes):
            for j in range(i + 1, n_regimes):
                # Multiply probabilities of different regimes
                pairwise_sum += state_probs[i] * state_probs[j]
        
        # Step 3: Calculate RCM for this chain
        # General formula: 100 * n * (n-1) * mean(sum of pairwise products)
        rcm = 100 * n_regimes * (n_regimes - 1) * np.mean(pairwise_sum)
        chain_rcms.append(rcm)
    
    return calculate_aggregate_metrics(chain_rcms)



### POSTERIOR PREDICTIVE CHECKS

### PREDICTION ACCURACY


def calculate_accuracy_metrics(returns: np.ndarray, posterior_pred: az.InferenceData) -> dict:
    """
    Calculate performance metrics accounting for posterior uncertainty
    :param returns: observed returns
    :param posterior_pred: posterior predictive samples from PyMC
    :return: dictionary containing metrics with uncertainty
    Note: Each metric is calculated for each draw in each chain,
    resulting in n_chains * n_draws total values per metric
    """
    # Get predicted returns array (shape: chains x draws x time)
    pred_returns = posterior_pred.posterior_predictive.returns_obs
    
    # Initialize metrics dictionary
    metrics = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'directional_accuracy': [],
        'hit_rate': []
    }
    
    # Iterate over chains and draws (total iterations = n_chains * n_draws)
    for chain in range(pred_returns.chain.size):
        for draw in range(pred_returns.draw.size):
            # Get predictions for this chain and draw
            preds = pred_returns.isel(chain=chain, draw=draw)
            
            # Calculate metrics for this draw
            residuals = returns - preds
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            
            # Directional accuracy (for consecutive returns)
            pred_direction = np.diff(preds) > 0
            actual_direction = np.diff(returns) > 0
            dir_accuracy = np.mean(pred_direction == actual_direction)
            
            # Hit rate (for positive/negative returns)
            pred_positive = preds > 0
            actual_positive = returns > 0
            hit_rate = np.mean(pred_positive == actual_positive)
            
            # Store metrics for this draw
            metrics['mse'].append(mse)
            metrics['rmse'].append(rmse)
            metrics['mae'].append(mae)
            metrics['directional_accuracy'].append(dir_accuracy)
            metrics['hit_rate'].append(hit_rate)
    
    # Calculate summary statistics across all draws
    summary = {}
    for metric_name, values in metrics.items():
        summary[metric_name] = {
            'mean': round(np.mean(values), 6),
            'std': round(np.std(values), 6),
            'median': round(np.median(values), 6),
            '2.5%': round(np.percentile(values, 2.5), 6),
            '97.5%': round(np.percentile(values, 97.5), 6)
        }

        # Add additional statistics for directional metrics
        if metric_name in ['directional_accuracy', 'hit_rate']:
            better_than_random = np.mean(np.array(values) > 0.5)
            summary[metric_name]['prob_better_than_random'] = round(better_than_random, 6)
    
    return summary