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
    Calculate AIC and BIC for the model
    :param trace: ArviZ InferenceData object
    :param n_samples: number of observations
    :return: dictionary with AIC, BIC, and n_params
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
