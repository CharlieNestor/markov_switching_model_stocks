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