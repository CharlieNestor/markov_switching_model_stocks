# Markov Switching Model for Financial Time Series

A Bayesian implementation of a Markov Switching Model for analyzing stock returns and identifying distinct market regimes. The model captures regime-switching behavior, volatility clustering, and non-normal return distributions commonly observed in financial markets.

## Overview

This project implements a Hidden Markov Model (HMM) to identify and characterize different market regimes in stock returns. The approach combines:

- Latent regime states following a Markov process
- Regime-specific return and volatility dynamics
- Full Bayesian inference using PyMC

## Project Structure

### 1. Exploratory Data Analysis ([`EDA_analysis.ipynb`](EDA_analysis.ipynb))

Comprehensive statistical analysis providing context for model design:
- Distribution analysis and statistical tests
- Volatility clustering and persistence patterns
- Autocorrelation structure in returns

For detailed discussion of the EDA methodology and findings, check out this article [Understanding Financial Time Series](https://medium.com/@carlo.baroni.89/understanding-financial-time-series-a-statistical-deep-dive-cd4ea99d299c).

### 2. Markov Switching Model ([`MSModel_2Regimes.ipynb`](MSModel_2Regimes.ipynb))

Core implementation featuring:
- **Two-regime model** with hidden Markov chain
- **Regime-specific parameters**: mean returns, volatility dynamics, and autoregressive components
- **Volatility modeling**: Base level + ARCH component + regime-specific memory
- **Bayesian inference**: Full posterior distributions via MCMC (NUTS sampler)

### 3. Supporting Modules

- [`stock.py`](stock.py): Data handling and preprocessing
- [`aux.py`](aux.py): Analysis utilities and diagnostic functions

## Key Features

- Identifies bull and bear market regimes
- Models regime persistence via transition probabilities
- Captures volatility clustering within regimes
- Provides uncertainty quantification for all parameters
- Includes diagnostic tools for convergence assessment


## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/markov_switching.git
cd markov_switching
pip install -r requirements.txt
```

## Requirements

See [`requirements.txt`](requirements.txt) for full dependencies. Key packages:
- PyMC (Bayesian modeling)
- ArviZ (diagnostics)
- pandas, numpy (data handling)
- matplotlib, seaborn (visualization)
- yfinance up to date (data retrieval)

