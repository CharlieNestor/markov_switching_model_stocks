# Bayesian Markov Switching Model for Financial Time Series

This project implements a Bayesian Markov Switching Model for analyzing financial time series data, with a particular focus on capturing the complex dynamics of stock returns. The model employs a full Bayesian approach to identify and characterize different market regimes, providing a robust framework for understanding market behavior.

## Project Overview

Financial markets exhibit several key characteristics that challenge traditional modeling approaches:

1. **Regime-Switching Behavior**: Markets transition between distinct states (e.g., bull markets, bear markets)
2. **Volatility Dynamics**: Returns generally show persistent volatility patterns and clustering
3. **Non-Normal Features**: Returns often display fat tails and excess kurtosis
4. **Mean Reversion**: Returns exhibit varying degrees of mean reversion across regimes

This project addresses these challenges through a Bayesian Hidden Markov Model that:

- Uses latent state variables to identify market regimes
- Implements regime-specific dynamics for returns and volatility
- Incorporates prior knowledge through carefully chosen Bayesian priors
- Provides full posterior distributions for all model parameters

## Key Features

### 1. Exploratory Data Analysis

The EDA component provides a detailed statistical analysis for understanding financial time series behavior:

- **Statistical Tests**: Tests including Jarque-Bera, Augmented Dickey-Fuller, and Ljung-Box
- **Distribution Analysis**: Examination of return distributions and normality
- **Volatility Analysis**: Investigation of volatility clustering, persistence, and leverage effects
- **Time Series Properties**: Analysis of autocorrelation patterns in returns and squared returns

### 2. Bayesian Markov Switching Model

Building on the insights from the EDA, the Markov Switching Model implements a Bayesian approach:

- **Hidden Markov Chain**: Latent state variables following a Markov process
- **Regime-Specific Parameters**:
  - Mean returns with autoregressive components
  - Volatility dynamics with GARCH-like features
  - Transition probabilities between regimes
- **Prior Specifications**:
  - Beta priors for transition probabilities
  - Normal priors for mean returns
  - Half-Normal priors for volatility parameters
  - Stationarity constraints for autoregressive components

### 3. Volatility Modeling

The model implements a volatility modeling approach that:

- Combines base volatility levels with regime-specific dynamics
- Incorporates ARCH-like components responding to recent returns
- Includes a memory component with adaptive weighting
- Ensures stationarity through parameter constraints

### 4. Inference and Analysis

The project provides tools for:

- MCMC sampling with NUTS algorithm
- Posterior predictive analysis
- Regime probability estimation
- Model diagnostics and convergence checks
- Visualization of regime classifications and parameter distributions

## Project Structure

- `EDA_analysis.ipynb`: Comprehensive statistical analysis of financial time series
- `MSModel_2Regimes.ipynb`: Core implementation of the Markov Switching Model
- `stock.py`: Stock data handling and preprocessing
- `aux.py`: Utility functions for analysis and visualization

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/markov_switching.git
cd markov_switching
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

