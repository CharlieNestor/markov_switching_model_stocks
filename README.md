# Regime Switching Models for Financial Time Series

This project implements **Regime Switching Models** to analyze stock returns and identify distinct market environments (e.g., Bull vs. Bear markets). It combines statistical rigor with practical financial application, offering two distinct methodological approaches to regime detection:

*   **Bayesian Inference**: Captures parameter uncertainty using MCMC (PyMC).
*   **Frequentist Estimation**: Utilizes Maximum Likelihood and the EM algorithm (hmmlearn).

## Project Structure

### 1. Exploratory Data Analysis ([`EDA_analysis.ipynb`](EDA_analysis.ipynb))

Comprehensive statistical groundwork providing context for model design:
- Distribution analysis and statistical tests
- Volatility clustering and autocorrelation patterns

For detailed discussion, check out this article [Understanding Financial Time Series](https://medium.com/@carlo.baroni.89/understanding-financial-time-series-a-statistical-deep-dive-cd4ea99d299c).

### 2. Bayesian MS Model ([`MSModel_2Regimes.ipynb`](MSModel_2Regimes.ipynb))
A robust Bayesian implementation focusing on uncertainty quantification:
- **Core**: Two-regime Markov Switching Model with regime-specific mean and volatility.
- **Inference**: Full posterior distributions via NUTS sampler (PyMC).
- **Features**: Captures volatility persistence and provides probability intervals for all parameters.

### 3. Frequentist HMM ([`hmm_model.ipynb`](hmm_model.ipynb))
A flexible Hidden Markov Model approach focusing on likelihood maximization and forecasting:
- **Core**: Gaussian Mixture HMM (GMM-HMM) with dynamically learned transition matrices.
- **Method**: Expectation-Maximization (EM) algorithm for parameter estimation.
- **Features**: 
    - Regime decoding (Viterbi/Smoothing)
    - One-step-ahead density forecasting
    - Empirical vs. Theoretical mixture distribution analysis

### 4. Walk-Forward Analysis ([`hmm_walk_forward.ipynb`](hmm_walk_forward.ipynb))
A comprehensive backtesting engine that simulates real-world performance by iteratively re-estimating the model and predicting future market states:
- **Label Switching & Consistency**: Employs a utility-based sorting algorithm to resolve state permutations and ensure regimes remain economically consistent over time.
- **Forecast Validation**: Systematically evaluates the accuracy of "next-day" predictions against ex-post smoothed beliefs.
- **Investment Strategies**: Implements and evaluates out-of-sample portfolio allocations, comparing naive regime-switching approaches with more advanced strategies.

### 5. Supporting Modules
- [`stock.py`](stock.py): Data handling and preprocessing wrapper.
- [`aux.py`](aux.py): Analysis utilities and diagnostic functions.

## Key Features
- **Dual Perspective**: Compare results from Bayesian (probabilistic) and Frequentist (likelihood-based) models.
- **Regime Identification**: Automatically segments time series into volatility regimes.
- **Forecasting**: Generates predictive distributions conditioning on the current market state.
- **Diagnostics**: Includes convergence checks and goodness-of-fit visualisations.

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
- hmmlearn (Frequentist HMM)
- ArviZ (diagnostics)
- pandas, numpy (data handling)
- matplotlib, seaborn (visualization)
- yfinance (data retrieval)
