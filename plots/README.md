# Visualizations of Estimated Probabilities

This folder contains visual outputs of estimated probabilities and their deviations from the true probabilities under different shapes of true probability distributions (TDS):

- **U-shaped**
- **Bell-shaped**
- **Asymmetric**
- **Uniform**

Each figure includes:
- Histograms of predicted probabilities (`all_models_histograms.*`)
- Scatter plots of prediction errors (`all_models_errors.*`)

These plots allow quick visual comparison of how well different classifiers estimate the true probabilities.

## Files

- `all_models_histograms.png/pdf`: One figure with histograms for each model × distribution
- `all_models_errors.png/pdf`: One figure with error scatter plots for each model × distribution

To reproduce: run `plot_estimated_probabilities.R` from the root directory.
