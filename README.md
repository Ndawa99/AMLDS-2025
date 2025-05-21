# AMLDS-2025
# Calibration Simulation Study

This repository contains R code to simulate and evaluate calibration metrics under various shapes of true probability distributions.

## Structure

- `AMLDS metrics.R`: Main script to run simulations and generate metrics.
- `AMLDS graphiques.R`: Main script to run simulations and generate graphics.
- `utils`: Contains helper functions (AUC, Brier score, etc.).
- `output/`: Contains simulation results (tables, CSV).
- `plots/`: (Optional) Visualizations of the results.

## Dependencies

Install required packages:
```r
install.packages(c("dplyr", "randomForest", "ggplot2", "caret", "faux",
                   "kernlab", "naivebayes", "moments", "xtable", "nnet", "doParallel"))
