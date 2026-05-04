# 07_evaluate_adversarial.R

library(tidyverse)
library(tidymodels)
library(ranger)
library(xgboost)
library(vip)

#' Prepare Data for Adversarial Classification
#'
#' @param data Combined real and synthetic dataset 
#' @return A dataset with engineered summary features and a binary target
prepare_adversarial_data <- function(data) {
  # Calculate daily profile features for each home/source
  feature_data <- data |>
    dplyr::group_by(source, home_id) |>
    dplyr::summarize(
      mean_usage = mean(usage, na.rm = TRUE),
      max_peak = max(usage, na.rm = TRUE),
      zero_proportion = mean(usage == 0, na.rm = TRUE),
      usage_variance = var(usage, na.rm = TRUE),
      p90_usage = quantile(usage, 0.90, na.rm = TRUE),
      .groups = "drop"
    ) |>
    # Handle NA variance if only one data point exists
    dplyr::mutate(
      usage_variance = tidyr::replace_na(usage_variance, 0),
      is_synthetic = factor(ifelse(source == "Synthetic", "Yes", "No"), levels = c("Yes", "No"))
    ) |>
    dplyr::select(-source, -home_id)
  
  return(feature_data)
}

# ------------------------------------------------------------------------------
# Adversarial Evaluation Pipeline
# ------------------------------------------------------------------------------

evaluate_adversaries <- function(combined_data) {
  
  # 1. Data Preparation
  adv_data <- prepare_adversarial_data(combined_data)
  
  # Split data into training and holdout test sets
  set.seed(42)
  data_split <- rsample::initial_split(adv_data, prop = 0.8, strata = is_synthetic)
  train_data <- rsample::training(data_split)
  test_data  <- rsample::testing(data_split)
  
  # 2. Modeling Pipeline setup
  # 10-fold Cross-Validation
  cv_folds <- rsample::vfold_cv(train_data, v = 10, strata = is_synthetic)
  
  # Preprocessing Recipe
  adv_recipe <- recipes::recipe(is_synthetic ~ ., data = train_data) |>
    recipes::step_normalize(recipes::all_numeric_predictors())
  
  # 3. Classifiers Specification
  rf_spec <- parsnip::rand_forest(trees = 500, mtry = tune(), min_n = tune()) |>
    parsnip::set_engine("ranger", importance = "impurity") |>
    parsnip::set_mode("classification")
  
  gbm_spec <- parsnip::boost_tree(trees = 500, tree_depth = tune(), learn_rate = tune()) |>
    parsnip::set_engine("xgboost") |>
    parsnip::set_mode("classification")
  
  # Workflows
  rf_wf <- workflows::workflow() |>
    workflows::add_recipe(adv_recipe) |>
    workflows::add_model(rf_spec)
  
  gbm_wf <- workflows::workflow() |>
    workflows::add_recipe(adv_recipe) |>
    workflows::add_model(gbm_spec)
  
  # Grid Search / Tuning
  # (Limiting grid to 5 to avoid overly long run times)
  message("Tuning Random Forest...")
  rf_res <- tune::tune_grid(
    rf_wf,
    resamples = cv_folds,
    grid = 5,
    control = tune::control_grid(save_pred = TRUE)
  )
  
  message("Tuning GBM...")
  gbm_res <- tune::tune_grid(
    gbm_wf,
    resamples = cv_folds,
    grid = 5,
    control = tune::control_grid(save_pred = TRUE)
  )
  
  # Extract best configurations based on AUC-ROC
  best_rf <- tune::select_best(rf_res, metric = "roc_auc")
  best_gbm <- tune::select_best(gbm_res, metric = "roc_auc")
  
  # Finalize workflows
  final_rf_wf <- tune::finalize_workflow(rf_wf, best_rf)
  final_gbm_wf <- tune::finalize_workflow(gbm_wf, best_gbm)
  
  # Evaluate on holdout folds
  message("Evaluating best models on test set...")
  rf_fit <- tune::last_fit(final_rf_wf, data_split)
  gbm_fit <- tune::last_fit(final_gbm_wf, data_split)
  
  # 4. Evaluation Metrics
  rf_metrics <- tune::collect_metrics(rf_fit)
  gbm_metrics <- tune::collect_metrics(gbm_fit)
  
  cat("\n============================================\n")
  cat("Random Forest Test Set Metrics:\n")
  print(rf_metrics |> dplyr::select(.metric, .estimate))
  
  cat("\nGBM Test Set Metrics:\n")
  print(gbm_metrics |> dplyr::select(.metric, .estimate))
  cat("============================================\n")
  cat("(An ROC-AUC close to 0.50 indicates the synthetic data is highly realistic)\n\n")
  
  # 5. Feature Importance
  # Identify best model between RF and GBM based on AUC
  rf_auc <- rf_metrics |> dplyr::filter(.metric == "roc_auc") |> dplyr::pull(.estimate)
  gbm_auc <- gbm_metrics |> dplyr::filter(.metric == "roc_auc") |> dplyr::pull(.estimate)
  
  best_model_fit <- if (rf_auc > gbm_auc) {
    message("Random Forest had a better AUC. Extracting its feature importance...")
    parsnip::extract_fit_parsnip(rf_fit)
  } else {
    message("GBM had a better AUC. Extracting its feature importance...")
    parsnip::extract_fit_parsnip(gbm_fit)
  }
  
  importance_plot <- vip::vip(best_model_fit, geom = "col", aesthetics = list(fill = "steelblue")) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Feature Importance: Adversarial Classification",
      subtitle = "Features most used by the model to distinguish real vs. synthetic data",
      x = "Importance Score",
      y = "Engineered Feature"
    )
  
  print(importance_plot)
}

# # Usage Example:
# # (Assumes `combined_data` is generated using Script 1)
# # evaluate_adversaries(combined_data)
