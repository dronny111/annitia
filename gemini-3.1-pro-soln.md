It is totally normal to see a gap between your Cross-Validation score and the Public Leaderboard, especially with a highly imbalanced dataset (only 47 hepatic events!). When event counts are that low, the model can easily overfit to the specific patients in your training folds. 

To help close that gap, here are two highly actionable code blocks you can integrate directly into your notebook. The first implements **Optuna** for Bayesian hyperparameter tuning, and the second upgrades your imputation strategy to use **IterativeImputer** (MICE) to better capture the relationships between those longitudinal features.

### 1. Optuna Hyperparameter Tuning for XGBoost AFT

Currently, your `XGBoostAFTModel` relies on hardcoded parameters. Bayesian optimization will systematically search for parameters that generalize better across your `followup_stratified_splits`. 

You can add this cell right after your `XGBoostAFTModel` class definition. *Note: You'll need to `pip install optuna` if you haven't already.*

```python
import optuna
from sksurv.metrics import concordance_index_censored
import numpy as np

def optimize_xgb_aft(X_raw, y, n_trials=30):
    """
    Uses Optuna to find the best XGBoost AFT hyperparameters 
    via your custom follow-up stratified CV.
    """
    def objective(trial):
        # 1. Define the hyperparameter search space
        params = {
            "distribution": trial.suggest_categorical("distribution", ["normal", "logistic", "extreme_value"]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "num_boost_round": trial.suggest_int("num_boost_round", 100, 400)
        }

        # 2. Get splits and evaluate
        splits = followup_stratified_splits(X_raw, y, n_splits=N_SPLITS)
        scores = []

        for tr_idx, te_idx in splits:
            X_tr, X_te, _, _ = preprocess_features(X_raw.iloc[tr_idx], X_val_raw=X_raw.iloc[te_idx])
            
            # Initialize and fit model with trial params
            model = XGBoostAFTModel(**params)
            model.fit(X_tr, y[tr_idx])
            preds = model.predict(X_te)
            
            # Evaluate C-index
            c_index = concordance_index_censored(y[te_idx]["event"], y[te_idx]["time"], preds)[0]
            scores.append(c_index)
            
        return np.mean(scores)

    # 3. Run the study
    study = optuna.create_study(direction="maximize")
    
    # Suppress excessive logging
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    
    print("Starting Optuna optimization for XGBoost AFT...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Best CV C-index: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")
    
    return study.best_params

# Example Usage (Run this before your final ensemble CV):
# best_xgb_hep_params = optimize_xgb_aft(X_hep, y_hep, n_trials=30)
# best_xgb_death_params = optimize_xgb_aft(X_death, y_death, n_trials=30)
```

---

### 2. Advanced Imputation (IterativeImputer)

Your current pipeline uses `SimpleImputer(strategy="median")`. While safe, replacing missing FibroScan or Aixplorer slopes with a population median destroys the variance that tree models rely on. 

Using `IterativeImputer` (with a lightweight `ExtraTreesRegressor`) models each feature as a function of the others. This is incredibly powerful for biochemical data, as missing `AST` can often be accurately inferred if `ALT` and `GGT` are present.

You can replace your existing `preprocess_features` function with this updated version:

```python
from sklearn.experimental import enable_iterative_imputer  # Required to use IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

def preprocess_features_advanced(
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame | None = None,
    X_test_raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    """
    Impute missing values using multivariate imputation (IterativeImputer) 
    and filter high-missingness columns. Fit ONLY on training data.
    """
    missing_rate = X_train_raw.isna().mean()
    keep_cols = missing_rate[missing_rate <= MISSINGNESS_THRESHOLD].index.tolist()
    
    if not keep_cols:
        raise ValueError("No features passed the missingness filter.")

    # Use ExtraTrees for fast, non-linear imputation
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=RANDOM_STATE),
        max_iter=5,
        random_state=RANDOM_STATE,
        add_indicator=False # Set to True if you want missingness indicator columns
    )
    
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train_raw[keep_cols]),
        columns=keep_cols,
        index=X_train_raw.index,
    )

    def _apply(df):
        if df is None:
            return None
        return pd.DataFrame(
            imputer.transform(df.reindex(columns=keep_cols, fill_value=np.nan)),
            columns=keep_cols,
            index=df.index,
        )

    return X_train, _apply(X_val_raw), _apply(X_test_raw), keep_cols
```

### Next Steps to Implement
1. Swap `preprocess_features` with `preprocess_features_advanced` in Section 3 of your notebook.
2. Add the Optuna block right before Section 5.
3. Update your `cross_validate_full` function to accept the `best_params` dictionary and pass it to your models. 

This should help your models learn more robust, generalizable patterns rather than memorizing the limited training events!