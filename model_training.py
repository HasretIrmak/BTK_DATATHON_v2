import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train_lgbm(X_train, y_train, groups):
    """LightGBM modelini Optuna ile eğitir ve tahminleri döndürür."""
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    def objective_lgb(trial):
        lgb_params = {
            'objective': 'regression_l1', 'metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'verbose': -1, 'n_jobs': -1, 'seed': 42
        }
        mse_scores = []
        for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            if len(X_va) > 0:
                model = lgb.LGBMRegressor(**lgb_params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
                y_pred = model.predict(X_va)
                mse_scores.append(mean_squared_error(np.expm1(y_va), np.expm1(y_pred)))
        return np.mean(mse_scores)

    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(objective_lgb, n_trials=20)
    lgb_params = study_lgb.best_params
    lgb_params.update({'objective': 'regression_l1', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1, 'seed': 42})
    print("LightGBM için en iyi parametreler:", lgb_params)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    models = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        if len(X_va) > 0:
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_preds[va_idx] = model.predict(X_va)
            test_preds += model.predict(X_test) / n_splits
            models.append(model)

    print("LGBM eğitimi tamamlandı.")
    return oof_preds, test_preds, models


def train_xgb(X_train, y_train, groups):
    """XGBoost modelini Optuna ile eğitir ve tahminleri döndürür."""
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    def objective_xgb(trial):
        xgb_params = {
            'objective': 'reg:squarederror', 'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12), 'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 100
        }
        mse_scores = []
        for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            if len(X_va) > 0:
                model = xgb.XGBRegressor(**xgb_params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                y_pred = model.predict(X_va)
                mse_scores.append(mean_squared_error(np.expm1(y_va), np.expm1(y_pred)))
        return np.mean(mse_scores)

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=20)
    xgb_params = study_xgb.best_params
    xgb_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 100})
    print("XGBoost için en iyi parametreler:", xgb_params)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    models = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        if len(X_va) > 0:
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_preds[va_idx] = model.predict(X_va)
            test_preds += model.predict(X_test) / n_splits
            models.append(model)

    print("XGBoost eğitimi tamamlandı.")
    return oof_preds, test_preds, models


def train_cbt(X_train, y_train, groups):
    """CatBoost modelini Optuna ile eğitir ve tahminleri döndürür."""
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    def objective_cbt(trial):
        cbt_params = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10), 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0, 10.0),
            'loss_function': 'RMSE', 'verbose': 0, 'random_seed': 42
        }
        mse_scores = []
        for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            if len(X_va) > 0:
                model = cbt.CatBoostRegressor(**cbt_params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100)
                y_pred = model.predict(X_va)
                mse_scores.append(mean_squared_error(np.expm1(y_va), np.expm1(y_pred)))
        return np.mean(mse_scores)

    study_cbt = optuna.create_study(direction='minimize')
    study_cbt.optimize(objective_cbt, n_trials=20)
    cbt_params = study_cbt.best_params
    cbt_params.update({'loss_function': 'RMSE', 'verbose': 0, 'random_seed': 42})
    print("CatBoost için en iyi parametreler:", cbt_params)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    models = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        if len(X_va) > 0:
            model = cbt.CatBoostRegressor(**cbt_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100)
            oof_preds[va_idx] = model.predict(X_va)
            test_preds += model.predict(X_test) / n_splits
            models.append(model)

    print("CatBoost eğitimi tamamlandı.")
    return oof_preds, test_preds, models


def train_meta_model(oof_preds, y_train):
    """Meta-modeli eğitir ve nihai tahminleri döndürür."""
    meta_model = Ridge(alpha=1.0, random_state=42)
    meta_model.fit(oof_preds, y_train)
    return meta_model


def plot_feature_importance(models, feature_names, title):
    """Oluşturulan özelliklerin önemini görselleştirir."""
    feature_importances = pd.DataFrame()
    for i, model in enumerate(models):
        if hasattr(model, 'feature_importances_'):
            fold_importance = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
            feature_importances = pd.concat([feature_importances, fold_importance], ignore_index=True)

    if feature_importances.empty:
        print("Özellik önem bilgisi bulunamadı.")
        return

    avg_importance = feature_importances.groupby('feature')['importance'].mean().reset_index()
    avg_importance = avg_importance.sort_values(by='importance', ascending=False).head(20)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=avg_importance)
    plt.title(f'Top 20 {title} Feature Importance')
    plt.xlabel('Önem Derecesi')
    plt.ylabel('Özellik')
    plt.tight_layout()
    plt.show()