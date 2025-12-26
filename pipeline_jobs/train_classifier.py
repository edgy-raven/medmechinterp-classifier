import functools
import json
import random

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

OUTPUT_FILES = [
    "feature_xgb_metrics.json",
    "feature_xgb.json",
    "feature_xgb_importances.csv",
    "feature_xgb_predictions.csv",
]
SEED = 7
EARLY_STOPPING_ROUNDS = 50


def inputs(context):
    feature_path = context.get("combined_features_path")
    if not feature_path:
        raise FileNotFoundError("features output missing; run with --jobs features first.")
    return {
        "feature_path": feature_path,
        "output_dir": context["model_root"],
    }


def outputs(context):
    return {
        "feature_xgb_metrics_path": context["model_root"] / OUTPUT_FILES[0],
        "feature_xgb_model_path": context["model_root"] / OUTPUT_FILES[1],
        "feature_xgb_importances_path": context["model_root"] / OUTPUT_FILES[2],
        "feature_xgb_predictions_path": context["model_root"] / OUTPUT_FILES[3],
    }


def predict_with_best_iteration(model, dmatrix):
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        return model.predict(dmatrix, iteration_range=(0, model.best_iteration + 1))
    return model.predict(dmatrix)


def compute_scale_pos_weight(y_values):
    pos = np.sum(y_values == 1)
    neg = np.sum(y_values == 0)
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def objective(trial, X_train, y_train, X_val, y_val, w_train, w_val):
    num_boost_round = trial.suggest_int("num_boost_round", 200, 800)
    max_depth = trial.suggest_int("max_depth", 2, 8)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    colsample_bynode = trial.suggest_float("colsample_bynode", 0.4, 1.0)
    min_child_weight = trial.suggest_float("min_child_weight", 1.0, 50.0)
    gamma = trial.suggest_float("gamma", 0.0, 10.0)
    max_delta_step = trial.suggest_float("max_delta_step", 0.0, 10.0)
    reg_lambda = trial.suggest_float("reg_lambda", 1.0, 50.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
    scale_pos_weight = compute_scale_pos_weight(y_train.to_numpy())

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "colsample_bynode": colsample_bynode,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "max_delta_step": max_delta_step,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "scale_pos_weight": scale_pos_weight,
        "seed": SEED,
        "nthread": 48,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val, enable_categorical=True)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    y_pred = (predict_with_best_iteration(model, dval) >= 0.5).astype(np.int32)
    return (y_pred == y_val).mean()


def tune_hyperparams(X_train, y_train, X_val, y_val, w_train, w_val):
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_fn = functools.partial(
        objective,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        w_train=w_train,
        w_val=w_val,
    )
    study.optimize(trial_fn, n_trials=25)
    return study.best_params


def run(feature_path, output_dir):
    random.seed(SEED)
    np.random.seed(SEED)
    df = pd.read_csv(feature_path, index_col="pmcid")
    
    feature_cols = [
        c
        for c in df.columns
        if c not in {"is_correct", "is_second_correct", "source", "model"}
    ]
    
    X = df[feature_cols].copy()
    y = df["is_correct"]

    train_mask = df["source"] == "train"
    val_mask = df["source"] == "val"
    weights = np.ones(len(df), dtype=float)

    best_params = tune_hyperparams(
        X[train_mask],
        y[train_mask],
        X[val_mask],
        y[val_mask],
        weights[train_mask],
        weights[val_mask],
    )
    scale_pos_weight = compute_scale_pos_weight(y[train_mask].to_numpy())
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "learning_rate": best_params["learning_rate"],
        "max_depth": best_params["max_depth"],
        "subsample": best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
        "colsample_bynode": best_params["colsample_bynode"],
        "min_child_weight": best_params["min_child_weight"],
        "gamma": best_params["gamma"],
        "max_delta_step": best_params["max_delta_step"],
        "reg_lambda": best_params["reg_lambda"],
        "reg_alpha": best_params["reg_alpha"],
        "scale_pos_weight": scale_pos_weight,
        "seed": SEED,
        "nthread": 1,
    }
    dtrain = xgb.DMatrix(
        X[train_mask],
        label=y[train_mask],
        weight=weights[train_mask],
        enable_categorical=True,
    )
    dval = xgb.DMatrix(
        X[val_mask],
        label=y[val_mask],
        weight=weights[val_mask],
        enable_categorical=True,
    )
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_params["num_boost_round"],
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    metrics = {
        "best_params": best_params,
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "feature_count": len(feature_cols),
    }
    metrics_path = output_dir / "feature_xgb_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    model_path = output_dir / "feature_xgb.json"
    model.save_model(model_path)

    scores = model.get_score(importance_type="gain")
    importances = pd.DataFrame(
        {"feature": list(scores.keys()), "importance": list(scores.values())}
    ).sort_values("importance", ascending=False)
    importances_path = output_dir / "feature_xgb_importances.csv"
    importances.to_csv(importances_path, index=False)

    available_sources = set(df["source"].dropna().unique())
    eval_sources = ["test"]
    if "nejm" in available_sources:
        eval_sources.append("nejm")
    eval_mask = df["source"].isin(eval_sources)

    dmatrix_eval = xgb.DMatrix(X[eval_mask], enable_categorical=True)
    pred_probs = predict_with_best_iteration(model, dmatrix_eval)
    pred_labels = (pred_probs >= 0.5).astype(np.int32)
    pred_df = df[eval_mask].reset_index()[["pmcid", "model"]].copy()
    pred_df["model"] = pred_df["model"].astype(str)
    pred_df["predicted_judgement"] = pred_labels
    predictions_path = output_dir / "feature_xgb_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved importances to {importances_path}")
    print(f"Saved predictions to {predictions_path}")
