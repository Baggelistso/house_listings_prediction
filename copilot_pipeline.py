import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

RANDOM_SEED = 42
TARGET_COLUMN = "price_month"
TARGET_NEIGHBORHOOD = "Marousi"
NUMERIC_FEATURES = ["rooms", "sq_m", "bathrooms", "floor", "year_built"]
CATEGORICAL_FEATURES = ["area"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "relu",
    "solver": "adam",
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": RANDOM_SEED,
}

AREA_LOCATION = {
    "Kolonaki": {
        "town_municipality": "Δήμος Αθηναίων",
        "municipality_en": "Municipality of Athens",
        "city": "Αθήνα (Athens)",
        "regional_unit": "Κεντρικός Τομέας Αθηνών",
    },
    "Koukaki": {
        "town_municipality": "Δήμος Αθηναίων",
        "municipality_en": "Municipality of Athens",
        "city": "Αθήνα (Athens)",
        "regional_unit": "Κεντρικός Τομέας Αθηνών",
    },
    "Exarchia": {
        "town_municipality": "Δήμος Αθηναίων",
        "municipality_en": "Municipality of Athens",
        "city": "Αθήνα (Athens)",
        "regional_unit": "Κεντρικός Τομέας Αθηνών",
    },
    "Kypseli": {
        "town_municipality": "Δήμος Αθηναίων",
        "municipality_en": "Municipality of Athens",
        "city": "Αθήνα (Athens)",
        "regional_unit": "Κεντρικός Τομέας Αθηνών",
    },
    "Nea Smyrni": {
        "town_municipality": "Δήμος Νέας Σμύρνης",
        "municipality_en": "Municipality of Nea Smyrni",
        "city": "Αθήνα (Athens)",
        "regional_unit": "Νότιος Τομέας Αθηνών",
    },
    "Marousi": {
        "town_municipality": "Δήμος Αμαρουσίου",
        "municipality_en": "Municipality of Marousi",
        "city": "Αθήνα (Athens)",
        "regional_unit": "Βόρειος Τομέας Αθηνών",
    },
    "Glyfada": {
        "town_municipality": "Δήμος Γλυφάδας",
        "municipality_en": "Municipality of Glyfada",
        "city": "Αθήνα (Athens)",
        "regional_unit": "Νότιος Τομέας Αθηνών",
    },
    "Piraeus": {
        "town_municipality": "Δήμος Πειραιώς",
        "municipality_en": "Municipality of Piraeus",
        "city": "Αθήνα (Athens)",
        "regional_unit": "Περιφερειακή Ενότητα Πειραιώς",
    },
}

REGION = "Περιφέρεια Αττικής"
COUNTRY = "Ελλάδα (Greece)"
MODEL_LABEL = "GitHub Copilot"

OUTPUT_DIR = Path("results_copilot")
PRED_CSV_PATH = OUTPUT_DIR / "preds.csv"
PRED_JSON_PATH = OUTPUT_DIR / "preds.json"
RESULTS_JSON_PATH = OUTPUT_DIR / "results.json"
RUN_INFO_JSON_PATH = OUTPUT_DIR / "run_info.json"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    eps = 1e-12
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / np.clip(np.abs(y_true_arr), eps, None))) * 100.0)
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    return {"rmse": rmse, "mae": mae, "mape_pct": mape, "r2": r2}


def resolve_data_path() -> Path:
    candidates = [
        Path("monthly_rentals_greece.json"),
        Path("data/monthly_rentals_greece.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find monthly_rentals_greece.json in project root or data/"
    )


def apply_location_mapping(df: pd.DataFrame) -> pd.DataFrame:
    unknown = sorted(set(df["area"]) - set(AREA_LOCATION))
    if unknown:
        raise ValueError(
            "Missing location mapping for neighborhoods: " + ", ".join(unknown)
        )

    mapped = df["area"].map(AREA_LOCATION)
    out = df.copy()
    out["neighborhood"] = out["area"]
    out["town_municipality"] = mapped.str["town_municipality"]
    out["municipality_en"] = mapped.str["municipality_en"]
    out["city"] = mapped.str["city"]
    out["regional_unit"] = mapped.str["regional_unit"]
    out["region"] = REGION
    out["country"] = COUNTRY
    return out


def print_metrics(title: str, m: dict) -> None:
    print(title)
    print(f"  RMSE: {m['rmse']:.4f}")
    print(f"  MAE:  {m['mae']:.4f}")
    print(f"  MAPE: {m['mape_pct']:.4f}%")
    print(f"  R²:   {m['r2']:.6f}")


def format_currency(x: float) -> str:
    return f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data_path = resolve_data_path()
    df = pd.read_json(data_path)

    print("=== GitHub Copilot MLP Rental Pipeline ===")
    print(f"Data path: {data_path.resolve()}")
    print(f"Dataset size: {len(df)}")
    print(f"Target: {TARGET_COLUMN} (log1p/expm1 via TransformedTargetRegressor)")
    print(f"Numeric features: {NUMERIC_FEATURES}")
    print(f"Categorical features: {CATEGORICAL_FEATURES}")
    print(f"MLP parameters: {MLP_PARAMS}")
    print(f"Training size: {len(df)} (100% of data, in-sample evaluation)")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("mlp", MLPRegressor(**MLP_PARAMS)),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    model.fit(X, y)
    y_pred = model.predict(X)

    mlp_fitted = model.regressor_.named_steps["mlp"]
    print(f"Epochs actually run: {mlp_fitted.n_iter_}")

    overall_metrics = calculate_metrics(y, y_pred)

    enriched = df.copy()
    enriched["actual_price"] = y
    enriched["predicted_price"] = y_pred
    enriched["diff"] = enriched["predicted_price"] - enriched["actual_price"]
    enriched = apply_location_mapping(enriched)

    output_df = enriched.loc[enriched["neighborhood"] == TARGET_NEIGHBORHOOD].copy()
    output_df = output_df.sort_values("id").reset_index(drop=True)

    if output_df.empty:
        raise ValueError(f"No rows found for output neighborhood: {TARGET_NEIGHBORHOOD}")

    neighborhood_metrics = calculate_metrics(
        output_df["actual_price"].to_numpy(dtype=float),
        output_df["predicted_price"].to_numpy(dtype=float),
    )

    print_metrics("\nIn-sample metrics (all 3,000 listings):", overall_metrics)
    print_metrics(
        f"\nIn-sample metrics ({TARGET_NEIGHBORHOOD} only):", neighborhood_metrics
    )

    sample_n = min(15, len(output_df))
    sample_df = output_df.sample(n=sample_n, random_state=RANDOM_SEED).copy()
    sample_df = sample_df[
        [
            "neighborhood",
            "town_municipality",
            "city",
            "sq_m",
            "rooms",
            "bathrooms",
            "floor",
            "year_built",
            "actual_price",
            "predicted_price",
            "diff",
        ]
    ]

    sample_display = sample_df.rename(
        columns={
            "town_municipality": "municipality",
            "sq_m": "m²",
            "year_built": "year built",
            "actual_price": "actual €",
            "predicted_price": "predicted €",
            "diff": "difference (predicted − actual)",
        }
    )

    for c in ["actual €", "predicted €", "difference (predicted − actual)"]:
        sample_display[c] = sample_display[c].map(format_currency)

    print(f"\nRandom sample ({sample_n} rows) from {TARGET_NEIGHBORHOOD}:")
    print(sample_display.to_string(index=False))

    summary = (
        output_df.groupby("town_municipality", dropna=False)
        .agg(
            n=("id", "size"),
            mean_actual=("actual_price", "mean"),
            mean_predicted=("predicted_price", "mean"),
        )
        .reset_index()
    )
    summary["difference"] = summary["mean_predicted"] - summary["mean_actual"]
    summary["difference_pct"] = (
        np.where(
            summary["mean_actual"] != 0,
            summary["difference"] / summary["mean_actual"] * 100.0,
            np.nan,
        )
    )

    summary_print = summary.copy()
    for col in ["mean_actual", "mean_predicted", "difference"]:
        summary_print[col] = summary_print[col].map(format_currency)
    summary_print["difference_pct"] = summary_print["difference_pct"].map(
        lambda x: f"{x:.4f}%"
    )

    print("\nMean actual vs mean predicted rent per municipality:")
    print(summary_print.to_string(index=False))

    output_cols = [
        "id",
        "neighborhood",
        "town_municipality",
        "municipality_en",
        "city",
        "regional_unit",
        "region",
        "country",
        "rooms",
        "sq_m",
        "bathrooms",
        "floor",
        "year_built",
        "actual_price",
        "predicted_price",
        "diff",
    ]

    output_rows = output_df[output_cols].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_rows.to_csv(PRED_CSV_PATH, index=False, encoding="utf-8")
    output_rows.to_json(PRED_JSON_PATH, orient="records", force_ascii=False, indent=2)

    results_payload = {
        "model_label": MODEL_LABEL,
        "target": TARGET_COLUMN,
        "train_size": int(len(df)),
        "output_neighborhood": TARGET_NEIGHBORHOOD,
        "output_rows": int(len(output_rows)),
        "overall_in_sample_metrics": overall_metrics,
        "output_neighborhood_in_sample_metrics": neighborhood_metrics,
    }

    run_info_payload = {
        "data_path": str(data_path.resolve()),
        "model_label": MODEL_LABEL,
        "seed": RANDOM_SEED,
        "target": TARGET_COLUMN,
        "target_transform": {
            "func": "log1p",
            "inverse_func": "expm1",
            "wrapper": "TransformedTargetRegressor",
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "preprocessing": {
            "numeric": "StandardScaler",
            "categorical": "OneHotEncoder(handle_unknown='ignore')",
        },
        "mlp_parameters": MLP_PARAMS,
        "training": {
            "train_size": int(len(df)),
            "split": "none (100% train, in-sample metrics only)",
            "epochs_run": int(mlp_fitted.n_iter_),
        },
        "output_scope": {
            "target_neighborhood": TARGET_NEIGHBORHOOD,
            "output_rows": int(len(output_rows)),
        },
        "metrics": {
            "overall_in_sample": overall_metrics,
            "output_neighborhood_in_sample": neighborhood_metrics,
        },
    }

    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    with RUN_INFO_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(run_info_payload, f, ensure_ascii=False, indent=2)

    print("\nWritten files:")
    for p in [PRED_CSV_PATH, PRED_JSON_PATH, RESULTS_JSON_PATH, RUN_INFO_JSON_PATH]:
        print(f"- {p.resolve()}")


if __name__ == "__main__":
    main()
