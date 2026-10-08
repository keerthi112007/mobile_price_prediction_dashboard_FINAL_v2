from pathlib import Path
import json
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
DATA_DIR = BASE / "data"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/samyuthakanakamedala/mobile-price-prediction/refs/heads/main/smartphones.csv"
CSV_PATH = DATA_DIR / "smartphones.csv"

FEATURES = [
    "clock_speed_ghz", "storage_gb", "ram_gb", "rating_score",
    "res_width_px", "res_height_px", "display_inches", "battery_mah",
    "charging_watt", "rear_camera_count", "front_camera_count",
    "smartphone_brand", "processor_brand", "processor_name", "os_name",
    "has_5g", "has_nfc", "has_ir_blaster", "fast_charging",
    "memory_card_supported", "memory_card_type",
    "total_camera_count", "screen_pixels", "ppi",
    "charging_efficiency", "premium_features"
]


def get_dataset():
    if not CSV_PATH.exists():
        print("Dataset not found locally. Downloading smartphones.csv...")
        try:
            urllib.request.urlretrieve(DATA_URL, CSV_PATH)
        except Exception as exc:
            raise RuntimeError(
                "Could not download smartphones.csv. Put the dataset file at "
                f"{CSV_PATH} and run this script again.\nOriginal error: {exc}"
            ) from exc
    return pd.read_csv(CSV_PATH)


def clean_dataset(df):
    # Exact cleaning from Mobile_Price_Final_Model.ipynb
    df = df.drop_duplicates()
    df = df.dropna(subset=["price_inr"])

    num_cols_all = df.select_dtypes(include="number").columns
    cat_cols_all = df.select_dtypes(exclude="number").columns

    df[num_cols_all] = df[num_cols_all].fillna(df[num_cols_all].median())
    for col in cat_cols_all:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df[df["price_inr"] > 0]
    return df.reset_index(drop=True)


def add_engineered_features(df):
    # Exact feature engineering from the notebook.
    df = df.copy()
    df["total_camera_count"] = (
        df["front_camera_count"] + df["rear_camera_count"]
    )
    df["screen_pixels"] = (
        df["res_width_px"] * df["res_height_px"]
    )
    df["ppi"] = (
        np.sqrt(df["res_width_px"] ** 2 + df["res_height_px"] ** 2)
        / df["display_inches"]
    )
    df["charging_efficiency"] = (
        df["charging_watt"] / (df["battery_mah"] + 1)
    )
    df["premium_features"] = (
        df["has_5g"].astype(int)
        + df["has_nfc"].astype(int)
        + df["has_ir_blaster"].astype(int)
        + df["fast_charging"].astype(int)
    )
    return df


def cap_outliers(X_train, X_test, num_cols):
    # Exact train-based IQR capping from the notebook.
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in num_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X_train[col] = X_train[col].astype(float).clip(lower, upper)
        X_test[col] = X_test[col].astype(float).clip(lower, upper)
    return X_train, X_test


def main():
    df = clean_dataset(get_dataset())
    df = add_engineered_features(df)

    X = df[FEATURES]
    y = df["price_inr"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test = cap_outliers(X_train, X_test, num_cols)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])

    # Exact model configurations from the notebook.
    rf_model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    gb_model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1
    )

    models = {
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "XGBoost": xgb_model
    }

    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_processed, y_train)
        predictions = model.predict(X_test_processed)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        results.append({
            "Model": name,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2)
        })

    results_df = pd.DataFrame(results)

    # Exact final-model selection from the notebook.
    final_model = gb_model
    final_model_name = "Gradient Boosting"

    # Save the same model/preprocessing artifacts as the notebook.
    joblib.dump(rf_model, MODEL_DIR / "random_forest_model.pkl")
    joblib.dump(gb_model, MODEL_DIR / "gradient_boosting_model.pkl")
    joblib.dump(xgb_model, MODEL_DIR / "xgboost_model.pkl")
    joblib.dump(final_model, MODEL_DIR / "mobile_price_final_model.pkl")
    joblib.dump(encoder, MODEL_DIR / "mobile_price_encoder.pkl")
    joblib.dump(scaler, MODEL_DIR / "mobile_price_scaler.pkl")

    metrics_data = {}
    for _, row in results_df.iterrows():
        metrics_data[row["Model"]] = {
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "R2": round(float(row["R2"]), 2)
        }
    metrics_data["Final Model"] = final_model_name
    (MODEL_DIR / "model_metrics.json").write_text(
        json.dumps(metrics_data, indent=4), encoding="utf-8"
    )

    feature_schema = {
        "features": FEATURES,
        "numerical_features": num_cols,
        "categorical_features": cat_cols
    }
    (MODEL_DIR / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=4), encoding="utf-8"
    )

    # Gradient Boosting feature importance is based on the processed features.
    feature_names = list(num_cols) + list(encoder.get_feature_names_out(cat_cols))
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": final_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    categories = {}
    for col in [
        "smartphone_brand", "processor_brand", "processor_name",
        "os_name", "memory_card_type"
    ]:
        categories[col] = sorted(
            [str(v) for v in df[col].dropna().unique().tolist()]
        )

    metadata = {
        "dataset_rows": int(len(df)),
        "dataset_columns": 27,
        "features": len(FEATURES),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "processed_features": int(X_train_processed.shape[1]),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "final_model": final_model_name,
        "metrics": metrics_data,
        "feature_importance": [
            {"Feature": str(r["Feature"]), "Importance": float(r["Importance"])}
            for _, r in importance.iterrows()
        ],
        "categories": categories,
        "boolean_values": ["0", "1"],
        "rating_min": float(df["rating_score"].min()),
        "rating_max": float(df["rating_score"].max()),
        "price_min": float(df["price_inr"].min()),
        "price_max": float(df["price_inr"].max()),
        "price_average": float(df["price_inr"].mean())
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=4), encoding="utf-8"
    )

    print("\nTraining complete.")
    print(results_df.round(2).to_string(index=False))
    final_result = results_df[results_df["Model"] == final_model_name].iloc[0]
    print("\nFinal Model:", final_model_name)
    print("MAE :", round(final_result["MAE"], 2))
    print("RMSE:", round(final_result["RMSE"], 2))
    print("R²  :", round(final_result["R2"], 2))
    print("\nArtifacts saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
