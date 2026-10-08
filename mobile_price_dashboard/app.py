from flask import Flask, render_template, request, jsonify
from pathlib import Path
import json
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
DATA_DIR = BASE / "data"
TRAIN_SCRIPT = BASE / "train_model.py"

app = Flask(__name__)

FINAL_MODEL_PATH = MODEL_DIR / "mobile_price_final_model.pkl"
ENCODER_PATH = MODEL_DIR / "mobile_price_encoder.pkl"
SCALER_PATH = MODEL_DIR / "mobile_price_scaler.pkl"
META_PATH = MODEL_DIR / "metadata.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

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

NUMERIC_INPUTS = [
    "clock_speed_ghz", "storage_gb", "ram_gb", "rating_score",
    "res_width_px", "res_height_px", "display_inches", "battery_mah",
    "charging_watt", "rear_camera_count", "front_camera_count"
]

BOOL_INPUTS = [
    "has_5g", "has_nfc", "has_ir_blaster", "fast_charging",
    "memory_card_supported"
]

CATEGORICAL_INPUTS = [
    "smartphone_brand", "processor_brand", "processor_name",
    "os_name", "memory_card_type"
]

model = None
encoder = None
scaler = None
metadata = None
feature_schema = None
df = None


def artifacts_exist():
    return all(p.exists() for p in [
        FINAL_MODEL_PATH, ENCODER_PATH, SCALER_PATH, META_PATH, SCHEMA_PATH
    ])


def ensure_artifacts():
    """Train automatically the first time if model artifacts are absent."""
    if artifacts_exist():
        return

    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    print("Model files not found. Running train_model.py automatically...")
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=str(BASE),
        capture_output=True,
        text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            "Model files are missing and automatic training failed. "
            "Run 'python train_model.py' in the project folder and read the "
            "terminal error. If the dataset cannot be downloaded, place "
            "smartphones.csv inside the data folder and run it again."
        )


def load_artifacts():
    global model, encoder, scaler, metadata, feature_schema
    ensure_artifacts()
    model = joblib.load(FINAL_MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    feature_schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def add_engineered_features(data):
    data = data.copy()
    data["total_camera_count"] = (
        data["front_camera_count"] + data["rear_camera_count"]
    )
    data["screen_pixels"] = (
        data["res_width_px"] * data["res_height_px"]
    )
    data["ppi"] = (
        np.sqrt(data["res_width_px"] ** 2 + data["res_height_px"] ** 2)
        / data["display_inches"]
    )
    data["charging_efficiency"] = (
        data["charging_watt"] / (data["battery_mah"] + 1)
    )
    data["premium_features"] = (
        data["has_5g"].astype(int)
        + data["has_nfc"].astype(int)
        + data["has_ir_blaster"].astype(int)
        + data["fast_charging"].astype(int)
    )
    return data


def load_dataset():
    global df
    csv_path = DATA_DIR / "smartphones.csv"
    if not csv_path.exists():
        return
    raw = pd.read_csv(csv_path)
    raw = raw.drop_duplicates()
    raw = raw.dropna(subset=["price_inr"])
    num_cols = raw.select_dtypes(include="number").columns
    cat_cols = raw.select_dtypes(exclude="number").columns
    raw[num_cols] = raw[num_cols].fillna(raw[num_cols].median())
    for col in cat_cols:
        if not raw[col].mode().empty:
            raw[col] = raw[col].fillna(raw[col].mode()[0])
    raw = raw[raw["price_inr"] > 0].reset_index(drop=True)
    df = add_engineered_features(raw)


def make_input(payload):
    """Build exactly the 26-feature input used by the notebook."""
    row = {}
    for c in NUMERIC_INPUTS:
        row[c] = float(payload[c])
    for c in BOOL_INPUTS:
        # In the notebook these columns are categorical booleans.
        row[c] = bool(int(payload[c]))
    for c in CATEGORICAL_INPUTS:
        row[c] = str(payload[c])

    x = pd.DataFrame([row])
    x = add_engineered_features(x)
    return x[FEATURES]


def prepare_prediction_input(payload):
    """Apply the exact encoder/scaler order used in Mobile_Price_Final_Model.ipynb."""
    if feature_schema is None:
        load_artifacts()

    x = make_input(payload)
    num_cols = feature_schema["numerical_features"]
    cat_cols = feature_schema["categorical_features"]

    # These lists are saved by the notebook in feature_schema.json.
    # Using them here avoids relying on metadata.json for preprocessing.
    x_num = scaler.transform(x[num_cols])
    x_cat = encoder.transform(x[cat_cols])
    return np.hstack([x_num, x_cat])


def segment(price):
    if price < 15000:
        return "Budget"
    if price < 30000:
        return "Mid-Range"
    if price < 60000:
        return "Upper-Mid Range"
    return "Premium"


@app.route("/")
def dashboard():
    return render_template("index.html", page="dashboard")


@app.route("/prediction")
def prediction():
    if metadata is None:
        load_artifacts()
    categories = metadata.get("categories", {})
    return render_template(
        "prediction.html",
        page="prediction",
        categories=categories,
        rating_min=metadata.get("rating_min", 0),
        rating_max=metadata.get("rating_max", 100),
    )


@app.route("/analytics")
def analytics():
    return render_template("analytics.html", page="analytics")


@app.route("/model")
def model_page():
    return render_template("model.html", page="model")


@app.route("/features")
def features_page():
    return render_template("features.html", page="features")


@app.route("/dataset")
def dataset_page():
    return render_template("dataset.html", page="dataset")


@app.get("/api/metrics")
def api_metrics():
    if metadata is None:
        load_artifacts()
    return jsonify(metadata)


@app.get("/api/feature-importance")
def api_feature_importance():
    if metadata is None:
        load_artifacts()
    return jsonify(metadata["feature_importance"])


@app.get("/api/analytics")
def api_analytics():
    if df is None:
        load_dataset()
    if df is None:
        return jsonify({
            "error": "Dataset not found. Run train_model.py first."
        }), 404

    def records(series):
        return [
            {"name": str(k), "value": float(v)}
            for k, v in series.items()
        ]

    top_brands = (
        df.groupby("smartphone_brand")["price_inr"]
        .mean().sort_values(ascending=False).head(15)
    )
    brand_count = df["smartphone_brand"].value_counts().head(15)

    return jsonify({
        "price_distribution": df["price_inr"].round(0).tolist(),
        "brand_average": records(top_brands),
        "brand_count": records(brand_count),
        "ram_price": [
            {"x": float(x), "y": float(y)}
            for x, y in zip(df["ram_gb"], df["price_inr"])
        ],
        "storage_price": [
            {"x": float(x), "y": float(y)}
            for x, y in zip(df["storage_gb"], df["price_inr"])
        ],
        "rating_price": [
            {"x": float(x), "y": float(y)}
            for x, y in zip(df["rating_score"], df["price_inr"])
        ],
        "battery_price": [
            {"x": float(x), "y": float(y)}
            for x, y in zip(df["battery_mah"], df["price_inr"])
        ],
        "charging_price": [
            {"x": float(x), "y": float(y)}
            for x, y in zip(df["charging_watt"], df["price_inr"])
        ],
        "stats": {
            "rows": int(len(df)),
            "columns": 27,
            "min_price": float(df["price_inr"].min()),
            "max_price": float(df["price_inr"].max()),
            "avg_price": float(df["price_inr"].mean()),
        },
    })


@app.post("/api/predict")
def api_predict():
    try:
        if model is None:
            load_artifacts()
        payload = request.get_json(force=True)
        x_processed = prepare_prediction_input(payload)
        price = float(model.predict(x_processed)[0])
        return jsonify({
            "price": round(price, 2),
            "segment": segment(price),
            "model": metadata.get("final_model", "Gradient Boosting"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Prepare the final model before serving so the first dashboard request does not
# produce a Missing metadata/model file error.
try:
    load_artifacts()
    load_dataset()
except Exception as startup_error:
    print("Startup model preparation message:", startup_error)


if __name__ == "__main__":
    # Debug reloader is intentionally disabled so training cannot run twice.
    app.run(host="127.0.0.1", port=5000, debug=False)
