import os, json, ast, tempfile
import numpy as np
import pandas as pd
import joblib

from google.cloud import storage

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Helpers GCS
# -----------------------------
def gcs_download_text(bucket_name, blob_path) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_text()

def gcs_download_to_file(bucket_name, blob_path, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)

def gcs_upload_file(bucket_name, local_path, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

# -----------------------------
# Feature engineering
# -----------------------------
def compute_amenities_count(x):
    if pd.isna(x) or not isinstance(x, str):
        return 0
    s = x.strip()
    if s == "" or s.lower() == "nan":
        return 0
    try:
        parsed = ast.literal_eval(s)   # soporta '["Wifi","TV"]'
        if isinstance(parsed, list):
            return int(len(parsed))
        return 0
    except Exception:
        # fallback simple
        return int(len([a for a in s.split(",") if a.strip()]))

def main():
    # Se recomienda pasar la ruta de config por env var (o dejar default)
    CONFIG_GCS_PATH = os.environ.get("CONFIG_GCS_PATH", "airbnb-project/artifacts/config.json")

    # Bucket también puede venir del config. Para leer el config, necesitamos bucket por env o hardcode.
    # Lo más simple: BUCKET por env.
    BUCKET = os.environ.get("BUCKET_NAME", "0282789_bucket")

    # 1) Leer config.json
    cfg_text = gcs_download_text(BUCKET, CONFIG_GCS_PATH)
    cfg = json.loads(cfg_text)

    listings_path = cfg["data"]["listings"]  # e.g. airbnb-project/data/listings.csv
    model_out_path = cfg["model"]["path"]    # e.g. airbnb-project/models/airbnb_price_pipe.joblib

    # 2) Descargar listings.csv a disco temporal
    with tempfile.TemporaryDirectory() as tmp:
        local_csv = os.path.join(tmp, "listings.csv")
        gcs_download_to_file(BUCKET, listings_path, local_csv)

        df = pd.read_csv(local_csv, encoding="latin1", low_memory=False)

        # 3) Crear amenities_count
        if "amenities" in df.columns:
            df["amenities_count"] = df["amenities"].apply(compute_amenities_count).astype(int)
        else:
            df["amenities_count"] = 0

        # 4) Limpiar price -> price_num
        price_str = (
            df["price"]
            .astype("string")
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
        )
        df["price_num"] = pd.to_numeric(price_str, errors="coerce")
        df = df[df["price_num"].notna()].copy()
        df = df[df["price_num"] > 0].copy()

        # 5) p99 en price
        p99_price = df["price_num"].quantile(0.99)
        df = df[df["price_num"] <= p99_price].copy()

        # 6) log target
        df["y_log"] = np.log1p(df["price_num"])

        # 7) filtrar minimum_nights extremos
        df["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce")
        df = df[(df["minimum_nights"].isna()) | (df["minimum_nights"] <= 14)].copy()

        # 8) Features
        features = [
            "neighbourhood_cleansed",
            "room_type",
            "accommodates",
            "bathrooms",
            "bedrooms",
            "beds",
            "minimum_nights",
            "latitude",
            "longitude",
            "amenities_count",
        ]

        missing = [c for c in features + ["y_log"] if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")

        X = df[features].copy()
        y = df["y_log"].copy()

        # Tipos numéricos
        for c in [
            "accommodates","bathrooms","bedrooms","beds","minimum_nights","latitude","longitude","amenities_count"
        ]:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        # 9) Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 10) Preprocess + Modelo
        cat_cols = ["neighbourhood_cleansed", "room_type"]
        num_cols = [c for c in features if c not in cat_cols]

        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ]
        )

        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=400,
            random_state=42
        )

        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", model),
        ])

        # 11) Entrenar + métricas
        pipe.fit(X_train, y_train)
        pred_log = pipe.predict(X_test)

        mae_log = mean_absolute_error(y_test, pred_log)
        rmse_log = np.sqrt(mean_squared_error(y_test, pred_log))
        r2 = r2_score(y_test, pred_log)

        pred_price = np.expm1(pred_log)
        real_price = np.expm1(y_test)

        mae_mxn = mean_absolute_error(real_price, pred_price)
        rmse_mxn = np.sqrt(mean_squared_error(real_price, pred_price))

        metrics = {
            "rows_after_cleaning": int(len(df)),
            "p99_price": float(p99_price),
            "mae_log": float(mae_log),
            "rmse_log": float(rmse_log),
            "r2_log": float(r2),
            "mae_mxn": float(mae_mxn),
            "rmse_mxn": float(rmse_mxn),
            "features": features,
            "target": "log1p(price_num)",
        }

        # 12) Guardar a archivos locales
        local_model = os.path.join(tmp, "airbnb_price_pipe.joblib")
        local_metrics = os.path.join(tmp, "metrics.json")
        joblib.dump(pipe, local_model)

        with open(local_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # 13) Subir a GCS
        gcs_upload_file(BUCKET, local_model, model_out_path)

        # metrics: mejor guardarlas en artifacts/metrics.json (según tu bucket actual)
        metrics_path = "airbnb-project/artifacts/metrics.json"
        gcs_upload_file(BUCKET, local_metrics, metrics_path)

        print("✅ Entrenamiento terminado")
        print("Modelo:", f"gs://{BUCKET}/{model_out_path}")
        print("Métricas:", f"gs://{BUCKET}/{metrics_path}")
        print("R2(log):", r2, "MAE(MXN):", mae_mxn, "RMSE(MXN):", rmse_mxn)

if __name__ == "__main__":
    main()
