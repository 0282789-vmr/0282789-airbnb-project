import json
import os
import tempfile
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from google.cloud import storage


# =========================
# Config (ENV)
# =========================
GCS_BUCKET = os.getenv("GCS_BUCKET", "0282789_bucket")
# OJO: ajusta esto a donde realmente quedó tu config.json (según tu screenshot está en models/)
GCS_CONFIG_PATH = os.getenv("GCS_CONFIG_PATH", "airbnb-project/models/config.json")

app = FastAPI(title="Airbnb Price + Payback API")


# =========================
# Helpers GCS
# =========================
def gcs_read_text(bucket_name: str, blob_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_text()

def gcs_download_file(bucket_name: str, blob_path: str, local_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


# =========================
# Cargar config + modelo al iniciar
# =========================
CONFIG = None
PIPE = None
PURCHASE_DF = None

def load_everything():
    global CONFIG, PIPE, PURCHASE_DF

    # 1) config.json
    config_text = gcs_read_text(GCS_BUCKET, GCS_CONFIG_PATH)
    CONFIG = json.loads(config_text)

    # 2) modelo
    model_path = CONFIG["model"]["path"]  # ruta dentro del bucket
    with tempfile.TemporaryDirectory() as td:
        local_model = os.path.join(td, "model.joblib")
        gcs_download_file(GCS_BUCKET, model_path, local_model)
        PIPE = joblib.load(local_model)

    # 3) tabla precios compra (por neighbourhood)
    purchase_path = CONFIG["data"]["purchase_price_by_neighbourhood"]
    with tempfile.TemporaryDirectory() as td:
        local_csv = os.path.join(td, "purchase.csv")
        gcs_download_file(GCS_BUCKET, purchase_path, local_csv)
        dfp = pd.read_csv(local_csv)

    # Normalización para join (sin acentos / minúsculas / strip)
    dfp.columns = [c.strip() for c in dfp.columns]
    # Ajusta el nombre de la columna si en tu csv se llama distinto
    # Aquí asumimos: neighbourhood_cleansed y purchase_price_mxn
    # Si tu CSV tiene "Alcaldia" y "Precio Promedio", lo adaptamos:
    if "Alcaldia" in dfp.columns and "Precio Promedio" in dfp.columns:
        dfp = dfp.rename(columns={
            "Alcaldia": "neighbourhood_cleansed",
            "Precio Promedio": "purchase_price_mxn"
        })

    dfp["neighbourhood_cleansed_key"] = (
        dfp["neighbourhood_cleansed"].astype(str).str.strip().str.lower()
    )

    # precio a num (por si viene con "MXN" o comas)
    dfp["purchase_price_mxn"] = (
        dfp["purchase_price_mxn"]
        .astype(str)
        .str.replace(r"[^\d\.]", "", regex=True)
    )
    dfp["purchase_price_mxn"] = pd.to_numeric(dfp["purchase_price_mxn"], errors="coerce")

    PURCHASE_DF = dfp[["neighbourhood_cleansed_key", "purchase_price_mxn"]].dropna()


@app.on_event("startup")
def startup_event():
    load_everything()


# =========================
# Input schema
# =========================
class PredictRequest(BaseModel):
    neighbourhood_cleansed: str
    room_type: str

    accommodates: Optional[float] = None
    bathrooms: Optional[float] = None
    bedrooms: Optional[float] = None
    beds: Optional[float] = None
    minimum_nights: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    amenities_count: Optional[int] = 0

    estimated_occupancy_1365d: Optional[float] = Field(
        default=None,
        description="Días ocupados esperados en 365 días (si viene de tu dataset)."
    )


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # defaults
    occ_default = CONFIG["defaults"].get("estimated_occupancy_1365d", 180)
    occ = req.estimated_occupancy_1365d if req.estimated_occupancy_1365d is not None else occ_default

    # armar dataframe input
    x = pd.DataFrame([{
        "neighbourhood_cleansed": req.neighbourhood_cleansed,
        "room_type": req.room_type,
        "accommodates": req.accommodates,
        "bathrooms": req.bathrooms,
        "bedrooms": req.bedrooms,
        "beds": req.beds,
        "minimum_nights": req.minimum_nights,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "amenities_count": req.amenities_count,
    }])

    # predicción (modelo fue entrenado con log1p -> y_log)
    pred_log = float(PIPE.predict(x)[0])
    pred_price_mxn = float(np.expm1(pred_log))

    annual_income_mxn = float(pred_price_mxn * occ)

    # lookup compra
    key = req.neighbourhood_cleansed.strip().lower()
    match = PURCHASE_DF.loc[PURCHASE_DF["neighbourhood_cleansed_key"] == key, "purchase_price_mxn"]

    purchase_price_mxn = float(match.iloc[0]) if len(match) else None

    # payback
    if purchase_price_mxn is None or annual_income_mxn <= 0:
        payback_years = None
        risk_level = CONFIG["risk_rules"].get("nan_label", "No evaluable")
    else:
        payback_years = float(purchase_price_mxn / annual_income_mxn)

        # bins riesgo
        bins = CONFIG["risk_rules"]["bins_years"]
        labels = CONFIG["risk_rules"]["labels"]
        cat = pd.cut([payback_years], bins=bins, labels=labels)[0]
        risk_level = str(cat) if pd.notna(cat) else CONFIG["risk_rules"].get("nan_label", "No evaluable")

    return {
        "pred_price_mxn": pred_price_mxn,
        "estimated_occupancy_1365d": occ,
        "annual_income_mxn": annual_income_mxn,
        "purchase_price_mxn": purchase_price_mxn,
        "payback_years": payback_years,
        "risk_level": risk_level
    }
