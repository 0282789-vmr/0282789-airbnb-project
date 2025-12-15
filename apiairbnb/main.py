# main.py
import os
import json
import tempfile
from typing import Literal, Optional

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.cloud import storage


# ============================================================
# Config por variables de entorno (Cloud Run)
# ============================================================
BUCKET_NAME = os.environ.get("BUCKET_NAME", "0282789_bucket")
CONFIG_GCS_PATH = os.environ.get("CONFIG_GCS_PATH", "airbnb-project/artifacts/config.json")


# ============================================================
# Helpers GCS
# ============================================================
def gcs_download_text(bucket_name: str, blob_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_text()

def gcs_download_to_file(bucket_name: str, blob_path: str, local_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)

def gcs_download_bytes(bucket_name: str, blob_path: str) -> bytes:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()


# ============================================================
# App + caché en memoria
# ============================================================
app = FastAPI(title="Airbnb Investment API", version="1.0.0")

CFG = None
MODEL = None
PURCHASE_TABLE = None  # df con purchase_price_by_neighbourhood


# ============================================================
# Esquemas de entrada/salida
# ============================================================
class PredictRequest(BaseModel):
    neighbourhood_cleansed: str = Field(..., example="Coyoacan")
    room_type: str = Field(..., example="Entire home/apt")

    accommodates: Optional[float] = Field(None, example=4)
    bathrooms: Optional[float] = Field(None, example=1)
    bedrooms: Optional[float] = Field(None, example=2)
    beds: Optional[float] = Field(None, example=2)
    minimum_nights: Optional[float] = Field(None, example=2)
    latitude: Optional[float] = Field(None, example=19.35)
    longitude: Optional[float] = Field(None, example=-99.16)

    # OJO: el modelo espera amenities_count numérico
    amenities_count: Optional[int] = Field(0, example=12)

    # Para el cálculo del negocio
    estimated_occupancy_1365d: Optional[float] = Field(
        None,
        ge=0,
        le=365,
        description="Noches ocupadas en un AÑO (0–365). (Nombre histórico: *_1365d)",
        example=180
    )

    square_meters: Optional[float] = Field(
        None,
        ge=1,
        le=2000,
        description="Metros cuadrados del inmueble. Se usa para calcular precio de compra total = purchase_price_mxn (precio/m²) * m².",
        example=80
    )



class PredictResponse(BaseModel):
    pred_price_mxn: float
    annual_income_mxn: Optional[float]
    purchase_price_mxn: Optional[float]
    payback_years: Optional[float]
    risk_level: str
    model_version: str


# ============================================================
# Lógica de negocio
# ============================================================
def load_config_and_assets():
    global CFG, MODEL, PURCHASE_TABLE

    # 1) config.json
    cfg_text = gcs_download_text(BUCKET_NAME, CONFIG_GCS_PATH)
    CFG = json.loads(cfg_text)

    # 2) modelo joblib (descargar a temp y joblib.load)
    model_path = CFG["model"]["path"]
    with tempfile.TemporaryDirectory() as tmp:
        local_model = os.path.join(tmp, "model.joblib")
        gcs_download_to_file(BUCKET_NAME, model_path, local_model)
        MODEL = joblib.load(local_model)

    # 3) tabla de precios compra por neighbourhood
    purchase_path = CFG["data"]["purchase_price_by_neighbourhood"]
    with tempfile.TemporaryDirectory() as tmp:
        local_csv = os.path.join(tmp, "purchase.csv")
        gcs_download_to_file(BUCKET_NAME, purchase_path, local_csv)
        PURCHASE_TABLE = pd.read_csv(local_csv)

    # Normalizar nombres esperados (según tu screenshot)
    # Debe tener: neighbourhood_cleansed, purchase_price_mxn
    needed = {"neighbourhood_cleansed", "purchase_price_mxn"}
    if not needed.issubset(set(PURCHASE_TABLE.columns)):
        raise RuntimeError(
            f"purchase_price_by_neighbourhood.csv debe contener columnas {needed}. "
            f"Encontradas: {list(PURCHASE_TABLE.columns)}"
        )

    # Asegurar numérico
    PURCHASE_TABLE["purchase_price_mxn"] = pd.to_numeric(
        PURCHASE_TABLE["purchase_price_mxn"], errors="coerce"
    )


def get_purchase_price(neighbourhood_cleansed: str) -> Optional[float]:
    """
    Busca precio_compra por neighbourhood.
    Nota: tu tabla NO tiene acentos (según me dijiste),
    así que usa exactamente ese formato en API.
    """
    if PURCHASE_TABLE is None or PURCHASE_TABLE.empty:
        return None

    mask = PURCHASE_TABLE["neighbourhood_cleansed"].astype(str) == str(neighbourhood_cleansed)
    row = PURCHASE_TABLE.loc[mask].head(1)
    if row.empty:
        return None

    v = row["purchase_price_mxn"].iloc[0]
    if pd.isna(v) or not np.isfinite(v):
        return None
    return float(v)


def compute_risk(payback_years: Optional[float]) -> str:
    """
    bins_years en config: [-1, 8, 12, 100]
    labels: ["Bajo", "Medio", "Alto"]
    nan_label: "No evaluable"
    """
    rr = CFG.get("risk_rules", {})
    bins = rr.get("bins_years", [-1, 8, 12, 100])
    labels = rr.get("labels", ["Bajo", "Medio", "Alto"])
    nan_label = rr.get("nan_label", "No evaluable")

    if payback_years is None or (isinstance(payback_years, float) and not np.isfinite(payback_years)):
        return nan_label

    # fuera de rango
    if payback_years < 0:
        return nan_label

    # binning manual
    # [-1,8] => Bajo, (8,12] => Medio, (12,100] => Alto
    if payback_years <= bins[1]:
        return labels[0]
    if payback_years <= bins[2]:
        return labels[1]
    return labels[2]


def build_feature_df(req: PredictRequest) -> pd.DataFrame:
    feats = CFG["features"]
    cat_cols = feats["categorical"]
    num_cols = feats["numerical"]
    all_cols = cat_cols + num_cols

    data = {
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
    }

    df = pd.DataFrame([data])

    # Asegurar que existan columnas esperadas
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Cast numéricos
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[all_cols]


# ============================================================
# Startup
# ============================================================
@app.on_event("startup")
def startup_event():
    try:
        load_config_and_assets()
    except Exception as e:
        # Si falla aquí, Cloud Run lo verá en logs y el contenedor podría seguir vivo,
        # pero mejor levantar error fuerte:
        raise RuntimeError(f"Error cargando assets desde GCS: {e}")


# ============================================================
# Endpoints
# ============================================================
@app.get("/health")
def health():
    if CFG is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "model": CFG["model"]["name"], "version": CFG["model"]["version"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if CFG is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1) armar df de features
    try:
        X = build_feature_df(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2) predecir en log y convertir a MXN
    pred_log = float(MODEL.predict(X)[0])
    pred_price_mxn = float(np.expm1(pred_log))  # porque entrenaste con log1p(price_num)

    # 3) ocupación (noches ocupadas en 1365d)
    default_occ = CFG.get("defaults", {}).get("estimated_occupancy_1365d", 180)
    occ_1365d = req.estimated_occupancy_1365d if req.estimated_occupancy_1365d is not None else default_occ

    # 4) estimated_occupancy_1365d = noches ocupadas en un año (0–365)
    annual_income_mxn = pred_price_mxn * occ_1365d


    # 5) purchase_price_mxn en la tabla es PRECIO POR m²
    price_m2 = get_purchase_price(req.neighbourhood_cleansed)

    purchase_price_mxn = None
    if price_m2 is not None and req.square_meters is not None:
        purchase_price_mxn = float(price_m2) * float(req.square_meters)


    # 6) payback
    payback_years = None
    if purchase_price_mxn is not None and annual_income_mxn is not None and annual_income_mxn > 0:
        payback_years = float(purchase_price_mxn / annual_income_mxn)

    # 7) riesgo
    risk_level = compute_risk(payback_years)

    return PredictResponse(
        pred_price_mxn=pred_price_mxn,
        annual_income_mxn=annual_income_mxn,
        purchase_price_mxn=purchase_price_mxn,
        payback_years=payback_years,
        risk_level=risk_level,
        model_version=str(CFG["model"]["version"]),
    )



