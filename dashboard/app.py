import os
import json
import requests
import pandas as pd
import streamlit as st
from google.cloud import storage

# -----------------------------
# ENV
# -----------------------------
API_URL = os.environ.get("API_URL", "https://airbnb-api-1069787915127.europe-west1.run.app")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "0282789_bucket")
CONFIG_GCS_PATH = os.environ.get("CONFIG_GCS_PATH", "airbnb-project/artifacts/config.json")

st.set_page_config(page_title="Airbnb Investment Dashboard", layout="centered")

# -----------------------------
# Helpers GCS
# -----------------------------
def gcs_download_text(bucket_name: str, blob_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_text()

def load_cfg():
    cfg_text = gcs_download_text(BUCKET_NAME, CONFIG_GCS_PATH)
    return json.loads(cfg_text)

@st.cache_data(ttl=600)
def load_neighbourhoods_and_purchase_table():
    cfg = load_cfg()
    purchase_path = cfg["data"]["purchase_price_by_neighbourhood"]

    # leer CSV directo de GCS
    gcs_uri = f"gs://{BUCKET_NAME}/{purchase_path}"
    df = pd.read_csv(gcs_uri)

    # Normaliza columnas esperadas
    if "neighbourhood_cleansed" not in df.columns:
        raise ValueError(f"CSV no tiene columna neighbourhood_cleansed. Columnas: {list(df.columns)}")

    # Lista de neighbourhoods (valores "raw" SIN acentos)
    neighbourhoods = sorted(df["neighbourhood_cleansed"].astype(str).unique().tolist())
    return cfg, df, neighbourhoods

def post_predict(payload: dict):
    url = f"{API_URL.rstrip('/')}/predict"
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Error {r.status_code}: {r.text}")
    return r.json()

# -----------------------------
# UI
# -----------------------------
st.title("🏠 Airbnb Investment Dashboard")
st.caption("API para predecir el costo estimado por noche para un AirBnb en CDMX")

with st.expander("🔧 Configuración", expanded=False):
    st.write("API_URL:", API_URL)
    st.write("BUCKET_NAME:", BUCKET_NAME)
    st.write("CONFIG_GCS_PATH:", CONFIG_GCS_PATH)

try:
    cfg, purchase_df, neighbourhoods = load_neighbourhoods_and_purchase_table()
except Exception as e:
    st.error(f"No pude cargar config/tabla desde GCS: {e}")
    st.stop()

room_types = [
    "Entire home/apt",
    "Private room",
    "Shared room",
    "Hotel room",
]

# -----------------------------
# Display map (UI con acentos) -> raw (sin acentos) para API/modelo
# -----------------------------
DISPLAY_MAP = {
    "Alvaro Obregon": "Álvaro Obregón",
    "Azcapotzalco": "Azcapotzalco",
    "Benito Juarez": "Benito Juárez",
    "Coyoacan": "Coyoacán",
    "Cuajimalpa de Morelos": "Cuajimalpa de Morelos",
    "Cuauhtemoc": "Cuauhtémoc",
    "Gustavo A. Madero": "Gustavo A. Madero",
    "Iztacalco": "Iztacalco",
    "Iztapalapa": "Iztapalapa",
    "La Magdalena Contreras": "La Magdalena Contreras",
    "Miguel Hidalgo": "Miguel Hidalgo",
    "Milpa Alta": "Milpa Alta",
    "Tlalpan": "Tlalpan",
    "Tlahuac": "Tláhuac",
    "Venustiano Carranza": "Venustiano Carranza",
    "Xochimilco": "Xochimilco",
}

# Armamos lista “bonita” para UI en el mismo orden que neighbourhoods
display_names = [DISPLAY_MAP.get(n, n) for n in neighbourhoods]
# Mapa inverso: display -> raw
display_to_raw = {DISPLAY_MAP.get(n, n): n for n in neighbourhoods}

st.subheader("📥 Ingresar Datos")

# Selectbox muestra con acentos, pero guardamos el raw
neigh_display = st.selectbox("Alcaldía", display_names)
neigh = display_to_raw[neigh_display]  # ESTE es el que se manda a la API

room_type = st.selectbox("Tipo de Alojamiento", room_types)

col1, col2 = st.columns(2)
with col1:
    accommodates = st.number_input("Huéspedes", min_value=1, max_value=20, value=4, step=1)
    bathrooms = st.number_input("Baños", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    bedrooms = st.number_input("Habitaciones", min_value=0.0, max_value=10.0, value=2.0, step=1.0)
    beds = st.number_input("Camas", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
    minimum_nights = st.number_input("Mínimo noches", min_value=1.0, max_value=30.0, value=2.0, step=1.0)

with col2:
    # Metros cuadrados (se usa para calcular compra total)
    square_meters = st.number_input(
        "Metros cuadrados (m²)",
        min_value=1.0,
        max_value=2000.0,
        value=80.0,
        step=1.0,
        help="Se usa para calcular compra estimada = (precio por m² del CSV) × m²."
    )

    latitude = st.number_input("Latitude", value=19.35, format="%.6f")
    longitude = st.number_input("Longitude", value=-99.16, format="%.6f")
    amenities_count = st.number_input("Amenidades", min_value=0, max_value=300, value=12, step=1)

st.markdown("---")
st.subheader("💼 Estimado de ocupación (anual)")

default_occ = cfg.get("defaults", {}).get("estimated_occupancy_1365d", 180)
occ_annual = st.slider(
    "Ocupación en un año (0–365)",
    min_value=0,
    max_value=365,
    value=int(default_occ) if default_occ is not None else 180,
    step=1,
)

payload = {
    "neighbourhood_cleansed": neigh,  # raw sin acentos
    "room_type": room_type,
    "accommodates": float(accommodates),
    "bathrooms": float(bathrooms),
    "bedrooms": float(bedrooms),
    "beds": float(beds),
    "minimum_nights": float(minimum_nights),
    "latitude": float(latitude),
    "longitude": float(longitude),
    "amenities_count": int(amenities_count),
    "estimated_occupancy_1365d": float(occ_annual),
    "square_meters": float(square_meters),
}

if st.button("🚀 Predecir", type="primary"):
    try:
        out = post_predict(payload)
        st.success("Predicción completada ✅")

        st.subheader("📊 Resultados")
        st.metric("Precio por noche (MXN)", f"{out['pred_price_mxn']:,.2f}")

        colA, colB = st.columns(2)
        with colA:
            st.metric("Ingreso anual (MXN)", f"{out['annual_income_mxn']:,.2f}" if out.get("annual_income_mxn") is not None else "—")
            st.metric("Compra estimada (MXN)", f"{out['purchase_price_mxn']:,.2f}" if out.get("purchase_price_mxn") is not None else "—")
        with colB:
            st.metric("Retorno de Inversión (años)", f"{out['payback_years']:.2f}" if out.get("payback_years") is not None else "—")
            st.metric("Riesgo", out.get("risk_level", "—"))

        st.caption(f"Model version: {out.get('model_version','—')}")
        with st.expander("JSON completo"):
            st.json(out)

    except Exception as e:
        st.error(f"Error llamando a la API: {e}")
        st.info("Tip: prueba primero abrir /docs de tu API y verificar que /predict responde.")

