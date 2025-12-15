import os
import json
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from google.cloud import storage

# -----------------------------
# ENV
# -----------------------------
API_URL = os.environ.get("API_URL", "https://airbnb-api-1069787915127.europe-west1.run.app")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "0282789_bucket")
CONFIG_GCS_PATH = os.environ.get("CONFIG_GCS_PATH", "airbnb-project/artifacts/config.json")

# NUEVO: path del listings
LISTINGS_GCS_PATH = os.environ.get("LISTINGS_GCS_PATH", "airbnb-project/data/listings.csv")

st.set_page_config(page_title="Airbnb Investment Dashboard", layout="centered")

# -----------------------------
# Helpers GCS
# -----------------------------
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

def load_cfg():
    cfg_text = gcs_download_text(BUCKET_NAME, CONFIG_GCS_PATH)
    return json.loads(cfg_text)

@st.cache_data(ttl=600)
def load_neighbourhoods_and_purchase_table():
    cfg = load_cfg()
    purchase_path = cfg["data"]["purchase_price_by_neighbourhood"]

    gcs_uri = f"gs://{BUCKET_NAME}/{purchase_path}"
    df = pd.read_csv(gcs_uri)

    if "neighbourhood_cleansed" not in df.columns:
        raise ValueError(f"CSV no tiene columna neighbourhood_cleansed. Columnas: {list(df.columns)}")

    neighbourhoods = sorted(df["neighbourhood_cleansed"].astype(str).unique().tolist())
    return cfg, df, neighbourhoods

def post_predict(payload: dict):
    url = f"{API_URL.rstrip('/')}/predict"
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Error {r.status_code}: {r.text}")
    return r.json()

# -----------------------------
# NUEVO: cargar listings.csv para mapa
# -----------------------------
@st.cache_data(ttl=600)
def load_listings_for_map():
    with tempfile.TemporaryDirectory() as tmp:
        local_csv = os.path.join(tmp, "listings.csv")
        gcs_download_to_file(BUCKET_NAME, LISTINGS_GCS_PATH, local_csv)
        df = pd.read_csv(local_csv, encoding="latin1", low_memory=False)

    needed_cols = {"price", "latitude", "longitude"}
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"listings.csv no trae columnas {missing}. Columnas: {list(df.columns)}")

    # price -> num
    df["price_mxn"] = (
        df["price"].astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
    )
    df["price_mxn"] = pd.to_numeric(df["price_mxn"], errors="coerce")

    # ✅ FIX: lat/lon numéricos + limpieza
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "price_mxn"]).copy()
    df = df[np.isfinite(df["price_mxn"])].copy()

    # ✅ FIX: filtra rangos razonables CDMX (evita puntos fuera por datos sucios)
    df = df[(df["latitude"].between(18.5, 20.5)) & (df["longitude"].between(-100.5, -98.0))].copy()

    return df

def pick_comparables(df_map: pd.DataFrame, pred_price: float, k: int = 5) -> pd.DataFrame:
    df = df_map.copy()
    df["diff"] = df["price_mxn"] - float(pred_price)

    below = df[df["diff"] < 0].sort_values("diff", ascending=False).head(k)
    above = df[df["diff"] > 0].sort_values("diff", ascending=True).head(k)

    out = pd.concat([below, above], ignore_index=True)
    return out
def build_map(plot_df: pd.DataFrame, center_lat: float, center_lon: float):
    plot_df = plot_df.copy()
    plot_df["latitude"] = pd.to_numeric(plot_df["latitude"], errors="coerce")
    plot_df["longitude"] = pd.to_numeric(plot_df["longitude"], errors="coerce")
    plot_df["price_mxn"] = pd.to_numeric(plot_df["price_mxn"], errors="coerce")
    plot_df = plot_df.dropna(subset=["latitude", "longitude", "price_mxn"]).copy()

    if plot_df.empty:
        st.warning("No hay puntos válidos para dibujar (lat/lon/price).")
        return

    # ✅ Colores:
    # - TU PREDICCIÓN: verde
    # - abajo: azul
    # - arriba: amarillo
    def _color(row):
        label = str(row.get("label", ""))
        side = str(row.get("side", ""))  # "below" | "above" | "pred"
        if "TU PREDICCIÓN" in label or side == "pred":
            return [0, 200, 0]
        if side == "below":
            return [0, 120, 255]     # azul
        if side == "above":
            return [255, 200, 0]     # amarillo
        return [160, 160, 160]       # fallback

    plot_df["color"] = plot_df.apply(_color, axis=1)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position=["longitude", "latitude"],
        get_radius=120,
        radius_min_pixels=6,
        pickable=True,
        auto_highlight=True,
        get_fill_color="color",
    )

    view_state = pdk.ViewState(
        latitude=float(center_lat),
        longitude=float(center_lon),
        zoom=12,
        pitch=0,
    )

    tooltip = {
        "html": "<b>{label}</b><br/>Precio: <b>${price_mxn}</b> MXN",
        "style": {"backgroundColor": "white", "color": "black"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v10",
    )
    st.pydeck_chart(deck, use_container_width=True)

# -----------------------------
# UI
# -----------------------------
st.title("🏠 Airbnb Investment Dashboard")
st.caption("API para predecir el costo estimado por noche para un AirBnb en CDMX")

with st.expander("🔧 Configuración", expanded=False):
    st.write("API_URL:", API_URL)
    st.write("BUCKET_NAME:", BUCKET_NAME)
    st.write("CONFIG_GCS_PATH:", CONFIG_GCS_PATH)
    st.write("LISTINGS_GCS_PATH:", LISTINGS_GCS_PATH)

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

display_names = [DISPLAY_MAP.get(n, n) for n in neighbourhoods]
display_to_raw = {DISPLAY_MAP.get(n, n): n for n in neighbourhoods}

st.subheader("📥 Ingresar Datos")

neigh_display = st.selectbox("Alcaldía", display_names)
neigh = display_to_raw[neigh_display]

room_type = st.selectbox("Tipo de Alojamiento", room_types)

col1, col2 = st.columns(2)
with col1:
    accommodates = st.number_input("Huéspedes", min_value=1, max_value=20, value=4, step=1)
    bathrooms = st.number_input("Baños", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    bedrooms = st.number_input("Habitaciones", min_value=0.0, max_value=10.0, value=2.0, step=1.0)
    beds = st.number_input("Camas", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
    minimum_nights = st.number_input("Mínimo noches", min_value=1.0, max_value=30.0, value=2.0, step=1.0)

with col2:
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
    "neighbourhood_cleansed": neigh,
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

        st.markdown("---")
        st.subheader("🗺️ Precios inmediatos arriba del precio estimado (amarillo) y abajo (azul)")

        try:
            df_map = load_listings_for_map()

            pred = float(out["pred_price_mxn"])
            comps = pick_comparables(df_map, pred, k=5).copy()

            # ✅ define si está arriba o abajo
            comps["side"] = np.where(comps["price_mxn"] < pred, "below", "above")
            comps["label"] = comps.apply(
            lambda r: f"{'Abajo' if r['side']=='below' else 'Arriba'} (${r['price_mxn']:,.0f})",
            axis=1
            )
            
            user_point = pd.DataFrame([{
                "latitude": float(payload["latitude"]),
                "longitude": float(payload["longitude"]),
                "price_mxn": pred,
                "label": "TU PREDICCIÓN",
                "side": "pred"
            }])

            plot_df = pd.concat(
                [user_point, comps[["latitude", "longitude", "price_mxn", "label", "side"]]],
                ignore_index=True
            )


            build_map(
                plot_df,
                center_lat=float(payload["latitude"]),
                center_lon=float(payload["longitude"])
            )

            with st.expander("Ver comparables (tabla)"):
                show_cols = [c for c in ["neighbourhood_cleansed", "room_type", "latitude", "longitude", "price_mxn"] if c in comps.columns]
                st.dataframe(comps[show_cols].sort_values("price_mxn"), use_container_width=True)

        except Exception as e:
            st.warning(f"No pude mostrar el mapa de comparables: {e}")

    except Exception as e:
        st.error(f"Error llamando a la API: {e}")
        st.info("Tip: prueba primero abrir /docs de tu API y verificar que /predict responde.")

