import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from azure.storage.blob import BlobServiceClient
import io
import re
import time
import json
import requests

st.set_page_config(
    page_title="BDA Gastronomía Perú",
    page_icon="🍽️",
    layout="wide"
)

# ── Constantes ────────────────────────────────────────────────
CONTENEDOR = "bda-proyecto"
PREFIJO    = "streamlit_data"
CATALOGO   = "proyecto_bda"
SCHEMA     = "bda_schema"

# Ruta del notebook de inferencia en tu workspace Databricks.
# Crea el notebook en esa ruta y ajusta si lo guardas en otro lugar.
NOTEBOOK_INFERENCIA = "/Shared/BDA_Inferencia"

NOMBRES_CLUSTERS = {
    0: "Decepcionado silencioso",
    1: "Recurrente satisfecho",
    2: "Validador masivo",
    3: "Crítico detallista",
    4: "Foodie activo",
    5: "Promotor genuino",
    6: "Experiencia errática",
}
NOMBRES_TOPICOS = {
    0: "Experiencia general",
    1: "Pollo a la brasa",
    2: "Calidad-precio",
    3: "Recomendación positiva",
    4: "Ceviche y platos bandera",
}


# ── Azure Blob ────────────────────────────────────────────────
@st.cache_resource
def get_blob_client():
    return BlobServiceClient.from_connection_string(
        st.secrets["AZURE_CONN_STR"]
    )


def leer_csv(nombre: str) -> pd.DataFrame:
    client = get_blob_client()
    blob   = client.get_blob_client(container=CONTENEDOR, blob=f"{PREFIJO}/{nombre}")
    data   = blob.download_blob().readall()
    return pd.read_csv(io.BytesIO(data))


# ── Jobs API helpers ──────────────────────────────────────────
def _headers() -> dict:
    return {
        "Authorization": f"Bearer {st.secrets['DATABRICKS_TOKEN']}",
        "Content-Type":  "application/json",
    }


def _base_url() -> str:
    host = st.secrets["DATABRICKS_HOST"].rstrip("/")
    return f"{host}/api/2.1/jobs"


def _submit_run(parametros: dict) -> str:
    """
    Lanza el notebook de inferencia vía runs/submit (one-time run, sin job permanente).
    Devuelve el run_id como string.
    """
    payload = {
        "run_name": "bda_inferencia_streamlit",
        "tasks": [
            {
                "task_key": "inferencia",
                "notebook_task": {
                    "notebook_path": NOTEBOOK_INFERENCIA,
                    "base_parameters": {k: str(v) for k, v in parametros.items()},
                    "source": "WORKSPACE",
                },
                "serverless": True,
            }
        ],
    }
    r = requests.post(
        f"{_base_url()}/runs/submit",
        headers=_headers(),
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return str(r.json()["run_id"])


def _poll_run(run_id: str, timeout: int = 300, intervalo: int = 5) -> dict:
    """
    Hace polling sobre runs/get hasta que el run termine (TERMINATED / ERROR / etc.).
    Devuelve el dict completo del run para que el caller inspeccione el resultado.
    Lanza RuntimeError si supera el timeout o si el run falla.
    """
    url       = f"{_base_url()}/runs/get"
    deadline  = time.time() + timeout
    estados_finales = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}

    while time.time() < deadline:
        r = requests.get(url, headers=_headers(), params={"run_id": run_id}, timeout=15)
        r.raise_for_status()
        data        = r.json()
        life_cycle  = data.get("state", {}).get("life_cycle_state", "")
        result_state= data.get("state", {}).get("result_state", "")

        if life_cycle in estados_finales:
            if result_state != "SUCCESS":
                msg = data.get("state", {}).get("state_message", "Run failed")
                # Intentar obtener el error detallado del task
                try:
                    r2 = requests.get(
                        f"{_base_url()}/runs/get-output",
                        headers=_headers(),
                        params={"run_id": run_id},
                        timeout=15,
                    )
                    out = r2.json()
                    error_detail = out.get("error", "") or out.get("error_trace", "")
                    task_errors  = [
                        t.get("state", {}).get("state_message", "")
                        for t in data.get("tasks", [])
                    ]
                    detalle = error_detail or " | ".join(filter(None, task_errors)) or msg
                except Exception:
                    detalle = msg
                raise RuntimeError(f"Run {run_id} [{result_state}]: {detalle}")
            return data

        time.sleep(intervalo)

    raise RuntimeError(f"Timeout esperando run {run_id} tras {timeout}s")


def _notebook_output(run_data: dict) -> dict:
    """
    Extrae el resultado JSON que emitió dbutils.notebook.exit().
    runs/get-output requiere el run_id del task, no el del run raíz.
    """
    tasks = run_data.get("tasks", [])
    task_run_id = tasks[0]["run_id"] if tasks else run_data["run_id"]

    r = requests.get(
        f"{_base_url()}/runs/get-output",
        headers=_headers(),
        params={"run_id": task_run_id},
        timeout=15,
    )
    r.raise_for_status()
    salida_raw = r.json().get("notebook_output", {}).get("result", "{}")
    return json.loads(salida_raw)


def inferir_kmeans(review_count: int, avg_rating: float,
                   std_rating: float, avg_word_count: int) -> tuple[int, str]:
    """Lanza el notebook en modo kmeans y devuelve (cluster_id, nombre)."""
    params = {
        "modo":               "kmeans",
        "log_review_count":   float(np.log1p(review_count)),
        "avg_rating":         avg_rating,
        "std_rating":         std_rating,
        "log_avg_word_count": float(np.log1p(avg_word_count)),
    }
    run_id   = _submit_run(params)
    run_data = _poll_run(run_id)
    resultado= _notebook_output(run_data)
    cluster  = resultado["cluster"]
    nombre   = resultado.get("nombre", NOMBRES_CLUSTERS.get(cluster, f"Cluster {cluster}"))
    return cluster, nombre


def inferir_lda(texto: str) -> tuple[int, str]:
    """Lanza el notebook en modo lda y devuelve (topico_id, nombre)."""
    params   = {"modo": "lda", "texto": texto}
    run_id   = _submit_run(params)
    run_data = _poll_run(run_id)
    resultado= _notebook_output(run_data)
    tid      = resultado["topico"]
    nombre   = resultado.get("nombre", NOMBRES_TOPICOS.get(tid, f"Tópico {tid}"))
    return tid, nombre


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
st.title("🍽️ BDA Gastronomía Perú")
st.caption("Framework de Big Data Analytics sobre reseñas de restaurantes peruanos · Databricks + Spark MLlib")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Inferencia en tiempo real",
    "📊 Estadísticas por categoría",
    "👥 Segmentación de usuarios",
    "💬 Tópicos de reseñas",
    "🗺️ Mapa geográfico",
])

# ── Tab 1: Inferencia ─────────────────────────────────────────
with tab1:
    st.header("Inferencia en tiempo real")
    st.info(
        "La inferencia corre en Databricks vía Jobs API. "
        "El cluster tarda ~2–3 min en inicializarse la primera vez; "
        "las siguientes llamadas son más rápidas si el cluster sigue activo."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 Segmentación KMeans")
        review_count     = st.number_input("Número de reseñas del usuario", min_value=1, max_value=5000, value=10)
        avg_rating_input = st.slider("Rating promedio", 1.0, 5.0, 4.0, 0.1)
        std_rating_input = st.slider("Desviación estándar del rating", 0.0, 2.5, 0.5, 0.1)
        avg_word_input   = st.number_input("Palabras promedio por reseña", min_value=1, max_value=500, value=20)

        if st.button("Predecir segmento", type="primary"):
            with st.spinner("Enviando job a Databricks… (puede tardar 2–3 min la primera vez)"):
                try:
                    cluster_id, nombre = inferir_kmeans(
                        review_count, avg_rating_input, std_rating_input, avg_word_input
                    )
                    st.success(f"**Cluster {cluster_id} — {nombre}**")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("💬 Clasificación LDA")
        texto_input = st.text_area(
            "Escribe una reseña de restaurante",
            placeholder="Ej: El ceviche estaba riquísimo, muy fresco y bien presentado.",
            height=150,
        )

        if st.button("Predecir tópico", type="primary"):
            if not texto_input.strip():
                st.warning("Escribe una reseña primero.")
            else:
                with st.spinner("Enviando job a Databricks… (puede tardar 2–3 min la primera vez)"):
                    try:
                        tid, nombre = inferir_lda(texto_input)
                        st.success(f"**Tópico {tid} — {nombre}**")
                    except Exception as e:
                        st.error(f"Error: {e}")

# ── Tab 2: Estadísticas ───────────────────────────────────────
with tab2:
    st.header("Estadísticas por categoría de restaurante")
    with st.spinner("Cargando datos..."):
        df_stats = leer_csv("gold_stats_categoria.csv")

    col_rating = [c for c in df_stats.columns if "rating" in c.lower() or "avg" in c.lower()][0]
    col_cat    = df_stats.columns[0]

    top_n  = st.slider("Top N categorías", 5, 30, 15)
    df_top = df_stats.nlargest(top_n, col_rating)

    fig = px.bar(
        df_top, x=col_rating, y=col_cat,
        orientation="h", title=f"Top {top_n} categorías por rating promedio",
        color=col_rating, color_continuous_scale="Teal",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_stats, use_container_width=True)

# ── Tab 3: Clusters ───────────────────────────────────────────
with tab3:
    st.header("Segmentación histórica de usuarios (KMeans, k=7)")
    with st.spinner("Cargando datos..."):
        df_cl = leer_csv("gold_clusters.csv")

    if "cluster_nombre" not in df_cl.columns:
        df_cl["cluster_nombre"] = df_cl["cluster"].map(NOMBRES_CLUSTERS)

    conteo = df_cl.groupby(["cluster", "cluster_nombre"]).size().reset_index(name="usuarios")
    fig2   = px.bar(
        conteo, x="cluster_nombre", y="usuarios",
        title="Usuarios por segmento", color="cluster",
        color_continuous_scale="Viridis",
        labels={"cluster_nombre": "Segmento", "usuarios": "Usuarios"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "avg_rating" in df_cl.columns:
        fig3 = px.scatter(
            df_cl.sample(min(5000, len(df_cl))),
            x="avg_rating", y="avg_word_count" if "avg_word_count" in df_cl.columns else df_cl.columns[-1],
            color="cluster_nombre",
            title="Satisfacción vs. Esfuerzo textual por segmento",
            labels={"avg_rating": "Rating promedio", "avg_word_count": "Palabras promedio"},
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── Tab 4: Tópicos ────────────────────────────────────────────
with tab4:
    st.header("Tópicos de reseñas (LDA, k=5)")
    with st.spinner("Cargando datos..."):
        df_top_data = leer_csv("gold_topicos.csv")

    if "nombre_topico" not in df_top_data.columns:
        df_top_data["nombre_topico"] = df_top_data["topico_id"].map(NOMBRES_TOPICOS)

    conteo_top = df_top_data.groupby(["topico_id", "nombre_topico"]).size().reset_index(name="reseñas")
    fig4       = px.pie(
        conteo_top, names="nombre_topico", values="reseñas",
        title="Distribución de tópicos en reseñas",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig4, use_container_width=True)

    topico_sel = st.selectbox("Ver reseñas de tópico:", options=conteo_top["nombre_topico"].tolist())
    muestra    = df_top_data[df_top_data["nombre_topico"] == topico_sel][["username", "caption_clean"]].head(10)
    st.dataframe(muestra, use_container_width=True)

# ── Tab 5: Mapa ───────────────────────────────────────────────
with tab5:
    st.header("Distribución geográfica de reseñas")
    with st.spinner("Cargando muestra geográfica..."):
        df_geo = leer_csv("gold_reviews_sample.csv")

    df_geo = df_geo.dropna(subset=["latitude", "longitude"])
    df_geo = df_geo[
        df_geo["latitude"].between(-18.5, -0.1) &
        df_geo["longitude"].between(-81.5, -68.5)
    ]

    fig5 = px.scatter_mapbox(
        df_geo.sample(min(10000, len(df_geo))),
        lat="latitude", lon="longitude",
        color="rating", color_continuous_scale="RdYlGn",
        size_max=5, zoom=4,
        center={"lat": -9.5, "lon": -75.0},
        mapbox_style="carto-positron",
        title="Muestra de reseñas georeferenciadas",
        hover_data=["rating"],
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.metric("Reseñas en muestra", f"{len(df_geo):,}")