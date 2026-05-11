# ================================================================
# CME Forecast App — Streamlit
# Bilingüe ES/EN | ARIMA vs ARIMAX | Solar Cycle 25
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import io

warnings.filterwarnings("ignore")

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="CME Forecast | Predicción de CMEs",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# CUSTOM CSS — Scientific dark theme
# ================================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Crimson Pro', Georgia, serif;
    background-color: #0d1117;
    color: #e6edf3;
  }
  h1, h2, h3 { font-family: 'Crimson Pro', serif; font-weight: 600; letter-spacing: 0.02em; }
  h1 { font-size: 2.4rem; color: #f0a500; border-bottom: 2px solid #f0a500; padding-bottom: 0.3rem; }
  h2 { font-size: 1.6rem; color: #58a6ff; }
  h3 { font-size: 1.2rem; color: #c9d1d9; }

  .stSidebar { background-color: #161b22 !important; border-right: 1px solid #30363d; }
  .stSidebar label { color: #8b949e !important; font-size: 0.85rem; }

  .metric-box {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-box:hover { border-color: #58a6ff; }
  .metric-label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; }
  .metric-value { font-size: 1.8rem; font-weight: 600; color: #f0a500; font-family: 'JetBrains Mono', monospace; }
  .metric-sub   { font-size: 0.82rem; color: #8b949e; margin-top: 0.2rem; }

  .info-card {
    background: #161b22;
    border-left: 4px solid #58a6ff;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
  }
  .warn-card {
    background: #1f1b0f;
    border-left: 4px solid #f0a500;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
  }

  .stButton>button {
    background: linear-gradient(135deg, #f0a500, #e06c00);
    color: #0d1117;
    font-family: 'Crimson Pro', serif;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    transition: opacity 0.2s;
  }
  .stButton>button:hover { opacity: 0.85; }

  .badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin: 0.1rem;
  }
  .badge-blue  { background: #1f3a5f; color: #58a6ff; border: 1px solid #58a6ff; }
  .badge-red   { background: #3d1515; color: #f85149; border: 1px solid #f85149; }
  .badge-gold  { background: #2d2000; color: #f0a500; border: 1px solid #f0a500; }

  .tab-header { font-size: 1rem; font-weight: 600; padding: 0.3rem 0; }
  footer { visibility: hidden; }

  /* Table */
  .stDataFrame { background: #161b22; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# LANGUAGE TOGGLE
# ================================================================
LANG = st.sidebar.radio("🌐 Language / Idioma", ["Español", "English"], index=0)
ES = (LANG == "Español")

def t(es_text, en_text):
    return es_text if ES else en_text

# ================================================================
# SIDEBAR — CONFIGURATION
# ================================================================
st.sidebar.markdown(f"## {'⚙️ Configuración' if ES else '⚙️ Configuration'}")

st.sidebar.markdown(f"### {'📂 Archivos de datos' if ES else '📂 Data Files'}")
cme_file = st.sidebar.file_uploader(
    t("Datos CME (datos_procesados_*.csv)", "CME Data (datos_procesados_*.csv)"),
    type=["csv"], key="cme"
)
ssn_annual_file = st.sidebar.file_uploader(
    t("SSN anual (SN_y_tot_V2.0.txt)", "Annual SSN (SN_y_tot_V2.0.txt)"),
    type=["txt"], key="ssn_a"
)
ssn_monthly_file = st.sidebar.file_uploader(
    t("SSN mensual (SN_m_tot_V2.0.txt) — opcional", "Monthly SSN (SN_m_tot_V2.0.txt) — optional"),
    type=["txt"], key="ssn_m"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### {'🔭 Filtros CME' if ES else '🔭 CME Filters'}")

min_speed = st.sidebar.slider(
    t("Velocidad mínima (km/s)", "Minimum speed (km/s)"),
    min_value=400, max_value=1500, value=750, step=50
)
min_width = st.sidebar.slider(
    t("Ancho mínimo (°)", "Minimum width (°)"),
    min_value=0, max_value=180, value=0, step=10
)
max_width = st.sidebar.slider(
    t("Ancho máximo (°)", "Maximum width (°)"),
    min_value=181, max_value=360, value=360, step=10
)
year_start = st.sidebar.slider(
    t("Año inicio entrenamiento", "Training start year"),
    min_value=1996, max_value=2010, value=1996, step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### {'📈 Modelos' if ES else '📈 Models'}")
show_insample = st.sidebar.checkbox(t("Mostrar ajuste en muestra", "Show in-sample fit"), value=True)
alpha_ci = st.sidebar.selectbox(
    t("Nivel de confianza del IC", "Confidence interval level"),
    options=[0.05, 0.10, 0.20],
    format_func=lambda x: f"{int((1-x)*100)}%",
    index=0
)

run_btn = st.sidebar.button(f"{'🚀 Ejecutar predicción' if ES else '🚀 Run Forecast'}", use_container_width=True)

# ================================================================
# HEADER
# ================================================================
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size:3.5rem; padding-top:0.4rem;'>☀️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown(f"# {t('Predicción de CMEs Solares 2025–2026', 'Solar CME Forecast 2025–2026')}")
    st.markdown(
        f"<span class='badge badge-blue'>ARIMA</span>"
        f"<span class='badge badge-red'>ARIMAX</span>"
        f"<span class='badge badge-gold'>SILSO SC-25</span>"
        f" &nbsp; <span style='color:#8b949e; font-size:0.9rem;'>"
        f"{t('Ciclo Solar 25 · Entrenamiento 1996–2024', 'Solar Cycle 25 · Training 1996–2024')}"
        f"</span>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ================================================================
# ABOUT SECTION (always visible)
# ================================================================
with st.expander(t("ℹ️ Acerca de esta aplicación", "ℹ️ About this application"), expanded=False):
    st.markdown(t(
        """
        Esta aplicación implementa los modelos estadísticos presentados en nuestro artículo científico
        para predecir el número de **Eyecciones de Masa Coronal (CMEs) de alta velocidad** durante 2025–2026,
        en el contexto del **Ciclo Solar 25**.

        **Metodología:**
        - Se filtran CMEs del catálogo LASCO/CDAW por velocidad y ancho angular
        - Se ajustan modelos **ARIMA** (univariado) y **ARIMAX** (con manchas solares como variable exógena)
        - Los escenarios de manchas solares provienen de las predicciones oficiales **SILSO** para el Ciclo Solar 25
        - Se proyectan 3 escenarios: Curva Estándar (SC), Método Combinado (CM), McNish & Lincoln (ML)

        **📂 Datos y código disponibles públicamente:**
        Los datos preprocesados, los criterios de preprocesamiento y el código completo de implementación
        de los modelos están disponibles en el repositorio de GitHub:
        [github.com/JuanAnarch7/Forecasting-Coronal-Mass-Ejection-Occurrence-Rates-Using-ARIMA-and-ARIMAX-Models](https://github.com/JuanAnarch7/Forecasting-Coronal-Mass-Ejection-Occurrence-Rates-Using-ARIMA-and-ARIMAX-Models)

        **Datos requeridos:** Archivos CSV/TXT de la misión LASCO y el catálogo SILSO (WDC-SILSO, Observatorio Real de Bélgica).

        ---

        **👥 Autores:** Este trabajo fue desarrollado por Juan Rafael Jiménez Sanabria, Cristian David Chávez Aponte y Daniel Felipe Pineda Cruz.
        """,
        """
        This application implements the statistical models presented in our scientific paper
        to predict the number of **high-speed Coronal Mass Ejections (CMEs)** during 2025–2026,
        in the context of **Solar Cycle 25**.

        **Methodology:**
        - CMEs from the LASCO/CDAW catalog are filtered by speed and angular width
        - **ARIMA** (univariate) and **ARIMAX** (with sunspots as exogenous variable) models are fitted
        - Sunspot scenarios come from official **SILSO** predictions for Solar Cycle 25
        - 3 scenarios are projected: Standard Curve (SC), Combined Method (CM), McNish & Lincoln (ML)

        **📂 Publicly available data and code:**
        The preprocessed data, preprocessing criteria, and full model implementation code
        are available at the GitHub repository:
        [github.com/JuanAnarch7/Forecasting-Coronal-Mass-Ejection-Occurrence-Rates-Using-ARIMA-and-ARIMAX-Models](https://github.com/JuanAnarch7/Forecasting-Coronal-Mass-Ejection-Occurrence-Rates-Using-ARIMA-and-ARIMAX-Models)

        **Required data:** CSV/TXT files from the LASCO mission and the SILSO catalog (WDC-SILSO, Royal Observatory of Belgium).

        ---

        **👥 Authors:** This work was developed by Juan Rafael Jiménez Sanabria, Cristian David Chávez Aponte and Daniel Felipe Pineda Cruz.
        """
    ))

# ================================================================
# SILSO DATA (hardcoded, as in original script)
# ================================================================
SILSO_MONTHLY = {
    'SC': [
        (2025,9,113.5),(2025,10,116.1),(2025,11,113.1),(2025,12,111.9),
        (2026,1,110.4),(2026,2,108.9),(2026,3,107.3),(2026,4,105.6),
        (2026,5,103.8),(2026,6,101.8),(2026,7,99.8),(2026,8,97.6),
        (2026,9,95.5),(2026,10,93.0),(2026,11,90.2),(2026,12,87.5),
    ],
    'CM': [
        (2025,9,115.0),(2025,10,113.1),(2025,11,113.6),(2025,12,114.6),
        (2026,1,114.9),(2026,2,114.9),(2026,3,114.5),(2026,4,113.0),
        (2026,5,111.5),(2026,6,109.4),(2026,7,107.0),(2026,8,105.3),
        (2026,9,104.3),(2026,10,101.7),(2026,11,98.6),(2026,12,95.4),
    ],
    'ML': [
        (2025,9,114.6),(2025,10,111.8),(2025,11,109.5),(2025,12,108.0),
        (2026,1,106.3),(2026,2,103.8),(2026,3,100.3),(2026,4,96.4),
        (2026,5,92.8),(2026,6,89.1),(2026,7,85.9),(2026,8,83.5),
        (2026,9,81.3),(2026,10,78.7),(2026,11,75.7),(2026,12,72.6),
    ],
}

SCENARIO_COLORS = {'SC': '#8E44AD', 'CM': '#E74C3C', 'ML': '#E67E22'}
SCENARIO_LABELS_ES = {'SC': 'SILSO Curva Estándar', 'CM': 'SILSO Método Combinado', 'ML': 'SILSO McNish & Lincoln'}
SCENARIO_LABELS_EN = {'SC': 'SILSO Standard Curve', 'CM': 'SILSO Combined Method', 'ML': 'SILSO McNish & Lincoln'}
COLOR_OBS    = '#2C3E50'
COLOR_ARIMA  = '#3498DB'
COLOR_ARIMAX = '#E74C3C'
COLOR_GRID   = '#BDC3C7'
FORECAST_YEARS = [2025, 2026]
YEAR_TRAIN_END = 2024
RANDOM_SEED    = 42

# ================================================================
# MAIN LOGIC
# ================================================================
if not run_btn:
    st.markdown(
        f"<div class='info-card'>"
        f"{'⬅️ Carga los archivos de datos en la barra lateral y presiona <strong>Ejecutar predicción</strong> para comenzar.'  if ES else '⬅️ Upload data files in the sidebar and press <strong>Run Forecast</strong> to begin.'}"
        f"</div>",
        unsafe_allow_html=True
    )
    st.stop()

# --- Validate uploads ---
if not cme_file or not ssn_annual_file:
    st.error(t(
        "❌ Debes cargar al menos el archivo de CMEs y el de SSN anual.",
        "❌ You must upload at least the CME file and the annual SSN file."
    ))
    st.stop()

# ================================================================
# 1. LOAD CME DATA
# ================================================================
_progress = st.empty()
_progress.info(t("⏳ Cargando datos CME...", "⏳ Loading CME data..."))

df_cmes = None
_load_ok = True
try:
    df_cmes = pd.read_csv(cme_file, low_memory=False)
    df_cmes['Fecha'] = pd.to_datetime(df_cmes['Fecha'], errors='coerce')
    df_cmes[['Central','Ancho','Rapidez']] = (
        df_cmes[['Central','Ancho','Rapidez']].apply(pd.to_numeric, errors='coerce')
    )
    df_cmes['Year'] = df_cmes['Fecha'].dt.year
except Exception as e:
    _load_ok = False
    _progress.empty()
    # Re-read just the header to show detected columns
    try:
        cme_file.seek(0)
        _detected_cols = list(pd.read_csv(cme_file, nrows=0).columns)
        cme_file.seek(0)
    except Exception:
        _detected_cols = ['no se pudo leer']
    st.error(
        t(f"❌ Error al cargar CMEs: {e}",
          f"❌ Error loading CMEs: {e}") +
        f"\n\n**Columnas detectadas / Detected columns:** `{_detected_cols}`" +
        t("\n\nVerifica que el CSV tenga columnas: `Fecha`, `Rapidez`, `Ancho`, `Central`",
          "\n\nCheck that your CSV has columns: `Fecha`, `Rapidez`, `Ancho`, `Central`")
    )

if not _load_ok or df_cmes is None:
    _progress.empty()
    st.error(t("❌ No se pudieron cargar CMEs — revisa el archivo y vuelve a intentarlo.", "❌ Could not load CME data — check the file and try again."))
    # Attempt to stop Streamlit execution; if not running under Streamlit, ensure process exits
    try:
        st.stop()
    except Exception:
        pass
    import sys
    sys.exit(1)

df_filt = df_cmes[
    (df_cmes['Rapidez'] >= min_speed) &
    (df_cmes['Ancho']   >= min_width) &
    (df_cmes['Ancho']   <= max_width)
].copy()

conteo = (
    df_filt.groupby('Year').size()
    .rename('CMEs')
    .reindex(range(year_start, YEAR_TRAIN_END + 1), fill_value=0)
    .reset_index()
)
conteo.columns = ['Year', 'CMEs']

# ================================================================
# 2. LOAD ANNUAL SSN
# ================================================================
_progress.info(t("⏳ Cargando manchas solares anuales...", "⏳ Loading annual sunspot numbers..."))

df_sn_hist = None
_ssn_ok = True
try:
    df_sn_hist = pd.read_csv(
        ssn_annual_file, sep=r'\s+', header=None,
        usecols=[0,1], names=['Year','SSN']
    )
    df_sn_hist['Year'] = df_sn_hist['Year'].astype(int)
    df_sn_hist = df_sn_hist[
        (df_sn_hist['Year'] >= year_start) &
        (df_sn_hist['Year'] <= YEAR_TRAIN_END)
    ].copy()
except Exception as e:
    _ssn_ok = False
    _progress.empty()
    st.error(t(f"❌ Error al cargar SSN anual: {e}", f"❌ Error loading annual SSN: {e}"))

if not _ssn_ok or df_sn_hist is None:
    st.stop()

# ================================================================
# 3. MONTHLY SSN (optional)
# ================================================================
use_monthly = False
jan_aug_2025 = np.array([])

if ssn_monthly_file:
    try:
        df_sn_monthly = pd.read_csv(
            ssn_monthly_file, sep=r'\s+', header=None,
            usecols=[0,1,3], names=['Year','Month','SSN']
        )
        df_sn_monthly['Year']  = df_sn_monthly['Year'].astype(int)
        df_sn_monthly['Month'] = df_sn_monthly['Month'].astype(int)
        jan_aug_2025 = df_sn_monthly[
            (df_sn_monthly['Year'] == 2025) & (df_sn_monthly['Month'] <= 8)
        ]['SSN'].values
        use_monthly = len(jan_aug_2025) > 0
    except:
        use_monthly = False

# ================================================================
# 4. BUILD SSN SCENARIOS
# ================================================================
ssn_scenarios = {}
for scenario, monthly_data in SILSO_MONTHLY.items():
    annual_ssn = {}
    for yr in FORECAST_YEARS:
        silso_vals = [v for (y,m,v) in monthly_data if y == yr]
        if yr == 2025:
            if use_monthly:
                annual_ssn[yr] = float(np.mean(np.concatenate([jan_aug_2025, silso_vals])))
            else:
                hist_2025 = df_sn_hist[df_sn_hist['Year'] == 2025]['SSN'].values
                annual_ssn[yr] = float(hist_2025[0]) if len(hist_2025) > 0 else float(np.mean(silso_vals))
        else:
            annual_ssn[yr] = float(np.mean(silso_vals))
    ssn_scenarios[scenario] = annual_ssn

# ================================================================
# 5. MERGE & BUILD TRAINING SERIES
# ================================================================
df_merged = pd.merge(df_sn_hist, conteo, on='Year', how='inner')

missing = set(range(year_start, YEAR_TRAIN_END+1)) - set(df_merged['Year'])
if missing:
    st.warning(t(
        f"⚠️ Años sin datos en la serie combinada: {sorted(missing)}",
        f"⚠️ Years missing from combined series: {sorted(missing)}"
    ))

index_all = pd.to_datetime([str(y) for y in df_merged['Year']])
endog_all = pd.Series(df_merged['CMEs'].values, index=index_all)
exog_all  = pd.DataFrame(df_merged['SSN'].values, index=index_all, columns=['SSN'])

# ================================================================
# 6. STATIONARITY TEST
# ================================================================
np.random.seed(RANDOM_SEED)
adf_result = adfuller(endog_all, autolag='AIC')
pval_0     = adf_result[1]
stationary_0 = pval_0 < 0.05

if stationary_0:
    d_suggested = 0
else:
    diff1 = endog_all.diff().dropna()
    pval_1 = adfuller(diff1, autolag='AIC')[1]
    if pval_1 < 0.05:
        d_suggested = 1
    else:
        d_suggested = 2

max_lags_lb = min(10, len(endog_all) // 4)
lb = acorr_ljungbox(endog_all, lags=max_lags_lb, return_df=True)
lb_pval = lb['lb_pvalue'].iloc[-1]

# ================================================================
# 7. FIT MODELS
# ================================================================
_progress.info(t(
    "⏳ Ajustando modelos ARIMA / ARIMAX (puede tardar ~30 segundos)...",
    "⏳ Fitting ARIMA / ARIMAX models (may take ~30 seconds)..."
))

max_order = min(5, len(endog_all) // 3)

arima_auto = auto_arima(
    endog_all, seasonal=False, trace=False,
    error_action='ignore', suppress_warnings=True,
    stepwise=False, random_state=RANDOM_SEED,
    start_p=0, start_q=0, max_p=max_order, max_q=max_order,
    information_criterion='aic', d=d_suggested
)
orden_arima = arima_auto.order
modelo_arima    = SARIMAX(endog_all, order=orden_arima,
                          enforce_stationarity=False, enforce_invertibility=False)
resultado_arima = modelo_arima.fit(disp=False)

arimax_auto = auto_arima(
    endog_all, X=exog_all, seasonal=False, trace=False,
    error_action='ignore', suppress_warnings=True,
    stepwise=False, random_state=RANDOM_SEED,
    start_p=0, start_q=0, max_p=max_order, max_q=max_order,
    information_criterion='aic', d=None
)
orden_arimax = arimax_auto.order
modelo_arimax    = SARIMAX(endog_all, exog=exog_all, order=orden_arimax,
                           enforce_stationarity=True, enforce_invertibility=True)
resultado_arimax = modelo_arimax.fit(disp=False)

_progress.empty()

fitted_arima  = resultado_arima.fittedvalues
fitted_arimax = resultado_arimax.fittedvalues
y_all = endog_all.values

rmse_arima_is  = np.sqrt(mean_squared_error(y_all, fitted_arima.values))
mae_arima_is   = mean_absolute_error(y_all, fitted_arima.values)
r2_arima_is    = r2_score(y_all, fitted_arima.values)
rmse_arimax_is = np.sqrt(mean_squared_error(y_all, fitted_arimax.values))
mae_arimax_is  = mean_absolute_error(y_all, fitted_arimax.values)
r2_arimax_is   = r2_score(y_all, fitted_arimax.values)

# ================================================================
# 8. GENERATE FORECASTS
# ================================================================
forecast_index = pd.to_datetime([str(y) for y in FORECAST_YEARS])
results = {}

for scenario, ssn_dict in ssn_scenarios.items():
    exog_future = pd.DataFrame(
        {'SSN': [ssn_dict[y] for y in FORECAST_YEARS]},
        index=forecast_index
    )
    fc_arima     = resultado_arima.get_forecast(steps=2)
    fc_arima_m   = fc_arima.predicted_mean.values
    ci_arima     = fc_arima.conf_int(alpha=alpha_ci)
    ci_arima_lo  = np.clip(ci_arima.iloc[:,0].values, 0, None)
    ci_arima_hi  = ci_arima.iloc[:,1].values

    fc_arimax    = resultado_arimax.get_forecast(steps=2, exog=exog_future)
    fc_arimax_m  = fc_arimax.predicted_mean.values
    ci_arimax    = fc_arimax.conf_int(alpha=alpha_ci)
    ci_arimax_lo = np.clip(ci_arimax.iloc[:,0].values, 0, None)
    ci_arimax_hi = ci_arimax.iloc[:,1].values

    results[scenario] = {
        'ssn'        : [ssn_dict[y] for y in FORECAST_YEARS],
        'arima_mean' : fc_arima_m,
        'arima_lo'   : ci_arima_lo,
        'arima_hi'   : ci_arima_hi,
        'arimax_mean': fc_arimax_m,
        'arimax_lo'  : ci_arimax_lo,
        'arimax_hi'  : ci_arimax_hi,
    }

arima_mean = results['SC']['arima_mean']
arima_lo   = results['SC']['arima_lo']
arima_hi   = results['SC']['arima_hi']

# ================================================================
# DISPLAY — METRICS ROW
# ================================================================
st.markdown(f"## {t('📊 Resumen del Modelo', '📊 Model Summary')}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.markdown(f"""<div class='metric-box'>
        <div class='metric-label'>CMEs {t('filtradas','filtered')}</div>
        <div class='metric-value'>{len(df_filt)}</div>
        <div class='metric-sub'>{min_speed}+ km/s</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-box'>
        <div class='metric-label'>ARIMA {t('orden','order')}</div>
        <div class='metric-value'>{orden_arima}</div>
        <div class='metric-sub'>AIC {resultado_arima.aic:.1f}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-box'>
        <div class='metric-label'>ARIMAX {t('orden','order')}</div>
        <div class='metric-value'>{orden_arimax}</div>
        <div class='metric-sub'>AIC {resultado_arimax.aic:.1f}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-box'>
        <div class='metric-label'>R² ARIMA</div>
        <div class='metric-value'>{r2_arima_is:.3f}</div>
        <div class='metric-sub'>RMSE {rmse_arima_is:.1f}</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class='metric-box'>
        <div class='metric-label'>R² ARIMAX</div>
        <div class='metric-value'>{r2_arimax_is:.3f}</div>
        <div class='metric-sub'>RMSE {rmse_arimax_is:.1f}</div>
    </div>""", unsafe_allow_html=True)
with c6:
    adf_status = t("Estacionaria","Stationary") if stationary_0 else t("No estac.","Non-stationary")
    st.markdown(f"""<div class='metric-box'>
        <div class='metric-label'>ADF p-value</div>
        <div class='metric-value'>{pval_0:.3f}</div>
        <div class='metric-sub'>{adf_status}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================================================================
# MAIN FIGURE
# ================================================================
st.markdown(f"## {t('📈 Gráfica de Predicción', '📈 Forecast Plot')}")

rcParams['font.family']     = 'serif'
rcParams['font.size']       = 11
rcParams['axes.labelsize']  = 12
rcParams['figure.dpi']      = 120
SCENARIO_LABELS = SCENARIO_LABELS_ES if ES else SCENARIO_LABELS_EN

hist_years = df_merged['Year'].values
hist_cmes  = df_merged['CMEs'].values

fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                          gridspec_kw={'height_ratios': [3, 1.2]},
                          facecolor='#0d1117')
ax  = axes[0]
axs = axes[1]

for a in [ax, axs]:
    a.set_facecolor('#161b22')
    for spine in a.spines.values():
        spine.set_edgecolor('#30363d')

# --- Shading ---
ax.axvspan(2024.5, 2026.5, alpha=0.08, color='gold', zorder=0)

# --- In-sample ---
if show_insample:
    ax.plot(hist_years, fitted_arima.values,
            linestyle='--', linewidth=1.4, color=COLOR_ARIMA, alpha=0.55, zorder=2,
            label=f'ARIMA{orden_arima} {t("ajustado","fitted")}  (R²={r2_arima_is:.3f}, RMSE={rmse_arima_is:.1f})')
    ax.plot(hist_years, fitted_arimax.values,
            linestyle='-.', linewidth=1.4, color=COLOR_ARIMAX, alpha=0.55, zorder=2,
            label=f'ARIMAX{orden_arimax} {t("ajustado","fitted")} (R²={r2_arimax_is:.3f}, RMSE={rmse_arimax_is:.1f})')

# --- ARIMA forecast ---
ax.fill_between(FORECAST_YEARS, arima_lo, arima_hi,
                alpha=0.15, color=COLOR_ARIMA, zorder=1,
                label=f'ARIMA {int((1-alpha_ci)*100)}% CI')
ax.plot(FORECAST_YEARS, arima_mean,
        linestyle='--', linewidth=2.4, color=COLOR_ARIMA,
        marker='D', markersize=7, markeredgecolor='white', markeredgewidth=0.8,
        zorder=5, label=f'ARIMA{orden_arima} {t("predicción","forecast")}')

# --- ARIMAX forecasts ---
for scenario, res in results.items():
    color = SCENARIO_COLORS[scenario]
    ax.fill_between(FORECAST_YEARS, res['arimax_lo'], res['arimax_hi'],
                    alpha=0.12, color=color, zorder=1)
    ax.plot(FORECAST_YEARS, res['arimax_mean'],
            linestyle='-.', linewidth=2.2, color=color,
            marker='s', markersize=7,
            markeredgecolor='white', markeredgewidth=0.8, zorder=5,
            label=f"ARIMAX{orden_arimax} — {SCENARIO_LABELS[scenario]}")

# --- Observed ---
ax.plot(hist_years, hist_cmes,
        marker='o', linestyle='-', linewidth=2.0, markersize=6,
        color='#58a6ff', zorder=6, markeredgewidth=0.8, markeredgecolor='white',
        label=t(' CMEs observadas', ' Observed CMEs'))

ax.axvline(x=2024.5, color='#f0a500', linestyle=':', linewidth=1.8, alpha=0.7, zorder=7,
           label=t('Límite de entrenamiento','Training boundary'))

ax.set_ylabel(t('Conteo de CMEs (eventos/año)', 'CME Count (events/year)'),
              fontsize=12, color='#c9d1d9')
ax.set_xlim(year_start-0.5, 2027.2)
ax.set_ylim(bottom=0)
ax.set_xticks(range(year_start, 2027, 2))
ax.tick_params(colors='#8b949e')
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#30363d')
ax.set_axisbelow(True)
legend = ax.legend(loc='upper left', frameon=True, framealpha=0.85,
                   edgecolor='#30363d', facecolor='#161b22',
                   fontsize=8.5, labelspacing=0.4,
                   title=t("Comparación de modelos","Model Comparison"), title_fontsize=9,
                   labelcolor='#c9d1d9')

# --- Inset ---
ax_inset = inset_axes(ax, width='48%', height='52%', loc='upper center',
                      bbox_to_anchor=(0.10, 0.0, 0.9, 1.0),
                      bbox_transform=ax.transAxes, borderpad=0)
ax_inset.set_facecolor('#1c2128')
for spine in ax_inset.spines.values():
    spine.set_edgecolor('#58a6ff'); spine.set_linewidth(0.8)

N_TAIL = 2
tail_years = hist_years[-N_TAIL:]
tail_cmes  = hist_cmes[-N_TAIL:]
ax_inset.plot(tail_years, tail_cmes, marker='o', linestyle='-',
              linewidth=1.8, markersize=5, color='#58a6ff', zorder=6,
              markeredgewidth=0.6, markeredgecolor='white')

for scenario, res in results.items():
    color = SCENARIO_COLORS[scenario]
    ax_inset.fill_between(FORECAST_YEARS, res['arimax_lo'], res['arimax_hi'],
                          alpha=0.15, color=color, zorder=1)
    ax_inset.plot(FORECAST_YEARS, res['arimax_mean'],
                  linestyle='-.', linewidth=2.0, color=color,
                  marker='s', markersize=7, markeredgecolor='white', markeredgewidth=0.8, zorder=5)

ax_inset.axvline(x=2024.5, color='#f0a500', linestyle=':', linewidth=1.4, alpha=0.6, zorder=7)
ax_inset.axvspan(2024.5, 2026.7, alpha=0.06, color='gold', zorder=0)

all_fc_vals = (
    [v for res in results.values()
     for v in list(res['arimax_mean']) + list(res['arimax_lo']) + list(res['arimax_hi'])]
    + list(tail_cmes)
)
y_pad = (max(all_fc_vals) - min(all_fc_vals)) * 0.05
inset_ylo = max(0, min(all_fc_vals) - y_pad)
inset_yhi = max(all_fc_vals) + y_pad * 1.5
ax_inset.set_xlim(2023-0.3, 2026+0.8)
ax_inset.set_ylim(inset_ylo, inset_yhi)
ax_inset.set_xticks([2023, 2024] + FORECAST_YEARS)
ax_inset.tick_params(axis='both', labelsize=7.5, colors='#8b949e')
ax_inset.grid(True, alpha=0.2, linestyle='-', linewidth=0.4, color='#30363d')
ax_inset.set_axisbelow(True)
ax_inset.set_title(t('Zoom predicción 2025–2026','Forecast zoom 2025–2026'),
                   fontsize=8, color='#8b949e', pad=3)

# --- SSN panel ---
axs.fill_between(hist_years, 0, df_merged['SSN'].values,
                 color='#58a6ff', alpha=0.1, zorder=1)
axs.plot(hist_years, df_merged['SSN'].values,
         color='#8b949e', linewidth=1.6, linestyle='-',
         marker='o', markersize=3.5, markeredgewidth=0.5, markeredgecolor='white',
         label=t(' SSN observado',' Observed SSN'), zorder=3)

for scenario, ssn_dict in ssn_scenarios.items():
    color  = SCENARIO_COLORS[scenario]
    ssn_fc = [ssn_dict[y] for y in FORECAST_YEARS]
    axs.plot([hist_years[-1], FORECAST_YEARS[0]],
             [df_merged['SSN'].values[-1], ssn_fc[0]],
             linestyle=':', linewidth=1.0, color=color, alpha=0.45, zorder=2)
    axs.plot(FORECAST_YEARS, ssn_fc,
             linestyle='-.', linewidth=1.8, color=color,
             marker='s', markersize=5, markeredgewidth=0.6, markeredgecolor='white',
             label=SCENARIO_LABELS[scenario], zorder=4)

axs.axvspan(2024.5, 2026.5, alpha=0.06, color='gold', zorder=0)
axs.axvline(x=2024.5, color='#f0a500', linestyle=':', linewidth=1.8, alpha=0.6, zorder=5)
axs.set_ylabel(t('Manchas solares (SSN)', 'Sunspot Number (SSN)'), fontsize=11, color='#c9d1d9')
axs.set_xlabel(t('Año', 'Year'), fontsize=12, color='#c9d1d9')
axs.set_xlim(year_start-0.5, 2027.2)
axs.set_ylim(bottom=0)
axs.set_xticks(range(year_start, 2027, 2))
axs.tick_params(colors='#8b949e')
axs.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#30363d')
axs.set_axisbelow(True)
legend_ssn = axs.legend(loc='upper center', frameon=True, fancybox=False,
                         framealpha=0.8, edgecolor='#30363d', facecolor='#161b22',
                         borderpad=0.6, labelspacing=0.3, fontsize=8.5, labelcolor='#c9d1d9')

plt.tight_layout(h_pad=1.8)
st.pyplot(fig, use_container_width=True)

# Download figure
buf = io.BytesIO()
fig.savefig(buf, format='pdf', dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
buf.seek(0)
st.download_button(
    label=t("⬇️ Descargar figura (PDF)", "⬇️ Download figure (PDF)"),
    data=buf,
    file_name="forecast_cme_2025_2026.pdf",
    mime="application/pdf"
)

# ================================================================
# RESULTS TABLE
# ================================================================
st.markdown(f"## {t('📋 Resultados Numéricos', '📋 Numerical Results')}")

rows = []
ci_pct = int((1-alpha_ci)*100)
for scenario, res in results.items():
    for i, yr in enumerate(FORECAST_YEARS):
        rows.append({
            t('Año','Year')              : yr,
            t('Escenario SILSO','SILSO Scenario') : scenario,
            'SSN'                        : round(res['ssn'][i], 1),
            f'ARIMA {t("predicción","forecast")}': round(res['arima_mean'][i], 2),
            f'ARIMA CI {ci_pct}% {t("inf","low")}': round(res['arima_lo'][i], 2),
            f'ARIMA CI {ci_pct}% {t("sup","high")}': round(res['arima_hi'][i], 2),
            f'ARIMAX {t("predicción","forecast")}': round(res['arimax_mean'][i], 2),
            f'ARIMAX CI {ci_pct}% {t("inf","low")}': round(res['arimax_lo'][i], 2),
            f'ARIMAX CI {ci_pct}% {t("sup","high")}': round(res['arimax_hi'][i], 2),
        })

df_out = pd.DataFrame(rows)
st.dataframe(df_out, use_container_width=True, hide_index=True)

csv_out = df_out.to_csv(index=False).encode('utf-8')
st.download_button(
    label=t("⬇️ Descargar resultados (CSV)", "⬇️ Download results (CSV)"),
    data=csv_out,
    file_name="forecast_cme_2025_2026.csv",
    mime="text/csv"
)

# ================================================================
# DIAGNOSTICS
# ================================================================
with st.expander(t("🔬 Diagnósticos estadísticos", "🔬 Statistical diagnostics"), expanded=False):
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"**{t('Prueba ADF (estacionariedad)','ADF Test (stationarity)')}**")
        st.markdown(f"""<div class='info-card'>
            p-value = <code>{pval_0:.4f}</code> → 
            {'✅ ' + t('Estacionaria','Stationary') if stationary_0 else '⚠️ ' + t('No estacionaria','Non-stationary')} 
            (d = {d_suggested})<br>
            Ljung-Box p = <code>{lb_pval:.4f}</code> → 
            {'✅ ' + t('Autocorrelación significativa','Significant autocorrelation') if lb_pval < 0.05 else '⚠️ ' + t('Ruido blanco','White noise')}
        </div>""", unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"**{t('Métricas en muestra','In-sample metrics')}**")
        df_metrics = pd.DataFrame({
            t('Métrica','Metric'): ['AIC', 'RMSE', 'MAE', 'R²'],
            'ARIMA': [f"{resultado_arima.aic:.2f}", f"{rmse_arima_is:.3f}", f"{mae_arima_is:.3f}", f"{r2_arima_is:.4f}"],
            'ARIMAX': [f"{resultado_arimax.aic:.2f}", f"{rmse_arimax_is:.3f}", f"{mae_arimax_is:.3f}", f"{r2_arimax_is:.4f}"],
        })
        st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    # Residual plot
    st.markdown(f"**{t('Residuales del modelo ARIMAX','ARIMAX Model Residuals')}**")
    fig_res, ax_res = plt.subplots(figsize=(10, 3), facecolor='#0d1117')
    ax_res.set_facecolor('#161b22')
    residuals = resultado_arimax.resid
    ax_res.plot(df_merged['Year'].values, residuals.values,
                color='#E74C3C', linewidth=1.2, marker='o', markersize=3)
    ax_res.axhline(0, color='#f0a500', linewidth=1.0, linestyle='--')
    ax_res.set_xlabel(t('Año','Year'), color='#8b949e')
    ax_res.set_ylabel(t('Residual','Residual'), color='#8b949e')
    ax_res.tick_params(colors='#8b949e')
    for spine in ax_res.spines.values():
        spine.set_edgecolor('#30363d')
    ax_res.grid(True, alpha=0.2, color='#30363d')
    plt.tight_layout()
    st.pyplot(fig_res, use_container_width=True)

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    f"<div style='text-align:center; color:#8b949e; font-size:0.85rem; padding: 0.5rem;'>"
    f"{'Aplicación desarrollada para la difusión científica de resultados sobre predicción de CMEs solares · Ciclo Solar 25'  if ES else 'Application developed for scientific dissemination of solar CME prediction results · Solar Cycle 25'}"
    f"<br><span style='color:#30363d;'>ARIMA · ARIMAX · SILSO · LASCO/CDAW</span>"
    f"</div>",
    unsafe_allow_html=True
)
