# CME-Forecasting-App
Solar CME Forecast App — Predicción de CMEs Solares

Aplicación interactiva para la predicción de Eyecciones de Masa Coronal (CMEs) de alta velocidad durante 2025–2026 usando modelos ARIMA y ARIMAX, en el contexto del Ciclo Solar 25.

Interactive app for forecasting high-speed Coronal Mass Ejections (CMEs) during 2025–2026 using ARIMA and ARIMAX models, in the context of Solar Cycle 25.
Demo

URL disponible tras el deploy en Streamlit Community Cloud.
Descripción | Description

ES: Esta aplicación implementa los modelos estadísticos desarrollados en nuestro artículo científico (en preparación). Permite explorar de forma interactiva cómo cambian las predicciones al ajustar los filtros de velocidad y ancho angular de los CMEs, y compara los resultados de los modelos ARIMA y ARIMAX bajo tres escenarios de actividad solar provistos por SILSO para el Ciclo Solar 25.

EN: This application implements the statistical models developed in our scientific paper (in preparation). It allows users to interactively explore how forecasts change when adjusting CME speed and angular width filters, and compares ARIMA and ARIMAX model results under three solar activity scenarios provided by SILSO for Solar Cycle 25.
Datos | Data

Los datos utilizados en esta aplicación están públicamente disponibles en el repositorio del artículo:

JuanAnarch7/Forecasting-Coronal-Mass-Ejection-Occurrence-Rates-Using-ARIMA-and-ARIMAX-Models

Allí encontrará:

    El archivo datos_procesados_*.csv con los CMEs filtrados

    Los archivos SN_y_tot_V2.0.txt y SN_m_tot_V2.0.txt (SSN anual y mensual — SILSO)

    Los criterios de preprocesamiento aplicados al catálogo LASCO/CDAW

    El código original de implementación de los modelos ARIMA y ARIMAX

Fuentes primarias:

    Catálogo LASCO/CDAW: cdaw.gsfc.nasa.gov/CME_list

    SILSO — WDC-SILSO, Royal Observatory of Belgium: sidc.be/SILSO

Funcionalidades | Features
Función	Descripción
Filtros interactivos	Velocidad mínima, ancho angular mínimo y máximo, año de inicio
ARIMA + ARIMAX	Ajuste automático de orden vía auto_arima (AIC)
Tres escenarios SILSO	Standard Curve · Combined Method · McNish & Lincoln
Métricas en muestra	AIC, RMSE, MAE, R² para ambos modelos
Diagnósticos	Prueba ADF, Ljung-Box, gráfica de residuales
Bilingüe	Español / English
Descarga	Figura en PDF + resultados en CSV
Instalación local | Local Setup
bash

git clone https://github.com/JuanAnarch7/cme-forecast-app.git
cd cme-forecast-app
pip install -r requirements.txt
streamlit run Applicacion.py

La app abrirá en su navegador en http://localhost:8501.
