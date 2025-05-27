import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import scipy.optimize as sco

import streamlit as st

st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: #90ee90;
        }
        .stApp {
            background-color: #000000;
            color: #90ee90;
        }

        /* Textos y encabezados */
        h1, h2, h3, h4, h5, h6, p, span, div {
            color: #90ee90 !important;
        }

        /* Subheaders y textos específicos */
        .css-18ni7ap.e8zbici2,
        .css-1v0mbdj.edgvbvh3,
        .css-1cpxqw2.edgvbvh3,
        .st-cm, .st-cn,
        .stMarkdown {
            color: white !important;
        }

        /* Métricas */
        .stMetric {
            color: #90ee90 !important;
        }

        /* Tablas */
        .css-qrbaxs.e16nr0p34 {
            background-color: #111111 !important;
            color: #90ee90 !important;
        }

        /* Cuadros de texto y entradas */
        .stTextInput>div>div>input,
        .stDateInput>div>input,
        .stNumberInput>div>input,
        .stSelectbox>div>div,
        .stMultiSelect>div>div {
            background-color: #111111 !important;
            color: #90ee90 !important;
            border: 1px solid #90ee90 !important;
        }

        /* Botones */
        .stButton>button {
            background-color: #111111;
            color: #90ee90;
            border: 1px solid #90ee90;
        }
        .stButton>button:hover {
            background-color: #90ee90;
            color: #000000;
        }

        /* Éxito, advertencia, error */
        .stAlert > div {
            background-color: #111111 !important;
            color: #90ee90 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Logo arriba a la derecha
st.markdown("""
    <div style="display: flex; justify-content: flex-end;">
        <img src="https://bing.com/th/id/BCO.675b93d9-05c5-4513-af60-cdd6f1cccf55.png" 
             alt="Logo ANA Seguros" width="200" style="margin-top: -60px; margin-right: 10px;">
    </div>
""", unsafe_allow_html=True)

plt.style.use('fivethirtyeight')

st.title("Análisis de Portafolio y Frontera Eficiente")
#st.markdown("Esta app permite analizar un portafolio de activos usando la simulación de Monte Carlo.")

# Entrada de TICKERS
tickers_input = st.text_input("Introduce los tickers separados por comas (ej: AAPL, MSFT, GOOGL):")
if tickers_input:
    activos = [t.strip().upper() for t in tickers_input.split(",")]

    # Descargar datos
    data = pd.DataFrame()
    for activo in activos:
        stock_data = yf.Ticker(activo).history(period="5y")
        if not stock_data.empty:
            data[activo] = stock_data["Close"]
        else:
            st.warning(f"No se encontraron datos para {activo}.")

    if not data.empty:
        data.index = pd.to_datetime(data.index).date

        st.subheader("Precios históricos (últimos 5 años)")
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        data.plot(ax=ax1)
        st.pyplot(fig1)

        rendimientos = data.pct_change().dropna()

        st.subheader("Rendimientos históricos")
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        rendimientos.plot(ax=ax2)
        st.pyplot(fig2)

        # Simulación de portafolios
        libor = 0.043
        rendimientos_esperados = rendimientos.mean()
        matriz_cov = rendimientos.cov()
        num_portafolios = 25000

        def portafolio_resultado_anual(pesos):
            rendimiento = np.sum(rendimientos_esperados * pesos) * 252
            std = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos))) * np.sqrt(252)
            return std, rendimiento

        def portafolios_aleatorios():
            resultados = np.zeros((3, num_portafolios))
            pesos_asignados = []
            for i in range(num_portafolios):
                pesos = np.random.random(len(rendimientos_esperados))
                pesos /= np.sum(pesos)
                pesos_asignados.append(pesos)
                std, rendimiento = portafolio_resultado_anual(pesos)
                resultados[0, i] = std
                resultados[1, i] = rendimiento
                resultados[2, i] = (rendimiento - libor) / std
            return resultados, pesos_asignados

        resultados, pesos = portafolios_aleatorios()
        max_sharpe_idx = np.argmax(resultados[2])
        min_vol_idx = np.argmin(resultados[0])

        sdp, rp = resultados[0, max_sharpe_idx], resultados[1, max_sharpe_idx]
        sdp_min, rp_min = resultados[0, min_vol_idx], resultados[1, min_vol_idx]

        max_sharpe_asignacion = pd.Series(pesos[max_sharpe_idx], index=data.columns)
        min_vol_asignacion = pd.Series(pesos[min_vol_idx], index=data.columns)

        st.subheader("Asignación del Portafolio Óptimo (Mayor Sharpe)")
        st.write((max_sharpe_asignacion * 100).round(2))

        st.subheader("Asignación del Portafolio de Menor Volatilidad")
        st.write((min_vol_asignacion * 100).round(2))

        # Gráfico Frontera Eficiente
        st.subheader("Frontera Eficiente (Simulada)")
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        scatter = ax3.scatter(resultados[0], resultados[1], c=resultados[2], cmap='YlGnBu', s=10)
        ax3.scatter(sdp, rp, marker='*', color='r', s=500, label='Máximo Sharpe')
        ax3.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Mínima Volatilidad')
        ax3.set_xlabel('Volatilidad anualizada')
        ax3.set_ylabel('Rendimiento anualizado')
        ax3.legend()
        fig3.colorbar(scatter, ax=ax3, label='Sharpe Ratio')
        st.pyplot(fig3)

        # Cálculo de VaR
        z = 1.65
        VaR_95 = z * sdp
        st.metric("VaR al 95% (1 día)", f"{VaR_95*100:.2f}%")

        # Evaluación de portafolio en un periodo
        st.subheader("Evaluación del Portafolio Óptimo")
        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio = st.date_input("Fecha de inicio", value=pd.to_datetime(data.index[0]))
        with col2:
            fecha_fin = st.date_input("Fecha de fin", value=pd.to_datetime(data.index[-1]))

        if fecha_inicio < fecha_fin:
            precios_periodo = pd.DataFrame()
            for activo in activos:
                hist = yf.Ticker(activo).history(start=fecha_inicio, end=fecha_fin)
                if not hist.empty:
                    precios_periodo[activo] = hist["Close"]

            if not precios_periodo.empty:
                rendimientos_periodo = precios_periodo.iloc[-1] / precios_periodo.iloc[0] - 1
                pesos = max_sharpe_asignacion
                rendimiento_portafolio = np.dot(rendimientos_periodo, pesos)
                st.success(f"Rendimiento del portafolio entre {fecha_inicio} y {fecha_fin}: {rendimiento_portafolio*100:.2f}%")
        else:
            st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
