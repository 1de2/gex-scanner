import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz

# Configuración de Zona Horaria
NY_TZ = pytz.timezone("America/New_York")

# Función para obtener los datos de la cadena de opciones
def fetch_option_chain(ticker: str, expiry: datetime) -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        expiry_str = expiry.strftime("%Y-%m-%d")
        available_dates = [pd.to_datetime(d) for d in stock.options]
        
        if expiry_str not in [d.strftime("%Y-%m-%d") for d in available_dates]:
            st.warning(f"Usando la fecha más cercana disponible: {available_dates[0]}")
            expiry_str = available_dates[0].strftime("%Y-%m-%d")

        # Obtener la cadena de opciones para la fecha de expiración seleccionada
        chain = stock.option_chain(expiry_str)
        return pd.concat([chain.calls, chain.puts], ignore_index=True)

    except Exception as e:
        st.error(f"Error al obtener los datos de opciones: {str(e)}")
        return pd.DataFrame()

# Función para calcular el GEX (Gamma Exposure)
def calculate_gex(S: float, K: float, T: float, iv: float, option_type: str, open_interest: float) -> float:
    """Calcula el Gamma Exposure (GEX) de una opción"""
    try:
        d1 = (np.log(S / K) + (0.01 + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
        gex = gamma * open_interest * (S ** 2) * 0.01 / 1e6
        return gex
    except Exception as e:
        st.error(f"Error calculando GEX: {str(e)}")
        return 0.0

# Función para obtener la hora actual en Nueva York
def get_ny_time():
    ny_time = datetime.now(NY_TZ)
    return ny_time

# Función para verificar si estamos a 5 minutos antes de pre-market o apertura
def check_alert_time():
    current_time = get_ny_time().time()
    pre_market_start = datetime.strptime("04:00", "%H:%M").time()  # 04:00 AM NY time
    market_open = datetime.strptime("09:30", "%H:%M").time()  # 09:30 AM NY time

    if (current_time >= (datetime.combine(datetime.today(), pre_market_start) - timedelta(minutes=5)).time() and 
        current_time < pre_market_start):
        return "pre-market"
    elif (current_time >= (datetime.combine(datetime.today(), market_open) - timedelta(minutes=5)).time() and
          current_time < market_open):
        return "market-open"
    return None

# Función para ejecutar el análisis de GEX
def run_analysis(ticker: str, expiry_date: datetime):
    st.write(f"Ejecutando análisis para {ticker}...")

    # Obtener datos de la cadena de opciones
    chain = fetch_option_chain(ticker, expiry_date)

    # Obtener datos del spot price (último precio de mercado)
    stock_data = yf.Ticker(ticker).history(period="1d")
    spot_price = stock_data["Close"].iloc[-1]

    # Calcular el GEX
    total_gex = 0
    top_calls = []
    top_puts = []

    for _, row in chain.iterrows():
        # Asegurarnos de que la fecha de expiración esté presente
        if pd.notna(row['expiry']):
            T = (pd.to_datetime(row["expiry"]) - get_ny_time()).days / 365.25
            gex = calculate_gex(spot_price, row["strike"], T, row["impliedVolatility"], row["option_type"], row["openInterest"])
            total_gex += gex
            if row["option_type"] == "call":
                top_calls.append((row["strike"], gex))
            else:
                top_puts.append((row["strike"], gex))

    top_calls.sort(key=lambda x: x[1], reverse=True)
    top_puts.sort(key=lambda x: x[1])

    st.write(f"Total GEX: {total_gex}")

    # Mostrar los 3 strikes más importantes para calls y puts
    st.write("Top 3 Calls por GEX:")
    for strike, gex in top_calls[:3]:
        st.write(f"Strike: {strike}, GEX ($M): {gex}")

    st.write("Top 3 Puts por GEX:")
    for strike, gex in top_puts[:3]:
        st.write(f"Strike: {strike}, GEX ($M): {gex}")

# Función principal para Streamlit
def main():
    st.set_page_config(page_title="🚨 GEX Dashboard", layout="wide", page_icon="📊")
    st.title("📊 Gamma Exposure Dashboard")

    # Configuración en la barra lateral
    ticker = st.sidebar.selectbox("Seleccionar Activo:", ["SPY", "QQQ", "AAPL", "TSLA"])
    expiry_date = st.sidebar.date_input("Fecha de Expiración:", min_value=datetime.today())

    # Llamada para ejecutar análisis
    if st.sidebar.button("🔍 Ejecutar Análisis"):
        run_analysis(ticker, expiry_date)

    # Verificar si estamos en el horario adecuado para alertas automáticas
    alert_time = check_alert_time()
    if alert_time:
        st.write(f"¡Es hora de ejecutar el análisis {alert_time}!")
        run_analysis(ticker, expiry_date)

if __name__ == "__main__":
    main()
