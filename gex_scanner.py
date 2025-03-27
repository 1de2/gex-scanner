import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz
import json
from typing import Dict, Optional, List

# Configuración de Zona Horaria
NY_TZ = pytz.timezone("America/New_York")

# Configuración de Tradier (si se usa la API de Tradier en el futuro)
TRADIER_API_KEY = "TU_API_KEY_DE_TRADIER"
TRADIER_URL = "https://api.tradier.com/v1/markets/options/chains"

# Función para obtener datos de opciones desde Tradier
def fetch_option_chain_tradier(ticker: str, expiry: datetime) -> pd.DataFrame:
    """Obtiene la cadena de opciones desde Tradier"""
    try:
        headers = {'Authorization': f'Bearer {TRADIER_API_KEY}', 'Accept': 'application/json'}
        params = {'symbol': ticker, 'expiration': expiry.strftime('%Y-%m-%d')}
        response = requests.get(TRADIER_URL, headers=headers, params=params)
        
        if response.status_code != 200:
            st.error(f"Error al obtener los datos de opciones de Tradier: {response.status_code}")
            return pd.DataFrame()

        data = response.json()
        options = data['options']['option']
        
        # Crear DataFrame con las opciones
        df = pd.DataFrame(options)
        return df

    except Exception as e:
        st.error(f"Error al obtener los datos de opciones: {str(e)}")
        return pd.DataFrame()

# Función para obtener la cadena de opciones desde Yahoo Finance
def fetch_option_chain_yf(ticker: str, expiry: datetime) -> pd.DataFrame:
    """Obtiene la cadena de opciones desde Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        expiry_str = expiry.strftime("%Y-%m-%d")
        available_dates = [pd.to_datetime(d) for d in stock.options]
        
        # Asegurarse de que la fecha de expiración esté disponible
        if expiry_str not in [d.strftime("%Y-%m-%d") for d in available_dates]:
            st.warning(f"Usando la fecha más cercana disponible: {available_dates[0]}")
            expiry_str = available_dates[0].strftime("%Y-%m-%d")

        # Obtener la cadena de opciones para la fecha de expiración seleccionada
        chain = stock.option_chain(expiry_str)
        return pd.concat([chain.calls, chain.puts], ignore_index=True)

    except Exception as e:
        st.error(f"Error al obtener la cadena de opciones de Yahoo Finance: {str(e)}")
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

# Función para enviar un análisis manual a Telegram
def send_telegram_alert(message: str):
    """Envío de alerta a través de Telegram (deberías configurar tu bot y token)"""
    token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url)

# Lógica de ejecución del análisis
def run_analysis(ticker: str, expiry_date: datetime, mode: str = "manual"):
    st.write(f"Ejecutando análisis {mode} para {ticker}...")

    # Obtener datos de la cadena de opciones (puedes elegir Tradier o Yahoo Finance)
    chain = fetch_option_chain_yf(ticker, expiry_date)

    # Obtener datos del spot price (último precio de mercado)
    stock_data = yf.Ticker(ticker).history(period="1d")
    spot_price = stock_data["Close"].iloc[-1]

    # Calcular el GEX
    total_gex = 0
    for _, row in chain.iterrows():
        # Asegurarnos de que la fecha de expiración esté presente
        if pd.notna(row['expiry']):
            T = (pd.to_datetime(row["expiry"]) - get_ny_time()).days / 365.25
            gex = calculate_gex(spot_price, row["strike"], T, row["impliedVolatility"], row["option_type"], row["openInterest"])
            total_gex += gex

    st.write(f"Total GEX: {total_gex}")

    # Enviar alerta por Telegram
    if mode == "manual":
        send_telegram_alert(f"Análisis manual para {ticker}: Total GEX: {total_gex}")

# Función principal para Streamlit
def main():
    st.set_page_config(page_title="🚨 GEX Scanner Pro", layout="wide", page_icon="📊")
    st.title("📊 Gamma Exposure Scanner (SPY/QQQ)")

    # Configuración en la barra lateral
    ticker = st.sidebar.selectbox("Seleccionar Activo:", ["SPY", "QQQ", "IWM", "AAPL", "TSLA"])
    expiry_date = st.sidebar.date_input("Fecha de Expiración:", min_value=datetime.today())

    # Llamada para ejecutar análisis manual
    if st.sidebar.button("🔍 Ejecutar Análisis Manual"):
        run_analysis(ticker, expiry_date, mode="manual")

    # Verificar si estamos en el horario adecuado para alertas automáticas
    alert_time = check_alert_time()
    if alert_time:
        st.write(f"¡Es hora de ejecutar el análisis {alert_time}!")
        run_analysis(ticker, expiry_date, mode="automatic")

if __name__ == "__main__":
    main()
