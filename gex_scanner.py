# gex_scanner.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz
import json
import argparse

# Configuración
NY_TZ = pytz.timezone("America/New_York")
TODAY = datetime.now(NY_TZ).date()
MAX_MANUAL_SCANS = 1  # Límite de análisis manuales

class GEXScanner:
    def __init__(self):
        self.counter_file = "scan_counter.json"
        self.load_counter()

    def load_counter(self):
        """Carga el contador de escaneos desde JSON"""
        try:
            with open(self.counter_file, "r") as f:
                data = json.load(f)
                if data["date"] == str(TODAY):
                    self.counter = data
                    return
        except:
            pass
        self.counter = {"date": str(TODAY), "manual_scans": 0}

    def save_counter(self):
        """Guarda el contador en JSON"""
        with open(self.counter_file, "w") as f:
            json.dump(self.counter, f)

    @staticmethod
    def calculate_greeks(S, K, T, iv, r=0.01, option_type="call"):
        """Calcula delta y gamma para una opción"""
        d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
        return delta, gamma

    def fetch_option_chain(self, ticker, expiry):
        """Obtiene la cadena de opciones"""
        try:
            chain = yf.Ticker(ticker).option_chain(expiry.strftime("%Y-%m-%d"))
            calls = chain.calls.assign(option_type="call")
            puts = chain.puts.assign(option_type="put")
            return pd.concat([calls, puts])
        except Exception as e:
            st.error(f"Error obteniendo datos: {str(e)}")
            return pd.DataFrame()

    def scan(self, ticker="SPY", expiry_date=None, auto_mode=False):
        """Ejecuta el análisis GEX"""
        if expiry_date is None:
            expiry_date = TODAY + timedelta(days=3)
        
        S = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        chain = self.fetch_option_chain(ticker, expiry_date)
        
        if chain.empty:
            return None
        
        # Calcular griegas
        chain["T"] = (pd.to_datetime(chain["expiry"]) - pd.Timestamp.now()).dt.days / 365
        chain[["delta", "gamma"]] = chain.apply(
            lambda row: self.calculate_greeks(
                S, row["strike"], row["T"], row["impliedVolatility"], row["option_type"]
            ),
            axis=1, result_type="expand"
        )
        
        # Calcular GEX
        chain["gex"] = chain["gamma"] * chain["openInterest"] * (S ** 2) * 0.01
        
        # Actualizar contador si es manual
        if not auto_mode:
            if self.counter["manual_scans"] >= MAX_MANUAL_SCANS:
                return {"error": "Límite diario alcanzado"}
            self.counter["manual_scans"] += 1
            self.save_counter()
        
        return {
            "timestamp": datetime.now(NY_TZ).strftime("%Y-%m-%d %H:%M"),
            "ticker": ticker,
            "spot_price": round(S, 2),
            "total_gex": "${:,.2f}M".format(chain["gex"].sum() / 1e6),
            "top_calls": chain[chain["option_type"] == "call"]
                         .nlargest(3, "gex")[["strike", "gex"]]
                         .assign(gex=lambda x: x["gex"].apply(lambda v: "${:,.1f}M".format(v/1e6)))
                         .to_dict("records"),
            "top_puts": chain[chain["option_type"] == "put"]
                        .nsmallest(3, "gex")[["strike", "gex"]]
                        .assign(gex=lambda x: x["gex"].apply(lambda v: "-${:,.1f}M".format(abs(v)/1e6)))
                        .to_dict("records")
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-scan", action="store_true", help="Modo automático para GitHub Actions")
    args = parser.parse_args()

    scanner = GEXScanner()
    
    if args.auto_scan:
        # Modo automático para GitHub Actions
        results = scanner.scan(auto_mode=True)
        with open("scan_results.json", "w") as f:
            json.dump(results, f)
        print("Análisis automático completado")
    else:
        # Interfaz Streamlit
        st.set_page_config(page_title="🚨 Scanner GEX Pro", layout="wide")
        st.title("📊 Gamma Exposure Scanner (SPY/QQQ)")

        # Sidebar
        ticker = st.sidebar.selectbox("Seleccionar Activo:", ["SPY", "QQQ"])
        expiry_date = st.sidebar.date_input("Fecha de Expiración:", TODAY + timedelta(days=3))

        # Botón de análisis manual
        if st.sidebar.button("🔍 Ejecutar Análisis Manual", help="Límite: 1 por día"):
            with st.spinner("Calculando GEX..."):
                results = scanner.scan(ticker, expiry_date)
                if results and "error" not in results:
                    st.success("✅ Análisis completado!")
                    st.json(results)
                elif "error" in results:
                    st.error(f"❌ {results['error']}")

        # Mostrar contador
        st.sidebar.markdown(f"""
        **📅 Límites Diarios:**  
        - Automáticos: 2 (7 AM & 9:30 AM NY)  
        - Manuales: {scanner.counter['manual_scans']}/{MAX_MANUAL_SCANS}  
        """)

        # Sección de información
        st.markdown("---")
        st.markdown("""
        ### 📌 Instrucciones:
        1. Usa el botón **"Ejecutar Análisis Manual"** para escanear GEX en cualquier momento (límite 1/día).
        2. Los análisis automáticos se ejecutan:
           - **⏰ 7:00 AM NY (Pre-Market)**
           - **🕤 9:30 AM NY (Apertura)**
        3. Datos proporcionados por Yahoo Finance (SPY/QQQ como proxies).
        """)

if __name__ == "__main__":
    main()
