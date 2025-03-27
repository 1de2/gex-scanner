import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz
import json
from typing import Dict, Optional, List

class GEXScanner:
    def __init__(self):
        self.history_file = "scan_history.json"
        self.load_history()

    def load_history(self):
        try:
            with open(self.history_file, "r") as f:
                self.history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = {"premarket": [], "marketopen": [], "manual": []}

    def save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f)

    def fetch_option_chain(self, ticker: str, expiry: datetime) -> pd.DataFrame:
        """Versión mejorada con manejo robusto de errores"""
        try:
            stock = yf.Ticker(ticker)
            
            if not hasattr(stock, 'options') or not stock.options:
                st.error(f"No hay datos de opciones para {ticker}")
                return pd.DataFrame()
                
            expiry_str = expiry.strftime("%Y-%m-%d")
            available_dates = pd.to_datetime(stock.options)
            
            if expiry_str not in stock.options:
                valid_dates = [d for d in available_dates if d >= pd.to_datetime(expiry)]
                expiry = min(valid_dates) if valid_dates else available_dates[-1]
                expiry_str = expiry.strftime("%Y-%m-%d")
                st.warning(f"Usando fecha {expiry_str} (la solicitada no estaba disponible)")
            
            chain = stock.option_chain(expiry_str)
            
            # Campos obligatorios con valores por defecto
            required_cols = {
                'impliedVolatility': 0.3,
                'openInterest': 0,
                'strike': 0,
                'lastPrice': 0
            }
            
            # Procesar calls y puts
            dfs = []
            for option_type, df in [('call', chain.calls), ('put', chain.puts)]:
                df = df.copy()
                for col, default in required_cols.items():
                    if col not in df.columns:
                        df[col] = default
                df['option_type'] = option_type
                dfs.append(df)
                
            return pd.concat(dfs, ignore_index=True)
            
        except Exception as e:
            st.error(f"Error al obtener datos: {str(e)}")
            return pd.DataFrame()

    def calculate_greeks(self, S: float, K: float, T: float, iv: float, 
                        option_type: str) -> tuple:
        try:
            d1 = (np.log(S / K) + (0.01 + 0.5 * iv**2) * T
            d1 /= (iv * np.sqrt(T))
            delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
            return delta, gamma
        except:
            return (0, 0)  # Valores por defecto si hay error en cálculo

    def scan(self, ticker: str, expiry_date: datetime, mode: str = None) -> Dict:
        try:
            S = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
            chain = self.fetch_option_chain(ticker, expiry_date)
            
            if chain.empty:
                return {"error": "Cadena de opciones vacía"}
            
            chain["T"] = (pd.to_datetime(chain["expiry"]) - pd.Timestamp.now()).dt.days / 365
            chain[["delta", "gamma"]] = chain.apply(
                lambda row: self.calculate_greeks(
                    S, row["strike"], row["T"], row["impliedVolatility"], row["option_type"]
                ),
                axis=1, result_type="expand"
            )
            
            chain["gex"] = chain["gamma"] * chain["openInterest"] * (S ** 2) * 0.01 / 1e6
            
            result = {
                "analysis_type": mode if mode else "Manual",
                "timestamp": datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d %H:%M"),
                "ticker": ticker,
                "spot_price": round(S, 2),
                "total_gex": round(chain["gex"].sum(), 2),
                "expiry_used": expiry_date.strftime("%Y-%m-%d"),
                "top_calls": chain[chain["option_type"] == "call"]
                             .nlargest(3, "gex")[["strike", "gex"]]
                             .rename(columns={"gex": "gex ($M)"})
                             .to_dict("records"),
                "top_puts": chain[chain["option_type"] == "put"]
                            .nsmallest(3, "gex")[["strike", "gex"]]
                            .rename(columns={"gex": "gex ($M)"})
                            .to_dict("records")
            }
            
            self.history[mode if mode else "manual"].append(result)
            self.save_history()
            return result
            
        except Exception as e:
            return {"error": str(e)}

def main():
    st.set_page_config(page_title="GEX Scanner Pro", layout="wide")
    st.title("📊 Gamma Exposure Scanner")
    
    scanner = GEXScanner()
    
    # Sidebar
    st.sidebar.header("Configuración")
    ticker = st.sidebar.selectbox("Seleccionar Activo:", ["SPY", "QQQ", "IWM", "AAPL"])
    
    try:
        available_dates = yf.Ticker(ticker).options
        expiry_date = st.sidebar.selectbox(
            "Fecha de Expiración:",
            options=available_dates,
            format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"),
            index=min(3, len(available_dates)-1)
        )
        expiry_date = pd.to_datetime(expiry_date)
    except Exception as e:
        st.sidebar.error(f"Error cargando fechas: {str(e)}")
        return
    
    if st.sidebar.button("🔍 Ejecutar Análisis"):
        with st.spinner("Calculando GEX..."):
            result = scanner.scan(ticker, expiry_date)
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success("✅ Análisis completado!")
                st.json(result)
    
    # Mostrar historial
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Limpiar Historial"):
        scanner.history = {"premarket": [], "marketopen": [], "manual": []}
        scanner.save_history()
    
    # Pestañas principales
    tab1, tab2 = st.tabs(["📈 Resultados", "🕒 Historial"])
    
    with tab1:
        if scanner.history["manual"]:
            st.json(scanner.history["manual"][-1])
    
    with tab2:
        st.header("Últimos Análisis")
        for scan_type, scans in scanner.history.items():
            with st.expander(f"{scan_type.title()} ({len(scans)} scans)"):
                for scan in scans[-5:]:
                    st.json(scan)

if __name__ == "__main__":
    main()
