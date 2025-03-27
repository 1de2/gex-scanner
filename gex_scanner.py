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
from typing import Dict, Optional, List

# Configuración
NY_TZ = pytz.timezone("America/New_York")
TODAY = datetime.now(NY_TZ).date()

class GEXScanner:
    def __init__(self):
        self.history_file = "scan_history.json"
        self.load_history()

    def load_history(self):
        """Carga el historial de escaneos"""
        try:
            with open(self.history_file, "r") as f:
                self.history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = {"premarket": [], "marketopen": [], "manual": []}

    def save_history(self):
        """Guarda el historial de escaneos"""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f)

    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, iv: float, 
                        r: float = 0.01, option_type: str = "call") -> tuple:
        """Calcula delta y gamma para una opción"""
        d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
        return delta, gamma

    def fetch_option_chain(self, ticker: str, expiry: datetime) -> pd.DataFrame:
        """Obtiene la cadena de opciones con manejo de fechas disponibles"""
        try:
            stock = yf.Ticker(ticker)
            available_expirations = stock.options
            
            if expiry.strftime("%Y-%m-%d") not in available_expirations:
                # Selecciona la fecha más cercana posterior
                valid_dates = [pd.to_datetime(e) for e in available_expirations 
                              if pd.to_datetime(e) >= pd.to_datetime(expiry)]
                if valid_dates:
                    expiry = min(valid_dates)
                else:
                    expiry = pd.to_datetime(available_expirations[-1])
            
            chain = stock.option_chain(expiry.strftime("%Y-%m-%d"))
            calls = chain.calls.assign(option_type="call")
            puts = chain.puts.assign(option_type="put")
            return pd.concat([calls, puts])
        except Exception as e:
            st.error(f"Error obteniendo datos: {str(e)}")
            return pd.DataFrame()

    def scan(self, ticker: str = "SPY", expiry_date: Optional[datetime] = None,
             mode: Optional[str] = None) -> Optional[Dict]:
        """Ejecuta el análisis GEX (sin límites manuales)"""
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
        
        # Calcular GEX (en millones)
        chain["gex"] = chain["gamma"] * chain["openInterest"] * (S ** 2) * 0.01 / 1e6
        
        # Determinar tipo de análisis
        analysis_type = {
            "premarket": "Pre-Market (7 AM NY)",
            "marketopen": "Market Open (9:30 AM NY)",
        }.get(mode, "Manual")
        
        # Preparar resultados
        result = {
            "analysis_type": analysis_type,
            "timestamp": datetime.now(NY_TZ).strftime("%Y-%m-%d %H:%M"),
            "ticker": ticker,
            "spot_price": round(S, 2),
            "total_gex": round(chain["gex"].sum(), 2),
            "top_calls": chain[chain["option_type"] == "call"]
                         .nlargest(3, "gex")[["strike", "gex"]]
                         .rename(columns={"gex": "gex ($M)"})
                         .to_dict("records"),
            "top_puts": chain[chain["option_type"] == "put"]
                        .nsmallest(3, "gex")[["strike", "gex"]]
                        .rename(columns={"gex": "gex ($M)"})
                        .to_dict("records"),
            "expiry_used": expiry_date.strftime("%Y-%m-%d")
        }
        
        # Guardar en historial
        if mode in ["premarket", "marketopen"]:
            self.history[mode].append(result)
        else:
            self.history["manual"].append(result)
        self.save_history()
        
        return result

    def get_history(self, mode: str, limit: int = 5) -> List[Dict]:
        """Obtiene el historial de escaneos"""
        return self.history.get(mode, [])[-limit:]

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--auto-scan", action="store_true")
    args.add_argument("--mode", choices=["premarket", "marketopen"], default=None)
    args = args.parse_args()

    scanner = GEXScanner()
    
    if args.auto_scan and args.mode:
        # Modo automático para GitHub Actions
        results = scanner.scan(mode=args.mode)
        with open("scan_results.json", "w") as f:
            json.dump(results, f)
        print(f"Análisis {args.mode} completado")
    else:
        # Interfaz Streamlit
        st.set_page_config(
            page_title="🚨 GEX Scanner Pro",
            layout="wide",
            page_icon="📊"
        )
        st.title("📊 Gamma Exposure Scanner (SPY/QQQ)")
        
        # Sidebar
        st.sidebar.header("Configuración")
        ticker = st.sidebar.selectbox("Activo:", ["SPY", "QQQ"])
        
        # Selector de fechas basado en disponibilidad
        try:
            available_dates = yf.Ticker(ticker).options
            expiry_date = st.sidebar.selectbox(
                "Fecha de Expiración:",
                options=available_dates,
                format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"),
                index=min(3, len(available_dates)-1)  # Selecciona una fecha cercana por defecto
            )
            expiry_date = pd.to_datetime(expiry_date)
        except Exception as e:
            st.sidebar.error(f"Error cargando fechas: {str(e)}")
            return
        
        # Botón de análisis manual (sin límites)
        if st.sidebar.button("🔍 Ejecutar Análisis Manual"):
            with st.spinner("Calculando GEX..."):
                results = scanner.scan(ticker, expiry_date, mode="manual")
                if results:
                    st.success("✅ Análisis completado!")
                    st.json(results)
        
        # Mostrar info (sin contador de límites)
        st.sidebar.markdown("""
        **📅 Análisis Automáticos:**  
        - ⏰ 7:00 AM NY (Pre-Market)  
        - 🕤 9:30 AM NY (Market Open)  
        """)
        
        # Pestañas
        tab1, tab2 = st.tabs(["📊 Resultados", "📅 Historial"])
        
        with tab1:
            if scanner.history["manual"]:
                st.json(scanner.history["manual"][-1])
            else:
                st.info("Ejecuta un análisis manual para ver resultados")
        
        with tab2:
            st.header("Historial Completo")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Automáticos")
                for scan_type in ["premarket", "marketopen"]:
                    with st.expander(f"{scan_type.replace('_', ' ').title()}"):
                        for scan in scanner.get_history(scan_type, 3):
                            st.json(scan)
            
            with col2:
                st.subheader("Manuales")
                for scan in scanner.get_history("manual", 10):
                    with st.expander(f"{scan['timestamp']}"):
                        st.json(scan)

if __name__ == "__main__":
    main()
