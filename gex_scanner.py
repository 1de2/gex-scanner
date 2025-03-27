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
MAX_MANUAL_SCANS = 1  # Límite de análisis manuales

class GEXScanner:
    def __init__(self):
        self.counter_file = "scan_counter.json"
        self.history_file = "scan_history.json"
        self.load_counter()
        self.load_history()

    def load_counter(self):
        """Carga el contador de escaneos desde JSON"""
        try:
            with open(self.counter_file, "r") as f:
                data = json.load(f)
                if data["date"] == str(TODAY):
                    self.counter = data
                    return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        self.counter = {"date": str(TODAY), "manual_scans": 0}

    def save_counter(self):
        """Guarda el contador en JSON"""
        with open(self.counter_file, "w") as f:
            json.dump(self.counter, f)

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
        """Obtiene la cadena de opciones"""
        try:
            chain = yf.Ticker(ticker).option_chain(expiry.strftime("%Y-%m-%d"))
            calls = chain.calls.assign(option_type="call")
            puts = chain.puts.assign(option_type="put")
            return pd.concat([calls, puts])
        except Exception as e:
            st.error(f"Error obteniendo datos: {str(e)}")
            return pd.DataFrame()

    def scan(self, ticker: str = "SPY", expiry_date: Optional[datetime] = None,
             mode: Optional[str] = None) -> Optional[Dict]:
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
        
        # Calcular GEX (en millones)
        chain["gex"] = chain["gamma"] * chain["openInterest"] * (S ** 2) * 0.01 / 1e6
        
        # Determinar tipo de análisis
        if mode == "premarket":
            analysis_type = "Pre-Market (7 AM NY)"
        elif mode == "marketopen":
            analysis_type = "Market Open (9:30 AM NY)"
        else:
            analysis_type = "Manual"
            if self.counter["manual_scans"] >= MAX_MANUAL_SCANS:
                return {"error": "Límite diario alcanzado"}
            self.counter["manual_scans"] += 1
            self.save_counter()
        
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
                        .to_dict("records")
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
        expiry_date = st.sidebar.date_input(
            "Fecha de Expiración:",
            TODAY + timedelta(days=3)
        )
        
        # Botón de análisis manual
        if st.sidebar.button("🔍 Ejecutar Análisis Manual"):
            with st.spinner("Calculando GEX..."):
                results = scanner.scan(ticker, expiry_date, mode="manual")
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
        
        # Pestañas
        tab1, tab2, tab3 = st.tabs(["📊 Resultados", "📅 Historial", "⚙️ Configuración"])
        
        with tab1:
            st.header("Análisis Más Reciente")
            if scanner.history["manual"]:
                st.json(scanner.history["manual"][-1])
            else:
                st.info("Ejecuta un análisis manual para ver resultados")
        
        with tab2:
            st.header("Historial de Escaneos")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Pre-Market (7 AM NY)")
                for scan in scanner.get_history("premarket"):
                    with st.expander(f"{scan['timestamp']} - ${scan['total_gex']}M"):
                        st.json(scan)
            
            with col2:
                st.subheader("Market Open (9:30 AM NY)")
                for scan in scanner.get_history("marketopen"):
                    with st.expander(f"{scan['timestamp']} - ${scan['total_gex']}M"):
                        st.json(scan)
        
        with tab3:
            st.header("Configuración Avanzada")
            st.markdown("""
            ### GitHub Actions Setup
            ```yaml
            # .github/workflows/scan.yml
            name: GEX Auto-Scanner
            on:
              schedule:
                - cron: '0 11 * * 1-5'  # 7 AM NY (UTC-4)
                - cron: '30 13 * * 1-5'  # 9:30 AM NY
            jobs:
              scan:
                runs-on: ubuntu-latest
                steps:
                  - uses: actions/checkout@v4
                  - run: pip install -r requirements.txt
                  - run: python gex_scanner.py --auto-scan --mode premarket
                  - run: python gex_scanner.py --auto-scan --mode marketopen
            ```
            """)

if __name__ == "__main__":
    main()
