import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz
import json
from typing import Dict, Optional, List

# Configuración
NY_TZ = pytz.timezone("America/New_York")

class GEXScanner:
    def __init__(self):
        self.history_file = "scan_history.json"
        self.load_history()

    def load_history(self):
        """Carga el historial de escaneos desde JSON"""
        try:
            with open(self.history_file, "r") as f:
                self.history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = {"premarket": [], "marketopen": [], "manual": []}

    def save_history(self):
        """Guarda el historial en JSON"""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def calculate_greeks(self, S: float, K: float, T: float, iv: float, 
                        option_type: str) -> tuple:
        """Calcula delta y gamma para una opción"""
        try:
            d1 = (np.log(S / K) + (0.01 + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
            
            # Asegurar que delta y gamma no sean NaN
            if np.isnan(delta) or np.isnan(gamma):
                delta, gamma = 0.0, 0.0
            
            return delta, gamma
        except Exception as e:
            st.error(f"Error calculando griegas: {str(e)}")
            return 0.0, 0.0

    def fetch_option_chain(self, ticker: str, expiry: datetime) -> pd.DataFrame:
        """Obtiene la cadena de opciones con manejo robusto de errores"""
        try:
            stock = yf.Ticker(ticker)
            
            if not hasattr(stock, 'options') or not stock.options:
                st.error(f"No hay fechas de expiración para {ticker}")
                return pd.DataFrame()
            
            expiry_str = expiry.strftime("%Y-%m-%d")
            available_dates = [pd.to_datetime(d) for d in stock.options]
            
            st.write(f"Fechas disponibles de expiración: {available_dates}")  # Depuración

            if expiry_str not in stock.options:
                valid_dates = [d for d in available_dates if d >= pd.to_datetime(expiry)]
                expiry = min(valid_dates) if valid_dates else max(available_dates)
                expiry_str = expiry.strftime("%Y-%m-%d")
                st.warning(f"Usando fecha {expiry_str} (la solicitada no estaba disponible)")

            chain = stock.option_chain(expiry_str)

            # Verificar las columnas disponibles en la cadena de opciones
            st.write("Columnas disponibles en la cadena de opciones:", chain.calls.columns)

            defaults = {
                'impliedVolatility': 0.3,
                'openInterest': 0,
                'lastPrice': 0.0,
                'strike': 0.0
            }
            
            # Asegurarse de que la columna 'contractSymbol' contiene la fecha de expiración
            dfs = []
            for opt_type, df in [('call', chain.calls), ('put', chain.puts)]:
                df = df.copy()
                for col, default in defaults.items():
                    if col not in df.columns:
                        df[col] = default
                df['option_type'] = opt_type

                # Extraer la fecha de expiración desde el 'contractSymbol' si está disponible
                if 'contractSymbol' in df.columns:
                    df['expiry'] = df['contractSymbol'].apply(self.extract_expiry_from_symbol)
                
                # Si no se pudo extraer la fecha, intentamos usar 'lastTradeDate'
                if 'expiry' not in df.columns:
                    df['expiry'] = df['lastTradeDate'].apply(lambda x: pd.to_datetime(x) + timedelta(days=30))  # Aproximación por defecto
                
                dfs.append(df)
            
            return pd.concat(dfs, ignore_index=True)
            
        except Exception as e:
            st.error(f"Error crítico: {str(e)}")
            return pd.DataFrame()

    def extract_expiry_from_symbol(self, symbol: str) -> pd.Timestamp:
        """Intenta extraer la fecha de expiración desde el símbolo del contrato de opción"""
        try:
            # Asegúrate de que el símbolo tenga el formato esperado
            expiry_str = symbol[-8:]  # Asumiendo que la fecha de expiración está al final del símbolo en formato YYYYMMDD
            expiry_date = pd.to_datetime(expiry_str, format='%y%m%d')
            return expiry_date
        except Exception as e:
            st.error(f"Error extrayendo fecha de expiración: {str(e)}")
            return pd.NaT

    def scan(self, ticker: str, expiry_date: datetime, mode: str = None) -> Dict:
        """Ejecuta el análisis GEX completo"""
        try:
            stock_data = yf.Ticker(ticker).history(period="1d")
            if stock_data.empty:
                return {"error": "No se pudo obtener el precio del activo"}
            S = stock_data["Close"].iloc[-1]
            
            chain = self.fetch_option_chain(ticker, expiry_date)
            if chain.empty:
                return {"error": "Cadena de opciones vacía"}
            
            if "expiry" in chain.columns:
                chain["T"] = (pd.to_datetime(chain["expiry"]) - pd.Timestamp.now()).dt.days / 365.25
            else:
                st.error("La columna 'expiry' no está presente en la cadena de opciones")
                return {}
            
            # Verificar los valores de delta, gamma y openInterest
            chain[["delta", "gamma"]] = chain.apply(
                lambda row: self.calculate_greeks(
                    S, row["strike"], row["T"], row["impliedVolatility"], row["option_type"]
                ),
                axis=1, result_type="expand"
            )
            
            # Verificar valores de gex
            chain["gex"] = chain["gamma"] * chain["openInterest"] * (S ** 2) * 0.01 / 1e6

            # Verificar si hay NaN en gex
            chain['gex'] = chain['gex'].fillna(0)
            
            result = {
                "analysis_type": mode if mode else "Manual",
                "timestamp": datetime.now(NY_TZ).strftime("%Y-%m-%d %H:%M"),
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
            
            key = mode if mode in ["premarket", "marketopen"] else "manual"
            self.history[key].append(result)
            self.save_history()
            
            return result
            
        except Exception as e:
            return {"error": f"Error en análisis: {str(e)}"}

def main():
    st.set_page_config(
        page_title="🚨 GEX Scanner Pro",
        layout="wide",
        page_icon="📊"
    )
    st.title("📊 Gamma Exposure Scanner (SPY/QQQ)")
    
    scanner = GEXScanner()
    
    st.sidebar.header("Configuración")
    ticker = st.sidebar.selectbox(
        "Seleccionar Activo:",
        ["SPY", "QQQ", "IWM", "AAPL", "TSLA"]
    )
    
    try:
        stock = yf.Ticker(ticker)
        available_dates = stock.options if hasattr(stock, 'options') else []
        
        if not available_dates:
            st.sidebar.error("No hay fechas disponibles para este activo")
            return
            
        expiry_date = st.sidebar.selectbox(
            "Fecha de Expiración:",
            options=available_dates,
            format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"),
            index=min(3, len(available_dates)-1)  # Paréntesis de cierre añadido
        )
        expiry_date = pd.to_datetime(expiry_date)
        
    except Exception as e:
        st.sidebar.error(f"Error cargando fechas: {str(e)}")
        return
    
    if st.sidebar.button("🔍 Ejecutar Análisis", type="primary"):
        with st.spinner("Calculando GEX..."):
            result = scanner.scan(ticker, expiry_date)
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success("✅ Análisis completado!")
                with st.expander("Ver resultados completos", expanded=True):
                    st.json(result)
    
    tab1, tab2 = st.tabs(["📈 Resultados", "🕒 Historial"])
    
    with tab1:
        if scanner.history["manual"]:
            st.subheader("Último Análisis")
            st.json(scanner.history["manual"][-1])
        else:
            st.info("Ejecuta un análisis para ver resultados")
    
    with tab2:
        st.header("Historial de Escaneos")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Automáticos")
            for scan_type in ["premarket", "marketopen"]:
                if scanner.history[scan_type]:
                    with st.expander(f"{scan_type.replace('market', ' Market ').title()}"):
                        for scan in scanner.history[scan_type][-3:]:
                            st.json(scan)
        
        with col2:
            st.subheader("Manuales")
            for scan in scanner.history["manual"][-10:]:
                with st.expander(f"{scan['timestamp']} - {scan['ticker']}"):
                    st.json(scan)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Limpiar Historial", type="secondary"):
        scanner.history = {"premarket": [], "marketopen": [], "manual": []}
        scanner.save_history()
        st.sidebar.success("Historial limpiado")

if __name__ == "__main__":
    main()
