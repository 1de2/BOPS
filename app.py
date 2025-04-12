import streamlit as st
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="BOPS Pro Backtester",
    page_icon="📈",
    layout="wide"
)

# Título y descripción
st.title("🚀 BOPS Strategy Backtester")
st.markdown("""
Backtesting automatizado para la estrategia BOPS con datos de mercado en tiempo real.
""")

# Sidebar con controles
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selector de activo y timeframe
    asset = st.selectbox(
        "Seleccionar Activo",
        ["MNQ=F", "MES=F", "YM=F"],
        index=0
    )
    
    timeframe_map = {
        "15 Minutos": "15m",
        "1 Hora": "1h",
        "4 Horas": "4h",
        "1 Día": "1d"
    }
    timeframe = st.selectbox(
        "Intervalo Temporal",
        list(timeframe_map.keys()),
        index=1
    )
    
    # Parámetros de estrategia
    st.subheader("📊 Parámetros BOPS")
    tick_high = st.slider("Umbral TICKQ Alto", 800, 1500, 1000)
    tick_low = st.slider("Umbral TICKQ Bajo", -1500, -800, -1000)
    
    # Gestión de riesgo
    st.subheader("🛡️ Gestión de Riesgo")
    tp = st.number_input("Take Profit (%)", 0.5, 10.0, 1.5, 0.1)
    sl = st.number_input("Stop Loss (%)", 0.1, 5.0, 0.75, 0.05)
    
    # Rango de fechas
    st.subheader("📅 Rango de Backtesting")
    start_date = st.date_input(
        "Fecha de inicio",
        datetime(2023, 1, 1),
        max_value=datetime.today()
    )
    end_date = st.date_input(
        "Fecha de fin",
        datetime.today(),
        max_value=datetime.today()
    )

# Clase de estrategia optimizada
class OptimizedBopsStrategy(Strategy):
    def init(self):
        close = self.data.Close
        self.sma20 = self.I(SMA, close, 20)
        self.atr = self.I(ATR, self.data.HLC, 14)
        
    def next(self):
        # Simulación de datos de mercado
        current_tick = np.random.normal(0, 500)
        vol_spread = self.data.Volume[-1] - self.data.Volume.rolling(20).mean()[-1]
        
        # Condiciones de entrada
        long_cond = (
            current_tick <= self.params.tick_low and
            vol_spread > 0 and
            crossover(self.data.Close, self.sma20)
        )
        
        short_cond = (
            current_tick >= self.params.tick_high and
            vol_spread < 0 and
            crossunder(self.data.Close, self.sma20)
        )
        
        # Gestión de posiciones
        price = self.data.Close[-1]
        if long_cond and not self.position.is_long:
            self.buy(
                sl=price * (1 - self.params.sl/100),
                tp=price * (1 + self.params.tp/100)
            )
            
        elif short_cond and not self.position.is_short:
            self.sell(
                sl=price * (1 + self.params.sl/100),
                tp=price * (1 - self.params.tp/100)
            )

# Carga de datos con caché
@st.cache_data(ttl=3600)
def load_market_data(ticker, start, end, interval):
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval)
        if data.empty:
            st.error("⚠️ No se encontraron datos. Prueba con otro rango de fechas.")
            return None
        return data
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None

# Ejecución principal
if st.sidebar.button("▶️ Ejecutar Backtest"):
    st.write(f"## 📊 Resultados para {asset} ({timeframe})")
    
    with st.spinner("Cargando datos y ejecutando backtest..."):
        data = load_market_data(
            asset,
            start_date,
            end_date,
            timeframe_map[timeframe]
        )
        
        if data is not None:
            bt = Backtest(
                data,
                OptimizedBopsStrategy,
                commission=.0002,
                margin=1.0,
                exclusive_orders=True
            )
            
            stats = bt.run(
                tick_high=tick_high,
                tick_low=tick_low,
                tp=tp,
                sl=sl
            )
            
            # Mostrar métricas clave
            col1, col2, col3 = st.columns(3)
            col1.metric("Retorno Total", f"{stats['Return [%]']:.2f}%")
            col2.metric("Ratio Sharpe", f"{stats['Sharpe Ratio']:.2f}")
            col3.metric("Operaciones", stats['# Trades'])
            
            # Gráfico interactivo
            st.plotly_chart(
                bt.plot(resample=False),
                use_container_width=True
            )
            
            # Resultados detallados
            with st.expander("📝 Ver estadísticas completas"):
                st.dataframe(stats)
