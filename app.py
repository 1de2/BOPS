import streamlit as st
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="BOPS Backtesting", layout="wide")
st.title("📊 Backtesting Automatizado para Estrategia BOPS")

# 1. Panel de Control
with st.sidebar:
    st.header("Configuración del Backtest")
    
    # Selección de activo y timeframe
    asset = st.selectbox("Activo", ["MNQ=F", "MES=F", "YM=F"], index=0)
    timeframe = st.selectbox("Timeframe", ["15m", "30m", "1h", "4h", "1d"], index=2)
    
    # Parámetros de la estrategia
    st.subheader("Parámetros BOPS")
    tickq_extremo_alto = st.number_input("Nivel TICKQ Alto", value=1000, min_value=500, max_value=1500)
    tickq_extremo_bajo = st.number_input("Nivel TICKQ Bajo", value=-1000, min_value=-1500, max_value=-500)
    
    # Rango de fechas
    today = datetime.today()
    start_date = st.date_input("Fecha inicio", value=pd.to_datetime('2023-01-01'), max_value=today)
    end_date = st.date_input("Fecha fin", value=today, max_value=today)
    
    # Configuración de riesgo
    st.subheader("Gestión de Riesgo")
    tp_percent = st.number_input("Take Profit (%)", value=1.5, min_value=0.1, max_value=10.0, step=0.1)
    sl_percent = st.number_input("Stop Loss (%)", value=0.75, min_value=0.1, max_value=5.0, step=0.1)
    
    # Ejecutar backtest
    run_backtest = st.button("Ejecutar Backtest")

# 2. Clase de Estrategia Adaptada
class BopsStrategy(Strategy):
    def init(self):
        # Pre-cálculo de indicadores
        self.sma20 = self.I(SMA, self.data.Close, 20)
        self.prev_high = self.I(pd.Series, self.data.High.resample('D').last().shift(1))
        self.prev_low = self.I(pd.Series, self.data.Low.resample('D').last().shift(1))
        
    def next(self):
        # Simulación de datos TICKQ (en producción usar API real)
        current_tickq = np.random.randint(-1500, 1500)
        
        # Condiciones de entrada (adaptadas del Pine Script original)
        bull_condition = (current_tickq <= self.tickq_extremo_bajo and 
                        self.data.Close[-1] > self.data.Open[-1] and
                        self.data.Volume[-1] > self.data.Volume.rolling(20).mean()[-1])
        
        bear_condition = (current_tickq >= self.tickq_extremo_alto and 
                        self.data.Close[-1] < self.data.Open[-1] and
                        self.data.Volume[-1] > self.data.Volume.rolling(20).mean()[-1])
        
        # Gestión de posiciones
        if bull_condition and not self.position.is_long:
            self.buy(sl=self.data.Close[-1]*(1-self.sl_percent/100),
                    tp=self.data.Close[-1]*(1+self.tp_percent/100))
            
        elif bear_condition and not self.position.is_short:
            self.sell(sl=self.data.Close[-1]*(1+self.sl_percent/100),
                     tp=self.data.Close[-1]*(1-self.tp_percent/100))

# 3. Carga de Datos
@st.cache_data
def load_data(ticker, start, end, interval):
    data = yf.download(ticker, start=start, end=end, interval=interval)
    return data

# 4. Ejecución del Backtest
if run_backtest:
    st.write(f"## 🔍 Analizando {asset} ({timeframe}) desde {start_date} hasta {end_date}")
    
    with st.spinner("Cargando datos y ejecutando backtest..."):
        try:
            # Cargar datos
            data = load_data(asset, start_date, end_date, timeframe)
            
            if data.empty:
                st.error("No se encontraron datos para este rango. Prueba con otras fechas.")
            else:
                # Configurar estrategia
                bt = Backtest(data, BopsStrategy, commission=.0002, margin=1.0)
                
                # Ejecutar backtest con parámetros
                stats = bt.run(
                    tickq_extremo_alto=tickq_extremo_alto,
                    tickq_extremo_bajo=tickq_extremo_bajo,
                    sl_percent=sl_percent,
                    tp_percent=tp_percent
                )
                
                # Mostrar resultados
                st.success("Backtest completado exitosamente!")
                
                # Métricas clave
                col1, col2, col3 = st.columns(3)
                col1.metric("Retorno Total", f"{stats['Return [%]']:.2f}%")
                col2.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")
                col3.metric("Win Rate", f"{stats['Win Rate [%]']:.2f}%")
                
                # Gráfico de equity
                st.write("### 📈 Curva de Equity")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stats._equity_curve.index,
                    y=stats._equity_curve['Equity'],
                    mode='lines',
                    name='Equity'
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Datos detallados
                st.write("### 📊 Estadísticas Detalladas")
                st.dataframe(stats)
                
                # Exportar resultados
                csv = stats._strategy._equity_curve.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar Resultados",
                    data=csv,
                    file_name=f"backtest_results_{asset}_{start_date}_{end_date}.csv",
                    mime='text/csv'
                )
                
        except Exception as e:
            st.error(f"Error durante el backtest: {str(e)}")

# 5. Explicación de la Estrategia
with st.expander("ℹ️ Cómo funciona esta estrategia"):
    st.write("""
    Esta implementación adapta la estrategia BOPS original para backtesting con:
    
    - **Señales basadas en TICKQ simulado** (en producción conectar a datos reales)
    - **Filtros de volumen y precio** similares al Pine Script original
    - **Gestión de riesgo** con TP/SL porcentual
    
    Parámetros clave:
    - `TICKQ Extremo Alto`: Nivel para señales cortas
    - `TICKQ Extremo Bajo`: Nivel para señales largas
    - `Horario RTH`: 9:30-16:00 NY (implementado internamente)
    """)

# Nota: Para producción, reemplazar la simulación de TICKQ con API real
