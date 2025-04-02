import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import time
warnings.filterwarnings('ignore')

# Streamlit configuration for dark mode
st.set_page_config(
    layout="wide",
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Enhanced Dark mode CSS
st.markdown("""
    <style>
    /* Base styling */
    .main {
        background-color: #121212;
        color: #f0f0f0;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1e1e2d;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        padding: 25px;
    }
    
    /* Headers */
    h1 {
        color: #4CAF50;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 1px solid #333340;
        margin-bottom: 20px;
    }
    
    h2, h3 {
        color: #4CAF50;
        font-weight: 600;
        margin-top: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50, #2E7D32);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #2E7D32, #1B5E20);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    
    /* Form inputs */
    .stTextInput>div>input, .stSelectbox>div>div>div, .stMultiSelect>div>div>div {
        background-color: #262630;
        color: #f0f0f0;
        border-radius: 8px;
        border: 1px solid #333340;
        padding: 10px;
    }
    
    .stMultiSelect>div>div>div:hover {
        border: 1px solid #4CAF50;
    }
    
    .stSlider>div>div>div {
        background-color: #333340;
    }
    
    .stSlider>div>div>div>div {
        background-color: #4CAF50;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2d, #262636);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 20px;
        border-left: 4px solid #4CAF50;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card h4 {
        color: #4CAF50;
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 18px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e2d;
        border-radius: 10px;
        padding: 5px;
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 16px;
        background-color: #262636;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    /* Info box */
    .stInfo {
        background-color: #1e3a5c;
        color: #f0f0f0;
        border: none;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #4299e1;
    }
    
    /* Success box */
    .stSuccess {
        background-color: #1e3b2c;
        color: #f0f0f0;
        border: none;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #4CAF50;
    }
    
    /* Error box */
    .stError {
        background-color: #3b1e1e;
        color: #f0f0f0;
        border: none;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #e53e3e;
    }
    
    /* Divider */
    hr {
        border-color: #333340;
        margin: 30px 0;
    }
    
    /* Welcome section */
    .welcome-box {
        background: linear-gradient(135deg, #1e1e2d, #262636);
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 30px;
        border-left: 4px solid #4CAF50;
    }
    
    .welcome-box h2 {
        color: #4CAF50;
        font-size: 24px;
        margin-bottom: 20px;
    }
    
    /* Glow effect for important elements */
    .glow-effect {
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.3);
    }
    
    /* Tooltip customization */
    .stTooltipIcon {
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Title with emoji
st.markdown("<h1>📈 Multi-Stock Prediction Dashboard</h1>", unsafe_allow_html=True)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("<h2>⚙️ Configuration</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Multi-select for tickers with default popular companies
    default_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    tickers = st.multiselect(
        "Stock Ticker Symbols",
        options=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"],
        default=default_tickers,
        help="Select multiple stocks to analyze (e.g., AAPL = Apple, MSFT = Microsoft)"
    )
    
    period = st.selectbox(
        "Historical Data Period",
        ["1y", "2y", "5y", "10y"],
        index=0,
        help="Choose how far back to fetch historical data"
    )
    
    prediction_days = st.slider(
        "Prediction Horizon (Days)",
        7, 90, 30,
        help="Number of days to predict into the future"
    )
    
    lstm_units = st.slider(
        "LSTM Model Complexity",
        50, 200, 100,
        step=10,
        help="Higher values increase model complexity"
    )
    
    epochs = st.slider(
        "Training Iterations",
        10, 100, 50,
        step=5,
        help="Number of training cycles"
    )
    
    # Technical indicator selection
    st.markdown("<h3>Technical Indicators</h3>", unsafe_allow_html=True)
    use_sma = st.checkbox("Simple Moving Average (SMA)", value=True)
    use_ema = st.checkbox("Exponential Moving Average (EMA)", value=True)
    use_rsi = st.checkbox("Relative Strength Index (RSI)", value=True)
    use_macd = st.checkbox("MACD", value=True)
    use_bollinger = st.checkbox("Bollinger Bands", value=True)
    
    # SMA/EMA periods if enabled
    if use_sma:
        sma_period = st.slider("SMA Period", 5, 200, 20)
        
    if use_ema:
        ema_period = st.slider("EMA Period", 5, 200, 20)
        
    st.markdown("<hr>", unsafe_allow_html=True)
    run_button = st.button("🚀 Run Analysis")

def get_stock_data(ticker, period, max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime(2025, 4, 2)
            period_days = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650}
            start_date = end_date - timedelta(days=period_days[period])
            df = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                raise ValueError(f"No data returned for {ticker}. Ticker may be invalid or data unavailable.")
            
            company_name = stock.info.get('shortName', ticker)
            st.success(f"✅ Data fetched for {ticker} ({company_name}) - {len(df)} rows")
            st.write(f"📅 Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            return df, company_name
        
        except ValueError as ve:
            st.error(f"⚠️ Attempt {attempt + 1}/{max_retries} - {str(ve)}")
            attempt += 1
            time.sleep(2)
        except Exception as e:
            st.error(f"⚠️ Attempt {attempt + 1}/{max_retries} - Error: {str(e)}")
            attempt += 1
            time.sleep(2)
    
    st.error(f"❌ Failed to fetch data for {ticker} after {max_retries} attempts.")
    return None, ticker

def prepare_data(df, lookback_days, feature='Close'):
    data = df[[feature]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback_days, len(scaled_data)):
        X.append(scaled_data[i-lookback_days:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val, scaler, scaled_data

def build_lstm_model(X_train, y_train, X_val, y_val, lstm_units, epochs):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=lstm_units//2),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Training: {int(progress*100)}% (Val Loss: {logs.get('val_loss', 0):.4f})")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, ProgressCallback()],
        verbose=0
    )
    status_text.text("✅ Training complete!")
    return model, history

def predict_future(model, df, lookback_days, future_days, scaler, feature='Close'):
    last_sequence = scaler.transform(df[feature].values[-lookback_days:].reshape(-1, 1))
    predictions = []
    current_batch = last_sequence.reshape((1, lookback_days, 1))
    
    for _ in range(future_days):
        pred = model.predict(current_batch, verbose=0)[0, 0]
        predictions.append(pred)
        current_batch = np.roll(current_batch, -1, axis=1)
        current_batch[0, -1, 0] = pred
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# With this corrected version:
def calculate_technical_indicators(df):
    # Create a copy so we don't modify the original dataframe
    indicators = df.copy()
    
    # Calculate SMA (Simple Moving Average)
    if use_sma:
        indicators[f'SMA_{sma_period}'] = indicators['Close'].rolling(window=sma_period).mean()
    else:
        # Default SMA
        indicators['SMA_20'] = indicators['Close'].rolling(window=20).mean()
        indicators['SMA_50'] = indicators['Close'].rolling(window=50).mean()
        indicators['SMA_200'] = indicators['Close'].rolling(window=200).mean()
    
    # Calculate EMA (Exponential Moving Average)
    if use_ema:
        indicators[f'EMA_{ema_period}'] = indicators['Close'].ewm(span=ema_period, adjust=False).mean()
    else:
        # Default EMA
        indicators['EMA_12'] = indicators['Close'].ewm(span=12, adjust=False).mean()
        indicators['EMA_26'] = indicators['Close'].ewm(span=26, adjust=False).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = indicators['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    indicators['MACD_Line'] = indicators['Close'].ewm(span=12, adjust=False).mean() - indicators['Close'].ewm(span=26, adjust=False).mean()
    indicators['MACD_Signal'] = indicators['MACD_Line'].ewm(span=9, adjust=False).mean()
    indicators['MACD_Histogram'] = indicators['MACD_Line'] - indicators['MACD_Signal']
    
    # Calculate Bollinger Bands
    window = 20
    indicators['Bollinger_MA'] = indicators['Close'].rolling(window=window).mean()
    indicators['Bollinger_STD'] = indicators['Close'].rolling(window=window).std()
    indicators['Bollinger_Upper'] = indicators['Bollinger_MA'] + (indicators['Bollinger_STD'] * 2)
    indicators['Bollinger_Lower'] = indicators['Bollinger_MA'] - (indicators['Bollinger_STD'] * 2)
    
    return indicators

# Function to plot technical indicators
def plot_technical_indicators(df, indicators, ticker):
    # Plot price and moving averages
    fig1 = go.Figure()
    
    # Add price line
    fig1.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'], 
        name='Close Price',
        line=dict(color='#00b4d8', width=2)
    ))
    
    # Add SMA lines if selected
    if use_sma:
        fig1.add_trace(go.Scatter(
            x=indicators.index, 
            y=indicators[f'SMA_{sma_period}'], 
            name=f'SMA ({sma_period})',
            line=dict(color='#4CAF50', width=1.5)
        ))
    else:
        fig1.add_trace(go.Scatter(
            x=indicators.index, 
            y=indicators['SMA_50'], 
            name='SMA (50)',
            line=dict(color='#4CAF50', width=1.5)
        ))
        fig1.add_trace(go.Scatter(
            x=indicators.index, 
            y=indicators['SMA_200'], 
            name='SMA (200)',
            line=dict(color='#ff9800', width=1.5)
        ))
    
    # Add EMA line if selected
    if use_ema:
        fig1.add_trace(go.Scatter(
            x=indicators.index, 
            y=indicators[f'EMA_{ema_period}'], 
            name=f'EMA ({ema_period})',
            line=dict(color='#f44336', width=1.5)
        ))
    
    # Add Bollinger Bands if selected
    if use_bollinger:
        fig1.add_trace(go.Scatter(
            x=indicators.index, 
            y=indicators['Bollinger_Upper'], 
            name='Bollinger Upper',
            line=dict(color='rgba(173, 216, 230, 0.5)', width=1, dash='dash')
        ))
        fig1.add_trace(go.Scatter(
            x=indicators.index, 
            y=indicators['Bollinger_Lower'], 
            name='Bollinger Lower',
            line=dict(color='rgba(173, 216, 230, 0.5)', width=1, dash='dash'),
            fill='tonexty', 
            fillcolor='rgba(173, 216, 230, 0.1)'
        ))
    
    fig1.update_layout(
        title=f"{ticker} Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, Arial", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',
            zeroline=False,
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',
            zeroline=False,
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)'
        )
    )
    
    # Create second figure for RSI
    fig2 = go.Figure()
    
    if use_rsi:
        fig2.add_trace(go.Scatter(
            x=indicators.index, 
            y=indicators['RSI'], 
            name='RSI',
            line=dict(color='#f44336', width=1.5)
        ))
        
        # Add overbought/oversold reference lines
        fig2.add_trace(go.Scatter(
            x=[indicators.index[0], indicators.index[-1]],
            y=[70, 70],
            name='Overbought (70)',
            line=dict(color='rgba(255, 127, 80, 0.7)', width=1, dash='dash')
        ))
        
        fig2.add_trace(go.Scatter(
            x=[indicators.index[0], indicators.index[-1]],
            y=[30, 30],
            name='Oversold (30)',
            line=dict(color='rgba(173, 255, 47, 0.7)', width=1, dash='dash')
        ))
        
        fig2.update_layout(
            title="Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI Value",
            height=300,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Arial", size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(range=[0, 100]),
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=False,
                showline=True,
                linecolor='rgba(255, 255, 255, 0.2)'
            ),
            yaxis_gridcolor='rgba(255, 255, 255, 0.1)'
        )
    
    # Create third figure for MACD
    fig3 = go.Figure()
    
    if use_macd:
        fig3.add_trace(go.Scatter(
            x=indicators.index, 
            y=indicators['MACD_Line'], 
            name='MACD Line',
            line=dict(color='#4CAF50', width=1.5)
        ))
        
        fig3.add_trace(go.Scatter(
            x=indicators.index, 
            y=indicators['MACD_Signal'], 
            name='Signal Line',
            line=dict(color='#ff9800', width=1.5)
        ))
        
        # Add histogram as bar chart
        colors = ['red' if val < 0 else 'green' for val in indicators['MACD_Histogram']]
        fig3.add_trace(go.Bar(
            x=indicators.index, 
            y=indicators['MACD_Histogram'],
            name='Histogram',
            marker_color=colors,
            opacity=0.5
        ))
        
        fig3.update_layout(
            title="MACD (Moving Average Convergence Divergence)",
            xaxis_title="Date",
            height=300,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, Arial", size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=False,
                showline=True,
                linecolor='rgba(255, 255, 255, 0.2)'
            ),
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.5)',
                showline=True,
                linecolor='rgba(255, 255, 255, 0.2)'
            )
        )
    
    return fig1, fig2, fig3

# Function to generate signals and insights
def generate_technical_signals(indicators, ticker):
    # Get the most recent values
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2]
    
    signals = []
    
    # SMA signals
    if use_sma:
        sma_col = f'SMA_{sma_period}'
        if latest['Close'] > latest[sma_col] and prev['Close'] <= prev[sma_col]:
            signals.append(f"⬆️ Price crossed above {sma_period}-day SMA (Bullish)")
        elif latest['Close'] < latest[sma_col] and prev['Close'] >= prev[sma_col]:
            signals.append(f"⬇️ Price crossed below {sma_period}-day SMA (Bearish)")
    else:
        if latest['SMA_50'] > latest['SMA_200'] and prev['SMA_50'] <= prev['SMA_200']:
            signals.append("⬆️ Golden Cross: 50-day SMA crossed above 200-day SMA (Strong Bullish)")
        elif latest['SMA_50'] < latest['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']:
            signals.append("⬇️ Death Cross: 50-day SMA crossed below 200-day SMA (Strong Bearish)")
    
    # RSI signals
    if use_rsi:
        if latest['RSI'] > 70:
            signals.append("⚠️ RSI above 70 - Stock may be overbought (Potential Reversal)")
        elif latest['RSI'] < 30:
            signals.append("✅ RSI below 30 - Stock may be oversold (Potential Buying Opportunity)")
        elif prev['RSI'] < 50 and latest['RSI'] > 50:
            signals.append("⬆️ RSI crossed above 50 (Bullish Momentum)")
        elif prev['RSI'] > 50 and latest['RSI'] < 50:
            signals.append("⬇️ RSI crossed below 50 (Bearish Momentum)")
    
    # MACD signals
    if use_macd:
        if latest['MACD_Line'] > latest['MACD_Signal'] and prev['MACD_Line'] <= prev['MACD_Signal']:
            signals.append("⬆️ MACD Line crossed above Signal Line (Bullish)")
        elif latest['MACD_Line'] < latest['MACD_Signal'] and prev['MACD_Line'] >= prev['MACD_Signal']:
            signals.append("⬇️ MACD Line crossed below Signal Line (Bearish)")
        
        if latest['MACD_Line'] > 0 and latest['MACD_Signal'] > 0:
            signals.append("📈 Both MACD lines above zero (Strong Bullish)")
        elif latest['MACD_Line'] < 0 and latest['MACD_Signal'] < 0:
            signals.append("📉 Both MACD lines below zero (Strong Bearish)")
    
    # Bollinger Bands signals
    if use_bollinger:
        if latest['Close'] > latest['Bollinger_Upper']:
            signals.append("⚠️ Price above upper Bollinger Band (Overbought)")
        elif latest['Close'] < latest['Bollinger_Lower']:
            signals.append("✅ Price below lower Bollinger Band (Oversold)")
        
        # Bollinger Band Squeeze (volatility indicator)
        current_band_width = latest['Bollinger_Upper'] - latest['Bollinger_Lower']
        prev_band_width = indicators['Bollinger_Upper'].iloc[-20:-1] - indicators['Bollinger_Lower'].iloc[-20:-1]
        
        if current_band_width < prev_band_width.mean() * 0.8:
            signals.append("🔄 Bollinger Band Squeeze detected (Potential Breakout Coming)")
    
    # Overall trend analysis
    sma20 = indicators['SMA_20'].iloc[-1] if 'SMA_20' in indicators else None
    sma50 = indicators['SMA_50'].iloc[-1] if 'SMA_50' in indicators else None
    sma200 = indicators['SMA_200'].iloc[-1] if 'SMA_200' in indicators else None
    
    # Count bullish vs bearish signals for overall sentiment
    bullish_count = len([s for s in signals if "Bullish" in s or "✅" in s])
    bearish_count = len([s for s in signals if "Bearish" in s or "⚠️" in s])
    
    if bullish_count > bearish_count:
        overall_sentiment = "BULLISH 📈"
    elif bearish_count > bullish_count:
        overall_sentiment = "BEARISH 📉"
    else:
        overall_sentiment = "NEUTRAL ↔️"
    
    return signals, overall_sentiment

# Main application with multi-stock support
if run_button:
    if not tickers:
        st.error("❌ Please select at least one ticker symbol.")
    else:
        with st.spinner('⏳ Processing data for selected stocks...'):
            for ticker in tickers:
                df, company_name = get_stock_data(ticker, period)
                
                if df is not None:
                    st.markdown(f"""
                    <div class="metric-card glow-effect">
                        <h2>📊 {company_name} ({ticker}) Analysis</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    tab1, tab2, tab3 = st.tabs([f"📈 {ticker} Data", f"📉 {ticker} Indicators", f"🔮 {ticker} Predictions"])
                    
                    with tab1:
                        st.subheader("Historical Price Data")
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                          vertical_spacing=0.1,
                                          row_heights=[0.7, 0.3],
                                          subplot_titles=("Price", "Volume"))
                        
                        # Ad# Add OHLC candlestick chart
                        fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name="OHLC",
                            increasing_line_color='#4CAF50',
                            decreasing_line_color='#F44336'
                        ), row=1, col=1)
                        
                        # Add volume bar chart
                        colors = ['#4CAF50' if row['Close'] >= row['Open'] else '#F44336' for _, row in df.iterrows()]
                        fig.add_trace(go.Bar(
                            x=df.index,
                            y=df['Volume'],
                            marker_color=colors,
                            name="Volume",
                            opacity=0.7
                        ), row=2, col=1)
                        
                        # Update layout
                        fig.update_layout(
                            height=600,
                            showlegend=False,
                            title_text=f"{company_name} ({ticker}) Historical Data",
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Segoe UI, Arial", size=12),
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis_rangeslider_visible=False,
                            xaxis=dict(
                                gridcolor='rgba(255,255,255,0.1)',
                                zeroline=False
                            ),
                            yaxis=dict(
                                gridcolor='rgba(255,255,255,0.1)',
                                zeroline=False
                            ),
                            xaxis2=dict(
                                gridcolor='rgba(255,255,255,0.1)',
                                zeroline=False
                            ),
                            yaxis2=dict(
                                gridcolor='rgba(255,255,255,0.1)',
                                zeroline=False
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Display recent price stats
                        price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
                        price_color = "#4CAF50" if price_change >= 0 else "#F44336"

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Latest Close</h4>
                                <p style="font-size: 24px; font-weight: bold;">${df['Close'].iloc[-1]:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Daily Change</h4>
                                <p style="font-size: 24px; font-weight: bold; color: {price_color};">{price_change:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>52-Week High</h4>
                                <p style="font-size: 24px; font-weight: bold;">${df['High'].max():.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>52-Week Low</h4>
                                <p style="font-size: 24px; font-weight: bold;">${df['Low'].min():.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    with tab2:
                        indicators = calculate_technical_indicators(df)
                        
                        # Plot technical indicators
                        price_ma_fig, rsi_fig, macd_fig = plot_technical_indicators(df, indicators, ticker)
                        
                        # Display the figures
                        st.plotly_chart(price_ma_fig, use_container_width=True)
                        
                        if use_rsi:
                            st.plotly_chart(rsi_fig, use_container_width=True)
                        
                        if use_macd:
                            st.plotly_chart(macd_fig, use_container_width=True)
                        
                        # Generate technical signals
                        signals, sentiment = generate_technical_signals(indicators, ticker)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Current Technical Signals</h4>
                            <p style="font-size: 20px; font-weight: bold; margin-bottom: 15px;">Overall Sentiment: {sentiment}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if signals:
                            st.write("Key Signals:")
                            for signal in signals:
                                st.write(f"• {signal}")
                        else:
                            st.write("No significant technical signals detected at this time.")

                    with tab3:
                        st.subheader("LSTM Price Prediction")
                        
                        with st.spinner(f"⏳ Training LSTM model for {ticker}..."):
                            # Lookback period for LSTM
                            lookback_days = 60
                            
                            # Prepare data for LSTM
                            X_train, y_train, X_val, y_val, scaler, scaled_data = prepare_data(df, lookback_days)
                            
                            # Build and train LSTM model
                            model, history = build_lstm_model(X_train, y_train, X_val, y_val, lstm_units, epochs)
                            
                            # Make predictions
                            pred_future = predict_future(model, df, lookback_days, prediction_days, scaler)
                            
                            # Create future dates for prediction
                            last_date = df.index[-1]
                            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='B')
                            
                            # Plot predictions
                            fig = go.Figure()
                            
                            # Plot historical data
                            fig.add_trace(go.Scatter(
                                x=df.index[-90:],
                                y=df['Close'].values[-90:],
                                mode='lines',
                                name='Historical Prices',
                                line=dict(color='#00b4d8', width=2)
                            ))
                            
                            # Plot predicted values
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=pred_future.flatten(),
                                mode='lines',
                                name='Predicted Prices',
                                line=dict(color='#4CAF50', width=2, dash='dash')
                            ))
                            
                            # Add confidence band (simple example - could be more sophisticated)
                            pred_std = df['Close'].std() * 0.1  # Using 10% of historical std as a simple approximation
                            
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=pred_future.flatten() + pred_std,
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=pred_future.flatten() - pred_std,
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(76, 175, 80, 0.2)',
                                showlegend=False
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                height=500,
                                title=f"{ticker} - {prediction_days} Day Price Prediction",
                                hovermode="x unified",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(family="Segoe UI, Arial", size=12),
                                margin=dict(l=20, r=20, t=40, b=20),
                                xaxis=dict(
                                    gridcolor='rgba(255,255,255,0.1)',
                                    zeroline=False,
                                    title="Date"
                                ),
                                yaxis=dict(
                                    gridcolor='rgba(255,255,255,0.1)',
                                    zeroline=False,
                                    title="Price (USD)"
                                ),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Final prediction summary
                            last_price = df['Close'].iloc[-1]
                            future_price = pred_future[-1][0]
                            price_change = (future_price - last_price) / last_price * 100
                            price_color = "#4CAF50" if price_change >= 0 else "#F44336"
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Current Price</h4>
                                    <p style="font-size: 24px; font-weight: bold;">${last_price:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Predicted Price ({prediction_days} days)</h4>
                                    <p style="font-size: 24px; font-weight: bold;">${future_price:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Predicted Change</h4>
                                    <p style="font-size: 24px; font-weight: bold; color: {price_color};">{price_change:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add prediction disclaimer
                            st.info("""
                            ⚠️ **Disclaimer**: These predictions are based on historical patterns and may not accurately predict future price movements. 
                            Financial markets are influenced by many factors that cannot be predicted. This tool should not be used as the sole basis for investment decisions.
                            """)
                    
                    # Add separator between stocks
                    st.markdown("<hr>", unsafe_allow_html=True)
else:
    # Welcome message when app first loads
    st.markdown("""
    <div class="welcome-box">
        <h2>👋 Welcome to the Multi-Stock Prediction Dashboard</h2>
        <p>This dashboard allows you to analyze and predict stock prices using machine learning and technical analysis.</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Analyze multiple stocks simultaneously</li>
            <li>View interactive candlestick charts with volume data</li>
            <li>Calculate and visualize technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)</li>
            <li>Generate technical signals and trading insights</li>
            <li>Predict future prices using LSTM neural networks</li>
        </ul>
        <p>To get started, select one or more stock tickers from the sidebar and click "Run Analysis".</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Preview of what the app does
    st.markdown("""
    <div class="metric-card">
        <h3>📊 How It Works</h3>
        <p>The dashboard uses a combination of technical analysis and deep learning to analyze stock data:</p>
        <ol>
            <li><strong>Data Collection:</strong> Historical price data is fetched from Yahoo Finance.</li>
            <li><strong>Technical Analysis:</strong> Indicators are calculated to identify potential trends and signals.</li>
            <li><strong>Prediction Model:</strong> An LSTM (Long Short-Term Memory) neural network is trained on historical data.</li>
            <li><strong>Future Forecasting:</strong> The model predicts future price movements based on learned patterns.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Display sample stocks
    st.markdown("""
    <div class="metric-card">
        <h3>💼 Available Stocks</h3>
        <p>The dashboard includes data for popular technology stocks including:</p>
    </div>
    """, unsafe_allow_html=True)
    
    sample_stocks = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc. (Google)",
        "TSLA": "Tesla, Inc.",
        "NVDA": "NVIDIA Corporation",
        "AMZN": "Amazon.com, Inc.",
        "META": "Meta Platforms, Inc. (Facebook)"
    }
    
    cols = st.columns(3)
    for i, (ticker, name) in enumerate(sample_stocks.items()):
        col = cols[i % 3]
        col.markdown(f"""
        <div style="padding: 10px; border-radius: 8px; background: linear-gradient(145deg, #1e1e2d, #262636); margin-bottom: 10px;">
            <p style="font-weight: bold; margin-bottom: 5px;">{ticker}</p>
            <p style="font-size: 14px; opacity: 0.8;">{name}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style="margin-top: 30px; padding: 15px; border-radius: 8px; background-color: rgba(255, 255, 255, 0.05);">
        <p style="font-size: 12px; opacity: 0.7;">
            <strong>Disclaimer:</strong> This dashboard is for educational and demonstration purposes only. 
            The predictions and signals provided should not be considered as financial advice. 
            Always conduct your own research or consult with a financial advisor before making investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #333340;">
    <p style="opacity: 0.7; font-size: 14px;">© 2025 Stock Price Prediction Dashboard | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)