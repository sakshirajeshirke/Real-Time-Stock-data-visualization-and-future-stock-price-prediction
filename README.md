# Real-Time-Stock-data-visualization-and-future-stock-price-prediction
# 📈 Multi-Stock Prediction Dashboard

A powerful, interactive dashboard for analyzing stocks and predicting future price movements using machine learning and technical analysis.

![Dashboard Screenshot]("C:\Users\skintern\Desktop\stock\dashboard.jpg",""C:\Users\skintern\Desktop\stock\dashboard2.jpg"")
## Features

- **Multi-Stock Analysis**: Analyze multiple stocks simultaneously
- **Interactive Charts**: 
  - Candlestick charts with volume data
  - Technical indicators visualization (SMA, EMA, RSI, MACD, Bollinger Bands)
- **Technical Analysis**:
  - Generate trading signals based on indicator patterns
  - Identify market sentiment and potential entry/exit points
- **Price Prediction**:
  - LSTM neural network for time series forecasting
  - Customizable prediction horizon
  - Confidence intervals for predictions
- **Modern Dark Mode UI**:
  - Sleek, responsive design optimized for analysis
  - Interactive components with real-time updates

## Demo

![Demo GIF]("C:\Users\skintern\Desktop\stock\demo1.jpg","C:\Users\skintern\Desktop\stock\demo2.jpg","C:\Users\skintern\Desktop\stock\demo3.jpg")

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/stock-prediction-dashboard.git
   cd stock-prediction-dashboard
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run pred.py
   ```

2. The dashboard will open in your default web browser at `http://localhost:8501`

3. Select ticker symbols, time period, and configuration options in the sidebar

4. Click "Run Analysis" to generate charts, indicators, and predictions

## Configuration Options

- **Stock Tickers**: Choose from popular stocks or enter your own
- **Historical Data Period**: Select how far back to analyze (1y, 2y, 5y, 10y)
- **Prediction Horizon**: Set how many days to forecast into the future
- **LSTM Model Complexity**: Adjust the complexity of the neural network
- **Technical Indicators**: Enable/disable various technical indicators

## How It Works

1. **Data Collection**: Historical price data is fetched from Yahoo Finance API
2. **Technical Analysis**: Indicators are calculated to identify potential trends and signals
3. **Prediction Model**: An LSTM neural network is trained on historical price data
4. **Future Forecasting**: The model predicts future price movements based on learned patterns

## Technical Details

### Libraries Used

- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Plotly, Matplotlib
- **Machine Learning**: TensorFlow, Keras, Scikit-learn
- **Web Interface**: Streamlit
- **Financial Data**: yfinance

### Machine Learning Model

The price prediction uses a Long Short-Term Memory (LSTM) neural network architecture:
- Input layer with lookback period
- LSTM layers with dropout for regularization
- Dense output layer for price prediction
- Early stopping to prevent overfitting

## Limitations

- Predictions are based on historical patterns and may not accurately predict future market movements
- Market events, news, and external factors can significantly impact stock prices
- The model performs best in trending markets and may struggle during highly volatile periods
- Limited to technical analysis; fundamental analysis is not included

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Disclaimer

This dashboard is for educational and demonstration purposes only. The predictions and signals provided should not be considered financial advice. Always conduct your own research or consult with a financial advisor before making investment decisions.

