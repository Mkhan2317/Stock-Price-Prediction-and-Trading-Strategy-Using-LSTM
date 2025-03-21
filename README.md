

# Stock Price Prediction and Trading Strategy Using LSTM  

## Overview  
This project combines **Deep Learning** and **Financial Analysis** to predict stock prices using an **LSTM model** and develop a **trading strategy**.  
- The goal is to **predict the next day's closing price** of a stock (e.g., **AAPL**) using **historical price data** and generate **buy/sell signals** to simulate a trading strategy.  
- The project includes **data preprocessing, model building, backtesting, and performance evaluation**.  

---

## **Key Components**  

### **1. Data Collection**  
- **Objective:** Collect historical stock price data for analysis and modeling.  
- **Tools Used:** `yfinance` for fetching stock data from Yahoo Finance.  
- **Process:**  
  - The `fetch_stock_data` function downloads **Open, High, Low, Close, Volume** for a given ticker (e.g., AAPL).  
  - Data is stored in a **Pandas DataFrame** for further processing.  

### **2. Feature Engineering**  
- **Objective:** Enhance the dataset with technical indicators to improve predictions.  
- **Technical Indicators Used:**  
  - **Simple Moving Averages (SMA)** (20-day, 50-day)  
  - **Relative Strength Index (RSI)**  
  - **Moving Average Convergence Divergence (MACD)**  
  - **Bollinger Bands**  

### **3. Data Preprocessing for LSTM**  
- **Objective:** Prepare the dataset for training the LSTM model.  
- **Process:**  
  - **MinMaxScaler** scales closing prices between **0 and 1**.  
  - Creates **60-day historical sequences** to predict the next day's price.  
  - Splits dataset into **80% training, 20% testing**.  

### **4. LSTM Model Building**  
- **Objective:** Build a deep learning model to predict stock prices.  
- **Tools Used:** TensorFlow, Keras.  
- **LSTM Architecture:**  
  - Two **LSTM layers** with 50 units each.  
  - **Dropout layers (20%)** to prevent overfitting.  
  - Dense layers for output.  
  - **Adam optimizer, MSE loss function**.  

### **5. Model Training**  
- **Trained for 10 epochs** with batch size **32**.  
- **80% training data**, **20% testing data**.  

### **6. Trading Signal Generation**  
- **Buy Signal (1):** Predicted price is increasing.  
- **Sell Signal (-1):** Predicted price is decreasing.  

### **7. Backtesting the Strategy**  
- **Initial capital:** **$10,000**.  
- **Simulated trading** based on generated buy/sell signals.  

### **8. Performance Evaluation**  
**Key Metrics:**  
- **Total Return:** **69.34%**  
- **Sharpe Ratio:** **1.09** (good risk-adjusted performance)  
- **Max Drawdown:** **-18.72%**  

### **9. Visualization**  
- **Actual vs. Predicted Prices**  
- **Portfolio Growth Over Time**  
- **Buy/Sell Trading Signals**  

---

## **Project Results**  
| Metric          | Value  |  
|----------------|--------|  
| **Total Return** | 69.34%  |  
| **Sharpe Ratio** | 1.09  |  
| **Max Drawdown** | -18.72%  |  

---

## **How to Run the Project**  

### **1. Install Dependencies**  
Ensure you have Python installed, then install the required libraries:  
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn yfinance
```

### **2. Clone the Repository**  
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm
```

### **3. Run the Code**  
Execute the script in a Python environment:  
```bash
python main.py
```

---

## **Code Overview**  

### **Data Collection**  
```python
def fetch_stock_data(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return stock_data
```

### **Feature Engineering**  
```python
def add_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'], data['MACD_Signal'] = calculate_macd(data['Close'])
    return data.dropna()
```

### **LSTM Model**  
```python
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```

### **Backtesting Strategy**  
```python
def backtest_trading_strategy(signals, prices, initial_capital=10000):
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['Signal'] = signals
    portfolio['Price'] = prices
    portfolio['Holdings'] = portfolio['Signal'] * initial_capital / portfolio['Price']
    portfolio['Cash'] = initial_capital - (portfolio['Holdings'] * portfolio['Price'])
    portfolio['Total'] = portfolio['Cash'] + (portfolio['Holdings'] * portfolio['Price'])
    return portfolio
```

---

## **Key Skills Demonstrated**  
- **Deep Learning**: LSTM for time-series forecasting.  
- **Financial Analysis**: Implemented RSI, MACD, Bollinger Bands.  
- **Data Preprocessing**: MinMaxScaler, sequence generation.  
- **Backtesting**: Simulated trading strategy.  
- **Data Visualization**: Matplotlib for financial insights.  
- **Programming**: Python, TensorFlow, Pandas, NumPy, Matplotlib.  

---


## **License**  
This project is open-source and available under the **MIT License**.  

---

## **Author**  
Developed by **[MD AMIR KHAN]**  
GitHub: [Your GitHub Profile](https://github.com/Mkhan2317))  



