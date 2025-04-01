# Calculate strategy returns
            hist['Returns'] = hist['Close'].pct_change()
            hist['Strategy'] = hist['Position'].shift(1) * hist['Returns']
            
            # Calculate cumulative returns
            hist['Cum_Market'] = (1 + hist['Returns']).cumprod()
            hist['Cum_Strategy'] = (1 + hist['Strategy']).cumprod()
            
            # Performance metrics
            total_return = hist['Cum_Strategy'].iloc[-1] - 1
            market_return = hist['Cum_Market'].iloc[-1] - 1
            sharpe = hist['Strategy'].mean() / hist['Strategy'].std() * np.sqrt(252)
            
            return {
                "data": hist,
                "metrics": {
                    "total_return": total_return,
                    "market_return": market_return,
                    "sharpe_ratio": sharpe,
                    "strategy": "Bollinger Bands"
                }
            }
        
        return None
    except Exception as e:
        st.error(f"Error backtesting strategy: {str(e)}")
        return None

def get_economic_indicators():
    """Fetch key economic indicators using FRED API"""
    try:
        # This would normally use FRED API but using sample data for example
        indicators = {
            "GDP Growth": [2.3, 2.2, 2.0, 2.1, 2.3, 2.2, 1.8, 1.9, 0.9, -5.1, -31.2, 33.8, 4.5, 6.3, 6.7],
            "Inflation": [1.9, 2.1, 2.3, 2.0, 2.3, 2.4, 2.3, 2.0, 1.5, 0.4, 1.0, 1.3, 1.4, 1.7, 2.6],
            "Unemployment": [3.9, 3.8, 3.8, 3.7, 3.6, 3.6, 3.5, 3.5, 3.5, 4.4, 13.3, 10.2, 8.4, 6.3, 6.1],
            "Fed Rate": [1.75, 2.00, 2.25, 2.50, 2.50, 2.50, 2.25, 2.00, 1.75, 1.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        }
        
        dates = pd.date_range(end=datetime.now(), periods=15, freq='M')
        return pd.DataFrame(indicators, index=dates)
    except Exception as e:
        st.error(f"Error fetching economic indicators: {str(e)}")
        return pd.DataFrame()

def add_advanced_ui_features():
    """Add more advanced UI components"""
    # Add custom color picker for charts
    chart_colors = {
        "price": st.sidebar.color_picker("Price line color", "#1f77b4"),
        "positive": st.sidebar.color_picker("Positive values color", "#2ca02c"),
        "negative": st.sidebar.color_picker("Negative values color", "#d62728")
    }
    
    # Add custom chart settings
    chart_settings = {
        "height": st.sidebar.slider("Chart height", 400, 1000, 600),
        "show_grid": st.sidebar.checkbox("Show grid", True),
        "show_legend": st.sidebar.checkbox("Show legend", True),
        "animations": st.sidebar.checkbox("Enable animations", True)
    }
    
    # Add advanced filter options
    advanced_filters = st.sidebar.expander("Advanced Filters")
    with advanced_filters:
        price_range = st.slider("Price Range ($)", 0, 1000, (0, 500))
        market_caps = st.multiselect("Market Capitalization", 
                                     ["Mega Cap (>$200B)", "Large Cap ($10B-$200B)", 
                                      "Mid Cap ($2B-$10B)", "Small Cap ($300M-$2B)"])
        min_dividend = st.number_input("Minimum Dividend Yield (%)", 0.0, 10.0, 0.0, 0.1)
    
    return {
        "chart_colors": chart_colors,
        "chart_settings": chart_settings,
        "filters": {
            "price_range": price_range,
            "market_caps": market_caps,
            "min_dividend": min_dividend
        }
    }

def save_analysis_config(config, filename="stock_analysis_config.json"):
    """Save analysis configuration to a file"""
    try:
        with open(filename, "w") as f:
            json.dump(config, f)
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

def load_analysis_config(uploaded_file):
    """Load analysis configuration from uploaded file"""
    try:
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8")
            config = json.loads(content)
            return config
        return None
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return None

def generate_ai_insights(ticker, data):
    """Generate AI-powered insights based on financial and technical data"""
    # This would normally use an NLP model, but is simplified for example
    insights = []
    
    # Basic stock info
    stock = yf.Ticker(ticker)
    info = stock.info if hasattr(stock, 'info') else {}
    
    # Financial insights
    if "financials" in data and not data["financials"].empty:
        financials = data["financials"]
        
        # Check revenue trend
        if "Total Revenue" in financials:
            revenue = financials["Total Revenue"]
            if revenue.iloc[-1] > revenue.iloc[-2]:
                insights.append(f"✅ Revenue increased by {((revenue.iloc[-1]/revenue.iloc[-2])-1)*100:.1f}% in the most recent period")
            else:
                insights.append(f"⚠️ Revenue decreased by {((1-revenue.iloc[-1]/revenue.iloc[-2]))*100:.1f}% in the most recent period")
        
        # Check profit margins
        if "Net Income" in financials and "Total Revenue" in financials:
            net_income = financials["Net Income"]
            revenue = financials["Total Revenue"]
            margin = net_income.iloc[-1] / revenue.iloc[-1] * 100
            if margin > 15:
                insights.append(f"✅ Strong profit margin of {margin:.1f}%")
            elif margin > 0:
                insights.append(f"ℹ️ Moderate profit margin of {margin:.1f}%")
            else:
                insights.append(f"⚠️ Negative profit margin of {margin:.1f}%")
    
    # Technical insights
    if "technical" in data and not data["technical"].empty:
        technical = data["technical"]
        
        # Check RSI for overbought/oversold
        if "RSI" in technical.columns:
            rsi = technical["RSI"].iloc[-1]
            if rsi > 70:
                insights.append(f"⚠️ RSI is {rsi:.1f}, suggesting the stock may be overbought")
            elif rsi < 30:
                insights.append(f"💡 RSI is {rsi:.1f}, suggesting the stock may be oversold")
        
        # Check moving average crossovers
        if "MA50" in technical.columns and "MA200" in technical.columns:
            ma50 = technical["MA50"].iloc[-1]
            ma200 = technical["MA200"].iloc[-1]
            if ma50 > ma200:
                insights.append(f"📈 The 50-day moving average is above the 200-day moving average, suggesting bullish momentum")
            else:
                insights.append(f"📉 The 50-day moving average is below the 200-day moving average, suggesting bearish momentum")
    
    # Valuation insights
    pe_ratio = info.get('trailingPE')
    if pe_ratio:
        if pe_ratio > 30:
            insights.append(f"⚠️ P/E ratio of {pe_ratio:.1f} is relatively high, which might indicate the stock is overvalued")
        elif pe_ratio < 15:
            insights.append(f"💡 P/E ratio of {pe_ratio:.1f} is relatively low, which might indicate the stock is undervalued")
    
    # Dividend insights
    dividend_yield = info.get('dividendYield')
    if dividend_yield:
        dividend_pct = dividend_yield * 100
        if dividend_pct > 4:
            insights.append(f"💰 High dividend yield of {dividend_pct:.2f}%, which may be attractive for income investors")
        elif dividend_pct > 0:
            insights.append(f"ℹ️ Dividend yield of {dividend_pct:.2f}%")
    
    return insights

# --- Watchlist Management ---
def initialize_watchlist():
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["DIS", "SBUX", "MSFT", "TSLA", "F", "AAPL", "SPYD", "SPY", "AMZN", "DHS", "SPHD", "DIV", 
                                       "KBWD", "LCID", "PATH", "MPC", "OXY", "CMG", "META", "WDAY", "SLG", "EPD", "ET", "AMD", 
                                       "INTC", "ON", "AXON", "LEN", "NVR", "DHI", "PANW", "QQQ", "UBER", "NVDA", "AVGO", "SMCI", 
                                       "BLK", "PLTR", "MRP"]

def add_to_watchlist(ticker):
    if ticker and ticker.upper() not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker.upper())

def remove_from_watchlist(ticker):
    if ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(ticker)

# === UI Functions ===

def single_stock_analysis():
    st.sidebar.header("Stock Settings")
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", "AAPL").upper()
    
    if st.sidebar.button("Add to Watchlist"):
        add_to_watchlist(ticker)
        st.success(f"Added {ticker} to watchlist!")
    
    period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"}
    time_period = st.sidebar.selectbox("Select Time Period", list(period_map.keys()), index=3)
    period = period_map[time_period]
    
    st.sidebar.subheader("Analysis Options")
    show_technical = st.sidebar.checkbox("Show Technical Indicators", True)
    show_ratios = st.sidebar.checkbox("Show Financial Ratios", True)
    show_news = st.sidebar.checkbox("Show News & Sentiment", True)
    show_ai_insights = st.sidebar.checkbox("Show AI Insights", True)
    show_peer_comparison = st.sidebar.checkbox("Show Peer Comparison", True)
    
    tech_indicators = []
    if show_technical:
        tech_indicators = st.sidebar.multiselect("Select Technical Indicators", 
                                              ["Moving Averages", "RSI", "MACD", "Bollinger Bands", 
                                               "Stochastic Oscillator", "ADX", "On-Balance Volume"],
                                              default=["Moving Averages", "RSI"])
    
    if st.sidebar.button("Analyze Stock"):
        # Rest of the single stock analysis code...
        st.write(f"Analyzing {ticker}...")
        # This is a placeholder - the actual implementation is extensive

def portfolio_analysis():
    st.sidebar.header("Portfolio Settings")
    default_tickers = "AAPL,MSFT,GOOG,AMZN,TSLA,NVDA,META"
    portfolio_tickers = st.sidebar.text_area("Enter Ticker Symbols (comma separated):", default_tickers)
    tickers = [ticker.strip().upper() for ticker in portfolio_tickers.split(",") if ticker.strip()]
    
    period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    time_period = st.sidebar.selectbox("Select Time Period", list(period_map.keys()), index=3)
    period = period_map[time_period]
    
    st.sidebar.subheader("Analysis Options")
    normalize_prices = st.sidebar.checkbox("Normalize Prices", True)
    show_statistics = st.sidebar.checkbox("Show Statistics", True)
    show_correlation = st.sidebar.checkbox("Show Correlation", True)
    show_optimization = st.sidebar.checkbox("Show Portfolio Optimization", True)
    
    if st.sidebar.button("Analyze Portfolio"):
        # Rest of the portfolio analysis code...
        st.write(f"Analyzing portfolio with {len(tickers)} stocks...")
        # This is a placeholder - the actual implementation is extensive

def advanced_analysis():
    st.header("Advanced Stock Analysis")
    
    ticker = st.text_input("Enter Stock Ticker Symbol:", "AAPL").upper()
    
    # Analysis options
    st.subheader("Analysis Options")
    
    col1, col2 = st.columns(2)
    with col1:
        show_valuation = st.checkbox("In-depth Valuation", True)
        show_advanced_technical = st.checkbox("Advanced Technical Analysis", True)
        show_industry_comparison = st.checkbox("Industry Comparison", True)
    with col2:
        show_financial_statements = st.checkbox("Financial Statements", True)
        show_growth_metrics = st.checkbox("Growth Metrics", True)
        show_news_sentiment = st.checkbox("News & Sentiment", True)
    
    if st.button("Perform Advanced Analysis"):
        # Rest of the advanced analysis code...
        st.write(f"Performing advanced analysis on {ticker}...")
        # This is a placeholder - the actual implementation is extensive

def stock_screener_tab():
    st.header("Stock Screener")
    
    # Criteria selection
    st.subheader("Screening Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sector selection
        sectors = [
            "Technology", "Healthcare", "Financial Services", 
            "Consumer Cyclical", "Industrials", "Communication Services",
            "Consumer Defensive", "Energy", "Basic Materials", 
            "Real Estate", "Utilities"
        ]
        selected_sectors = st.multiselect("Sectors", sectors)
        
        # Valuation criteria
        max_pe = st.number_input("Maximum P/E Ratio", 0, 1000, 50)
        min_dividend = st.number_input("Minimum Dividend Yield (%)", 0.0, 20.0, 0.0, 0.1) / 100
    
    with col2:
        # Market cap criteria
        market_cap_options = {
            "Mega Cap (>$200B)": 2e11,
            "Large Cap ($10B-$200B)": 1e10,
            "Mid Cap ($2B-$10B)": 2e9,
            "Small Cap ($300M-$2B)": 3e8,
            "Micro Cap (<$300M)": 0
        }
        selected_market_caps = st.multiselect("Market Cap", list(market_cap_options.keys()))
        min_market_cap = min([market_cap_options[cap] for cap in selected_market_caps]) if selected_market_caps else 0
        
        # Performance criteria
        min_growth = st.number_input("Minimum Revenue Growth (%)", -100.0, 1000.0, 0.0, 5.0) / 100
    
    # Additional options
    with st.expander("Additional Criteria"):
        max_debt_equity = st.number_input("Maximum Debt/Equity Ratio", 0.0, 10.0, 3.0, 0.1)
        min_roe = st.number_input("Minimum Return on Equity (%)", -100.0, 100.0, 0.0, 5.0) / 100
        min_profit_margin = st.number_input("Minimum Profit Margin (%)", -100.0, 100.0, 0.0, 5.0) / 100
    
    # Limit number of stocks to screen
    max_stocks = st.slider("Maximum stocks to screen", 10, 100, 50, 10)
    
    # Create criteria dictionary
    criteria = {
        "sectors": selected_sectors,
        "max_pe": max_pe,
        "min_dividend": min_dividend,
        "min_market_cap": min_market_cap,
        "min_growth": min_growth,
        "max_debt_equity": max_debt_equity,
        "min_roe": min_roe,
        "min_profit_margin": min_profit_margin,
        "max_stocks": max_stocks
    }
    
    if st.button("Run Stock Screener"):
        # Rest of the stock screener code...
        st.write("Running stock screener...")
        # This is a placeholder - the actual implementation is extensive

def trading_strategies_tab():
    st.header("Trading Strategy Backtesting")
    
    # Strategy selection
    ticker = st.text_input("Enter Stock Ticker Symbol:", "AAPL").upper()
    
    # Strategy selection
    strategy_options = {
        "sma_crossover": "Moving Average Crossover (50 day / 200 day)",
        "rsi": "RSI Overbought/Oversold",
        "bollinger_bands": "Bollinger Bands"
    }
    selected_strategy = st.selectbox("Select Strategy to Backtest", list(strategy_options.keys()), format_func=lambda x: strategy_options[x])
    
    # Time period selection
    period_options = {"1y": "1 Year", "2y": "2 Years", "5y": "5 Years", "10y": "10 Years", "max": "Maximum Available"}
    selected_period = st.selectbox("Select Backtest Period", list(period_options.keys()), format_func=lambda x: period_options[x])
    
    # Rest of the trading strategies code...
    if st.button("Run Backtest"):
        st.write(f"Backtesting {strategy_options[selected_strategy]} on {ticker}...")
        # This is a placeholder - the actual implementation is extensive

def economic_indicators_tab():
    st.header("Economic Indicators Analysis")
    
    # Get economic indicators
    with st.spinner("Fetching economic data..."):
        # Rest of the economic indicators code...
        st.write("Analyzing economic indicators...")
        # This is a placeholder - the actual implementation is extensive

# --- Main Function ---
def main():
    # Set page config as the first Streamlit command
    st.set_page_config(page_title="Enhanced Stock Analyzer", page_icon="📈", layout="wide")
    
    # Theme handling after page config
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
    if theme.lower() != st.session_state.theme:
        st.session_state.theme = theme.lower()
        st.rerun()  # Fixed: Use st.rerun() instead of st.experimental_rerun()
    
    if st.session_state.theme == "dark":
        st.markdown("""
            <style>
            body { background-color: #1E1E1E; color: #FFFFFF; }
            .stApp { background-color: #1E1E1E; color: #FFFFFF; }
            </style>
        """, unsafe_allow_html=True)
    
    st.title("📈 Enhanced Stock Analysis Tool")
    initialize_watchlist()
    
    # Add navigation options
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", 
                              ["Single Stock Analysis", 
                               "Portfolio Analysis", 
                               "Advanced Analysis", 
                               "Stock Screener", 
                               "Trading Strategies", 
                               "Economic Indicators"])
    
    st.sidebar.subheader("Watchlist")
    watchlist_cols = st.sidebar.columns([3, 1])
    with watchlist_cols[0]:
        new_ticker = st.text_input("Add to watchlist:", "")
    with watchlist_cols[1]:
        if st.button("Add") and new_ticker:
            add_to_watchlist(new_ticker)
            st.rerun()
    
    # Show watchlist in scrollable area with pagination
    st.sidebar.markdown("### Your Watchlist")
    if len(st.session_state.watchlist) > 0:
        items_per_page = 10
        if "watchlist_page" not in st.session_state:
            st.session_state.watchlist_page = 0
            
        total_pages = (len(st.session_state.watchlist) - 1) // items_per_page + 1
        
        start_idx = st.session_state.watchlist_page * items_per_page
        end_idx = min(start_idx + items_per_page, len(st.session_state.watchlist))
        
        for ticker in st.session_state.watchlist[start_idx:end_idx]:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(ticker)
            if col2.button("X", key=f"remove_{ticker}"):
                remove_from_watchlist(ticker)
                st.rerun()
        
        if total_pages > 1:
            prev, page_num, next_page = st.sidebar.columns([1, 2, 1])
            with prev:
                if st.button("←") and st.session_state.watchlist_page > 0:
                    st.session_state.watchlist_page -= 1
                    st.rerun()
            with page_num:
                st.write(f"Page {st.session_state.watchlist_page + 1}/{total_pages}")
            with next_page:
                if st.button("→") and st.session_state.watchlist_page < total_pages - 1:
                    st.session_state.watchlist_page += 1
                    st.rerun()
    else:
        st.sidebar.info("Your watchlist is empty. Add stocks to track them.")
    
    # Route to the selected mode
    if app_mode == "Single Stock Analysis":
        single_stock_analysis()
    elif app_mode == "Portfolio Analysis":
        portfolio_analysis()
    elif app_mode == "Advanced Analysis":
        advanced_analysis()
    elif app_mode == "Stock Screener":
        stock_screener_tab()
    elif app_mode == "Trading Strategies":
        trading_strategies_tab()
    elif app_mode == "Economic Indicators":
        economic_indicators_tab()

if __name__ == "__main__":
    main()
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import json
import matplotlib.pyplot as plt
from PIL import Image
import base64
try:
    from textblob import TextBlob
except ImportError:
    # Handle the case where TextBlob isn't installed
    st.error("TextBlob is not installed. Some sentiment analysis features will not be available.")
    class TextBlob:
        def __init__(self, text):
            self.text = text
            self.sentiment = type('obj', (object,), {'polarity': 0, 'subjectivity': 0})

import requests
from io import StringIO

# === Helper Functions ===

def percentIncrease(df):
    dfPercents = {}
    for i in df:
        # Convert 'N/A' to 0 for numeric processing
        mask = df[i] == 'N/A'
        df.loc[mask, i] = 0
        # Convert all values to numeric, force non-numeric to NaN, then fill NaN with 0
        df[i] = pd.to_numeric(df[i], errors='coerce').fillna(0)
        vals = df[i].values
        minVal = min(vals) if len(vals) > 0 else 0
        if minVal == 0:
            percents = vals  # If minimum is 0, just use values as is
        else:
            percents = (vals - minVal) / abs(minVal)  # Calculate percent change from minimum
        dfPercents[i] = percents
    dfPercents = pd.DataFrame(data=dfPercents, index=df.index)
    return dfPercents

@st.cache_data(ttl=3600)
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        
        # Safely extract financial data
        ebit = financials.loc["EBIT"] if "EBIT" in financials.index else pd.Series(["N/A"])
        ebitda = financials.loc["EBITDA"] if "EBITDA" in financials.index else pd.Series(["N/A"])
        grossProfit = financials.loc["Gross Profit"] if "Gross Profit" in financials.index else pd.Series(["N/A"])
        netIncome = financials.loc["Net Income"] if "Net Income" in financials.index else pd.Series(["N/A"])
        researchAndDevelopment = financials.loc["Research And Development"] if "Research And Development" in financials.index else pd.Series(["N/A"])
        totalRevenue = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else pd.Series(["N/A"])
        
        balance_sheet = stock.balance_sheet
        ordinaryShares = balance_sheet.loc["Ordinary Shares Number"] if "Ordinary Shares Number" in balance_sheet.index else pd.Series(["N/A"])
        stockHoldersEquity = balance_sheet.loc["Stockholders Equity"] if "Stockholders Equity" in balance_sheet.index else pd.Series(["N/A"])
        totalAssets = balance_sheet.loc["Total Assets"] if "Total Assets" in balance_sheet.index else pd.Series(["N/A"])
        totalDebt = balance_sheet.loc["Total Debt"] if "Total Debt" in balance_sheet.index else pd.Series(["N/A"])
        
        stockPriceHistory = stock.history(period="5y")["Close"]
        
        # Calculate market cap with better error handling
        marketCap = []
        if not isinstance(ordinaryShares, str) and not ordinaryShares.empty:
            for day in ordinaryShares.index:
                try:
                    startDate = day - timedelta(days=3)
                    endDate = day + timedelta(days=3)
                    history_data = stock.history(start=startDate, end=endDate)
                    if not history_data.empty and "Close" in history_data.columns:
                        stockPrice = np.average(history_data["Close"].values)
                        marketCap.append(ordinaryShares.loc[day] * stockPrice)
                    else:
                        marketCap.append(0)
                except Exception:
                    marketCap.append(0)
        
        # Calculate financial ratios with error handling
        financial_ratios = {}
        if isinstance(netIncome, pd.Series) and len(marketCap) > 0 and len(netIncome) == len(marketCap):
            pe_ratio = []
            for i in range(len(netIncome)):
                if i < len(netIncome) and netIncome.iloc[i] != 0:
                    pe_ratio.append(marketCap[i] / netIncome.iloc[i])
                else:
                    pe_ratio.append(0)
            financial_ratios["P/E Ratio"] = pe_ratio
        
        if isinstance(totalDebt, pd.Series) and isinstance(stockHoldersEquity, pd.Series) and len(totalDebt) == len(stockHoldersEquity):
            debt_to_equity = []
            for i in range(len(totalDebt)):
                if i < len(stockHoldersEquity) and stockHoldersEquity.iloc[i] != 0:
                    debt_to_equity.append(totalDebt.iloc[i] / stockHoldersEquity.iloc[i])
                else:
                    debt_to_equity.append(0)
            financial_ratios["Debt to Equity"] = debt_to_equity
        
        if isinstance(netIncome, pd.Series) and isinstance(stockHoldersEquity, pd.Series) and len(netIncome) == len(stockHoldersEquity):
            roe = []
            for i in range(len(netIncome)):
                if i < len(stockHoldersEquity) and stockHoldersEquity.iloc[i] != 0:
                    roe.append(netIncome.iloc[i] / stockHoldersEquity.iloc[i] * 100)
                else:
                    roe.append(0)
            financial_ratios["ROE (%)"] = roe
        
        # Calculate company value perception (P/B ratio)
        companyValuePerception = []
        if len(marketCap) > 0 and isinstance(stockHoldersEquity, pd.Series) and len(marketCap) == len(stockHoldersEquity):
            for i in range(len(marketCap)):
                if i < len(stockHoldersEquity) and stockHoldersEquity.iloc[i] != 0:
                    companyValuePerception.append(marketCap[i] / stockHoldersEquity.iloc[i])
                else:
                    companyValuePerception.append(0)
        
        # Assemble financials dictionary
        financials = {
            "EBIT": ebit, "EBITDA": ebitda, "Gross Profit": grossProfit, "Net Income": netIncome,
            "Research And Development": researchAndDevelopment, "Total Revenue": totalRevenue,
            "Ordinary Shares": ordinaryShares, "Stockholders Equity": stockHoldersEquity,
            "Total Assets": totalAssets, "Total Debt": totalDebt, "MarketCap": marketCap
        }
        
        if companyValuePerception:
            financials["Company value perception"] = companyValuePerception
            
        # Create DataFrames with error handling
        if isinstance(ordinaryShares, pd.Series) and not ordinaryShares.empty:
            dfTicker = pd.DataFrame(data=financials, index=ordinaryShares.index)
            scaleTicker = percentIncrease(dfTicker)
        else:
            dfTicker = pd.DataFrame()
            scaleTicker = pd.DataFrame()
            
        dfRatios = pd.DataFrame(data=financial_ratios, index=ordinaryShares.index) if financial_ratios and not isinstance(ordinaryShares, str) and not ordinaryShares.empty else None
        
        # Get company name with fallback
        info = stock.info if hasattr(stock, 'info') else {}
        companyName = info.get('longName', ticker)
        
        return [financials, scaleTicker, stockPriceHistory, companyName, dfRatios]
    except Exception as e:
        return [None, None, None, f"Error: {str(e)}", None]

def generate_technical_indicators(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if len(hist) < 14:
            return None
        
        # Calculate moving averages
        hist['MA50'] = hist['Close'].rolling(window=min(50, len(hist))).mean()
        hist['MA200'] = hist['Close'].rolling(window=min(200, len(hist))).mean()
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = hist['EMA12'] - hist['EMA26']
        hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['MACD_Histogram'] = hist['MACD'] - hist['Signal']
        
        # Calculate Bollinger Bands
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['STD20'] = hist['Close'].rolling(window=20).std()
        hist['Upper_Band'] = hist['MA20'] + (hist['STD20'] * 2)
        hist['Lower_Band'] = hist['MA20'] - (hist['STD20'] * 2)
        
        return hist
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None

def get_portfolio_data(tickers, period="1y"):
    try:
        # Handle case when tickers is empty
        if not tickers:
            return None
            
        data = yf.download(tickers, period=period)
        if data.empty:
            return None
            
        if isinstance(tickers, list) and len(tickers) > 1:
            close_data = data['Close']
        else:
            # Handle single ticker case
            ticker_name = tickers if isinstance(tickers, str) else tickers[0]
            close_data = pd.DataFrame(data['Close'], columns=[ticker_name])
        return close_data
    except Exception as e:
        st.error(f"Error fetching portfolio data: {str(e)}")
        return None

# === Enhanced Functions ===

def get_additional_fundamentals(ticker):
    """Get additional fundamental metrics for deeper analysis"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get quarterly data for more granular analysis
        quarterly_financials = stock.quarterly_financials
        quarterly_balance = stock.quarterly_balance_sheet
        
        # Add more advanced metrics
        info = stock.info
        key_metrics = {
            "Forward P/E": info.get('forwardPE', 'N/A'),
            "PEG Ratio": info.get('pegRatio', 'N/A'),
            "Enterprise Value": info.get('enterpriseValue', 'N/A'),
            "EV/EBITDA": info.get('enterpriseToEbitda', 'N/A'),
            "Profit Margin": info.get('profitMargins', 'N/A'),
            "Operating Margin": info.get('operatingMargins', 'N/A'),
            "Return on Assets": info.get('returnOnAssets', 'N/A'),
            "Return on Equity": info.get('returnOnEquity', 'N/A'),
            "Revenue Growth": info.get('revenueGrowth', 'N/A'),
            "Earnings Growth": info.get('earningsGrowth', 'N/A'),
            "Cash Per Share": info.get('totalCashPerShare', 'N/A'),
            "Dividend Yield": info.get('dividendYield', 'N/A'),
            "Payout Ratio": info.get('payoutRatio', 'N/A'),
            "Beta": info.get('beta', 'N/A')
        }
        
        return {
            "quarterly_financials": quarterly_financials,
            "quarterly_balance": quarterly_balance,
            "key_metrics": key_metrics
        }
    except Exception as e:
        return {"error": str(e)}

def generate_advanced_technical(ticker, period="1y"):
    """Calculate additional technical indicators"""
    try:
        hist = generate_technical_indicators(ticker, period)
        if hist is None or len(hist) < 30:
            return None
        
        # Add Stochastic Oscillator
        high_14 = hist['High'].rolling(window=14).max()
        low_14 = hist['Low'].rolling(window=14).min()
        hist['%K'] = 100 * ((hist['Close'] - low_14) / (high_14 - low_14))
        hist['%D'] = hist['%K'].rolling(window=3).mean()
        
        # Add Average Directional Index (ADX)
        plus_dm = hist['High'].diff()
        minus_dm = hist['Low'].diff(-1).abs()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = hist['High'] - hist['Low']
        tr2 = (hist['High'] - hist['Close'].shift()).abs()
        tr3 = (hist['Low'] - hist['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        hist['ADX'] = dx.rolling(window=14).mean()
        
        # Add On-Balance Volume (OBV)
        hist['OBV'] = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum()
        
        # Add Fibonacci Retracement levels
        high = hist['Close'].max()
        low = hist['Close'].min()
        diff = high - low
        hist['Fib_23.6%'] = high - 0.236 * diff
        hist['Fib_38.2%'] = high - 0.382 * diff
        hist['Fib_50%'] = high - 0.5 * diff
        hist['Fib_61.8%'] = high - 0.618 * diff
        
        return hist
    except Exception as e:
        st.error(f"Error calculating advanced technical indicators: {str(e)}")
        return None

def optimize_portfolio(portfolio_data, risk_free_rate=0.03):
    """Optimize portfolio weights using Modern Portfolio Theory"""
    try:
        # Calculate returns and covariance matrix
        returns = portfolio_data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Number of assets
        num_assets = len(portfolio_data.columns)
        
        # Monte Carlo Simulation
        num_portfolios = 10000
        results = np.zeros((num_portfolios, 3))  # Return, Risk, Sharpe Ratio
        weights_record = []
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
            
            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_std_dev
            results[i, 2] = sharpe_ratio
        
        # Find optimal portfolio (max Sharpe ratio)
        max_sharpe_idx = np.argmax(results[:, 2])
        optimal_weights = weights_record[max_sharpe_idx]
        
        # Find min volatility portfolio
        min_vol_idx = np.argmin(results[:, 1])
        min_vol_weights = weights_record[min_vol_idx]
        
        return {
            "assets": portfolio_data.columns,
            "all_results": results,
            "optimal_weights": dict(zip(portfolio_data.columns, optimal_weights)),
            "optimal_metrics": {
                "return": results[max_sharpe_idx, 0],
                "risk": results[max_sharpe_idx, 1],
                "sharpe": results[max_sharpe_idx, 2]
            },
            "min_vol_weights": dict(zip(portfolio_data.columns, min_vol_weights)),
            "min_vol_metrics": {
                "return": results[min_vol_idx, 0],
                "risk": results[min_vol_idx, 1],
                "sharpe": results[min_vol_idx, 2]
            }
        }
    except Exception as e:
        st.error(f"Error optimizing portfolio: {str(e)}")
        return None

def stock_screener(criteria=None):
    """Screen stocks based on user-defined criteria"""
    try:
        if criteria is None:
            criteria = {}
            
        # Get list of stocks (example: S&P 500 stocks)
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_table = pd.read_html(sp500_url)
        sp500_tickers = sp500_table[0]['Symbol'].tolist()
        
        # Limit the number of stocks to screen for performance
        max_stocks = criteria.get("max_stocks", 50)
        tickers_to_screen = sp500_tickers[:max_stocks]
        
        results = []
        for ticker in tickers_to_screen:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Skip if missing key info
                if not info or "marketCap" not in info:
                    continue
                
                # Apply filters
                if criteria.get("min_market_cap") and info.get("marketCap", 0) < criteria.get("min_market_cap"):
                    continue
                if criteria.get("max_pe") and info.get("trailingPE", float('inf')) > criteria.get("max_pe"):
                    continue
                if criteria.get("min_dividend") and (info.get("dividendYield", 0) or 0) < criteria.get("min_dividend"):
                    continue
                if criteria.get("sectors") and info.get("sector") not in criteria.get("sectors"):
                    continue
                
                # Add stock to results
                results.append({
                    "ticker": ticker,
                    "name": info.get("longName", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", None),
                    "forward_pe": info.get("forwardPE", None),
                    "peg_ratio": info.get("pegRatio", None),
                    "price_to_book": info.get("priceToBook", None),
                    "dividend_yield": info.get("dividendYield", 0),
                    "price": info.get("regularMarketPrice", 0)
                })
            except Exception:
                continue
        
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error running stock screener: {str(e)}")
        return pd.DataFrame()

def get_stock_news(ticker, limit=10):
    """Get recent news for a stock with sentiment analysis"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return []
        
        # Perform simple sentiment analysis
        news_with_sentiment = []
        for item in news[:limit]:
            title = item.get('title', '')
            sentiment = TextBlob(title).sentiment
            
            news_with_sentiment.append({
                "title": title,
                "publisher": item.get('publisher', ''),
                "link": item.get('link', ''),
                "published": datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                "sentiment": sentiment.polarity,
                "subjectivity": sentiment.subjectivity
            })
        
        return news_with_sentiment
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def industry_comparison(ticker):
    """Compare stock to industry peers"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get sector and industry
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        if not sector or not industry:
            return None
        
        # Get peer tickers (simplified for example)
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_table = pd.read_html(sp500_url)
        sp500_df = sp500_table[0]
        
        # Filter for same sector and industry
        peers_df = sp500_df[sp500_df['GICS Sector'] == sector]
        peers = peers_df['Symbol'].tolist()
        
        # Limit number of peers
        if ticker in peers:
            peers.remove(ticker)
        peers = [ticker] + peers[:9]  # Current stock + up to 9 peers
        
        # Get comparable data
        data = {}
        for peer in peers:
            try:
                peer_stock = yf.Ticker(peer)
                peer_info = peer_stock.info
                
                data[peer] = {
                    "name": peer_info.get('longName', peer),
                    "market_cap": peer_info.get('marketCap', 0),
                    "pe_ratio": peer_info.get('trailingPE', None),
                    "forward_pe": peer_info.get('forwardPE', None),
                    "peg_ratio": peer_info.get('pegRatio', None),
                    "price_to_book": peer_info.get('priceToBook', None),
                    "profit_margin": peer_info.get('profitMargins', None),
                    "beta": peer_info.get('beta', None),
                    "dividend_yield": peer_info.get('dividendYield', 0),
                }
            except Exception:
                continue
        
        return pd.DataFrame(data).T
    except Exception as e:
        st.error(f"Error in industry comparison: {str(e)}")
        return None

def backtest_strategy(ticker, strategy, period="5y"):
    """Backtest a simple trading strategy"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty or len(hist) < 50:
            return None
        
        # Calculate signals based on strategy
        if strategy == "sma_crossover":
            hist['SMA50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA200'] = hist['Close'].rolling(window=200).mean()
            
            # Generate buy/sell signals
            hist['Signal'] = 0
            hist.loc[hist['SMA50'] > hist['SMA200'], 'Signal'] = 1  # Buy signal
            hist.loc[hist['SMA50'] < hist['SMA200'], 'Signal'] = -1  # Sell signal
            
            # Calculate strategy returns
            hist['Returns'] = hist['Close'].pct_change()
            hist['Strategy'] = hist['Signal'].shift(1) * hist['Returns']
            
            # Calculate cumulative returns
            hist['Cum_Market'] = (1 + hist['Returns']).cumprod()
            hist['Cum_Strategy'] = (1 + hist['Strategy']).cumprod()
            
            # Performance metrics
            total_return = hist['Cum_Strategy'].iloc[-1] - 1
            market_return = hist['Cum_Market'].iloc[-1] - 1
            sharpe = hist['Strategy'].mean() / hist['Strategy'].std() * np.sqrt(252)
            
            return {
                "data": hist,
                "metrics": {
                    "total_return": total_return,
                    "market_return": market_return,
                    "sharpe_ratio": sharpe,
                    "strategy": "SMA Crossover (50, 200)"
                }
            }
        
        elif strategy == "rsi":
            # Calculate RSI
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            # Generate buy/sell signals
            hist['Signal'] = 0
            hist.loc[hist['RSI'] < 30, 'Signal'] = 1  # Buy when oversold
            hist.loc[hist['RSI'] > 70, 'Signal'] = -1  # Sell when overbought
            
            # Fill signal (hold between buy/sell signals)
            hist['Position'] = hist['Signal'].replace(0, np.nan).ffill().fillna(0)
            
            # Calculate strategy returns
            hist['Returns'] = hist['Close'].pct_change()
            hist['Strategy'] = hist['Position'].shift(1) * hist['Returns']
            
            # Calculate cumulative returns
            hist['Cum_Market'] = (1 + hist['Returns']).cumprod()
            hist['Cum_Strategy'] = (1 + hist['Strategy']).cumprod()
            
            # Performance metrics
            total_return = hist['Cum_Strategy'].iloc[-1] - 1
            market_return = hist['Cum_Market'].iloc[-1] - 1
            sharpe = hist['Strategy'].mean() / hist['Strategy'].std() * np.sqrt(252)
            
            return {
                "data": hist,
                "metrics": {
                    "total_return": total_return,
                    "market_return": market_return,
                    "sharpe_ratio": sharpe,
                    "strategy": "RSI Overbought/Oversold"
                }
            }
        
        elif strategy == "bollinger_bands":
            # Calculate Bollinger Bands
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['STD20'] = hist['Close'].rolling(window=20).std()
            hist['Upper_Band'] = hist['MA20'] + (hist['STD20'] * 2)
            hist['Lower_Band'] = hist['MA20'] - (hist['STD20'] * 2)
            
            # Generate buy/sell signals
            hist['Signal'] = 0
            hist.loc[hist['Close'] < hist['Lower_Band'], 'Signal'] = 1  # Buy when price below lower band
            hist.loc[hist['Close'] > hist['Upper_Band'], 'Signal'] = -1  # Sell when price above upper band
            
            # Fill signal (hold between buy/sell signals)
            hist['Position'] = hist['Signal'].replace(0, np.nan).ffill().fillna(0)
            
            # Calculate strategy returns
            hist['Returns'] = hist['Close'].pct_change()
            hist['Strategy'] = hist['Position
