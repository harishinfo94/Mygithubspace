import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

def percentIncrease(df):
    dfPercents = {}
    for i in df:
        mask = df[i] == 'N/A'
        df.loc[mask, i] = 0
        df[i] = pd.to_numeric(df[i], errors='coerce').fillna(0)
        vals = df[i].values
        minVal = min(vals)
        if minVal == 0:
            percents = vals
        else:
            percents = (vals - minVal) / abs(minVal)
        dfPercents[i] = percents
    dfPercents = pd.DataFrame(data=dfPercents, index=df.index)
    return dfPercents

@st.cache_data(ttl=3600)
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        ebit = financials.loc["EBIT"] if "EBIT" in financials.index else "N/A"
        ebitda = financials.loc["EBITDA"] if "EBITDA" in financials.index else "N/A"
        grossProfit = financials.loc["Gross Profit"] if "Gross Profit" in financials.index else "N/A"
        netIncome = financials.loc["Net Income"] if "Net Income" in financials.index else "N/A"
        researchAndDevelopment = financials.loc["Research And Development"] if "Research And Development" in financials.index else "N/A"
        totalRevenue = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else "N/A"
        
        balance_sheet = stock.balance_sheet
        ordinaryShares = balance_sheet.loc["Ordinary Shares Number"] if "Ordinary Shares Number" in balance_sheet.index else "N/A"
        stockHoldersEquity = balance_sheet.loc["Stockholders Equity"] if "Stockholders Equity" in balance_sheet.index else "N/A"
        totalAssets = balance_sheet.loc["Total Assets"] if "Total Assets" in balance_sheet.index else "N/A"
        totalDebt = balance_sheet.loc["Total Debt"] if "Total Debt" in balance_sheet.index else "N/A"
        
        stockPriceHistory = stock.history(period="5y")["Close"]
        
        marketCap = []
        for day in ordinaryShares.index:
            try:
                startDate = day - timedelta(days=3)
                endDate = day + timedelta(days=3)
                stockPrice = np.average(stock.history(start=startDate, end=endDate)["Close"].values)
                marketCap.append(ordinaryShares.loc[day] * stockPrice)
            except:
                marketCap.append(0)
        
        financial_ratios = {}
        if isinstance(netIncome, pd.Series) and len(marketCap) > 0:
            pe_ratio = [marketCap[i] / netIncome.iloc[i] if netIncome.iloc[i] != 0 else 0 for i in range(len(netIncome))]
            financial_ratios["P/E Ratio"] = pe_ratio
        
        if isinstance(totalDebt, pd.Series) and isinstance(stockHoldersEquity, pd.Series):
            debt_to_equity = [totalDebt.iloc[i] / stockHoldersEquity.iloc[i] if stockHoldersEquity.iloc[i] != 0 else 0 for i in range(len(totalDebt))]
            financial_ratios["Debt to Equity"] = debt_to_equity
        
        if isinstance(netIncome, pd.Series) and isinstance(stockHoldersEquity, pd.Series):
            roe = [netIncome.iloc[i] / stockHoldersEquity.iloc[i] * 100 if stockHoldersEquity.iloc[i] != 0 else 0 for i in range(len(netIncome))]
            financial_ratios["ROE (%)"] = roe
        
        companyValuePerception = [marketCap[i] / stockHoldersEquity.iloc[i] if stockHoldersEquity.iloc[i] != 0 else 0 for i in range(len(marketCap))] if len(marketCap) > 0 and isinstance(stockHoldersEquity, pd.Series) else []
        
        financials = {
            "EBIT": ebit, "EBITDA": ebitda, "Gross Profit": grossProfit, "Net Income": netIncome,
            "Research And Development": researchAndDevelopment, "Total Revenue": totalRevenue,
            "Ordinary Shares": ordinaryShares, "Stockholders Equity": stockHoldersEquity,
            "Total Assets": totalAssets, "Total Debt": totalDebt, "MarketCap": marketCap,
            "Company value perception": companyValuePerception
        }
        
        dfTicker = pd.DataFrame(data=financials, index=ordinaryShares.index)
        dfRatios = pd.DataFrame(data=financial_ratios, index=ordinaryShares.index) if financial_ratios else None
        scaleTicker = percentIncrease(dfTicker)
        info = stock.info
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
        
        hist['MA50'] = hist['Close'].rolling(window=min(50, len(hist))).mean()
        hist['MA200'] = hist['Close'].rolling(window=min(200, len(hist))).mean()
        
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = hist['EMA12'] - hist['EMA26']
        hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['MACD_Histogram'] = hist['MACD'] - hist['Signal']
        
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
        data = yf.download(tickers, period=period)
        if isinstance(tickers, list) and len(tickers) > 1:
            close_data = data['Close']
        else:
            close_data = pd.DataFrame(data['Close'], columns=[tickers])
        return close_data
    except Exception as e:
        st.error(f"Error fetching portfolio data: {str(e)}")
        return None

# --- Watchlist Management ---
def initialize_watchlist():
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["AAPL", "MSFT"]

def add_to_watchlist(ticker):
    if ticker and ticker.upper() not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker.upper())

def remove_from_watchlist(ticker):
    if ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(ticker)

# --- Streamlit UI ---
def main():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
    if theme.lower() != st.session_state.theme:
        st.session_state.theme = theme.lower()
        st.experimental_rerun()
    
    st.set_page_config(page_title="Enhanced Stock Analyzer", page_icon="📈", layout="wide")
    if st.session_state.theme == "dark":
        st.markdown("""
            <style>
            body { background-color: #1E1E1E; color: #FFFFFF; }
            .stApp { background-color: #1E1E1E; color: #FFFFFF; }
            </style>
        """, unsafe_allow_html=True)
    
    st.title("📈 Enhanced Stock Analysis Tool")
    initialize_watchlist()
    
    st.sidebar.header("Analysis Mode")
    app_mode = st.sidebar.radio("Select Mode", ["Single Stock Analysis", "Portfolio Analysis"])
    
    st.sidebar.subheader("Watchlist")
    for ticker in st.session_state.watchlist:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(ticker)
        if col2.button("X", key=f"remove_{ticker}"):
            remove_from_watchlist(ticker)
    
    if app_mode == "Single Stock Analysis":
        single_stock_analysis()
    else:
        portfolio_analysis()

def single_stock_analysis():
    st.sidebar.header("Stock Settings")
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", "AAPL").upper()
    
    if st.sidebar.button("Add to Watchlist"):
        add_to_watchlist(ticker)
    
    period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"}
    time_period = st.sidebar.selectbox("Select Time Period", list(period_map.keys()), index=3)
    period = period_map[time_period]
    
    st.sidebar.subheader("Analysis Options")
    show_technical = st.sidebar.checkbox("Show Technical Indicators", True)
    show_ratios = st.sidebar.checkbox("Show Financial Ratios", True)
    
    if show_technical:
        tech_indicators = st.sidebar.multiselect("Select Technical Indicators", ["Moving Averages", "RSI", "MACD", "Bollinger Bands"], default=["Moving Averages", "RSI"])
    
    if st.sidebar.button("Analyze Stock"):
        with st.spinner(f"Analyzing {ticker}..."):
            financials, scale_ticker, stock_price_history, company_name, ratios_df = get_financials(ticker)
            if financials is None:
                st.error(company_name)
            else:
                tech_data = generate_technical_indicators(ticker, period) if show_technical else None
                st.header(f"{company_name} ({ticker}) Analysis")
                tab1, tab2, tab3, tab4 = st.tabs(["Stock Performance", "Financial Analysis", "Technical Indicators", "Financial Ratios"])
                
                with tab1:
                    st.subheader("Stock Price History")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_price_history.index, y=stock_price_history, name="Price", line=dict(color="blue")))
                    fig.update_layout(title=f"{time_period} Stock Price History", xaxis_title="Date", yaxis_title="Price (USD)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    buffer = io.BytesIO()
                    fig.write_image(buffer, format="png")
                    st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_price.png", "image/png")
                
                with tab2:
                    st.subheader("Financial Analysis")
                    fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Revenue & Profit", "EBIT & EBITDA", "Equity & Market Cap"])
                    
                    with fin_tab1:
                        fig = go.Figure()
                        for metric in ["Total Revenue", "Gross Profit", "Net Income"]:
                            if metric in scale_ticker.columns:
                                fig.add_trace(go.Scatter(x=scale_ticker.index, y=scale_ticker[metric], name=metric, mode='lines+markers'))
                        fig.update_layout(title="Revenue and Profit Trends", xaxis_title="Date", yaxis_title="Percentage Change
