import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

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
          percents = (vals-minVal)/abs(minVal)
        dfPercents[i] = percents

    dfPercents = pd.DataFrame(data = dfPercents,
                              index = df.index)
    return dfPercents

# Set your OpenAI API key - Using Streamlit secrets for security
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except:
    OPENAI_API_KEY = None
    NEWS_API_KEY = None

# Function to fetch financial data
@st.cache_data(ttl=3600)
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        
        # Financial statement data
        ebit = financials.loc["EBIT"] if "EBIT" in financials.index else "N/A"
        ebitda = financials.loc["EBITDA"] if "EBITDA" in financials.index else "N/A"
        grossProfit = financials.loc["Gross Profit"] if "Gross Profit" in financials.index else "N/A"
        netIncome = financials.loc["Net Income"] if "Net Income" in financials.index else "N/A"
        researchAndDevelopment = financials.loc["Research And Development"] if "Research And Development" in financials.index else "N/A"
        totalRevenue = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else "N/A"
        
        # Balance sheet data
        balance_sheet = stock.balance_sheet
        ordinaryShares = balance_sheet.loc["Ordinary Shares Number"] if "Ordinary Shares Number" in balance_sheet.index else "N/A"
        stockHoldersEquity = balance_sheet.loc["Stockholders Equity"] if "Stockholders Equity" in balance_sheet.index else "N/A"
        totalAssets = balance_sheet.loc["Total Assets"] if "Total Assets" in balance_sheet.index else "N/A"
        totalDebt = balance_sheet.loc["Total Debt"] if "Total Debt" in balance_sheet.index else "N/A"
        
        # Stock price history
        stockPriceHistory = stock.history(period="5y")["Close"]
        
        # Calculate market cap
        marketCap = []
        for day in ordinaryShares.index:
            try:
                startDate = day - timedelta(days=3)
                endDate = day + timedelta(days=3)
                stockPrice = np.average(stock.history(start=startDate, end=endDate)["Close"].values)
                marketCap.append(ordinaryShares.loc[day] * stockPrice)
            except:
                marketCap.append(0)
        
        # Financial ratios
        financial_ratios = {}
        
        # P/E Ratio
        if isinstance(netIncome, pd.Series) and len(marketCap) > 0:
            pe_ratio = []
            for i in range(len(netIncome)):
                if netIncome.iloc[i] != 0:
                    pe_ratio.append(marketCap[i] / netIncome.iloc[i])
                else:
                    pe_ratio.append(0)
            financial_ratios["P/E Ratio"] = pe_ratio
        
        # Debt to Equity Ratio
        if isinstance(totalDebt, pd.Series) and isinstance(stockHoldersEquity, pd.Series):
            debt_to_equity = []
            for i in range(len(totalDebt)):
                if stockHoldersEquity.iloc[i] != 0:
                    debt_to_equity.append(totalDebt.iloc[i] / stockHoldersEquity.iloc[i])
                else:
                    debt_to_equity.append(0)
            financial_ratios["Debt to Equity"] = debt_to_equity
        
        # Return on Equity (ROE)
        if isinstance(netIncome, pd.Series) and isinstance(stockHoldersEquity, pd.Series):
            roe = []
            for i in range(len(netIncome)):
                if stockHoldersEquity.iloc[i] != 0:
                    roe.append(netIncome.iloc[i] / stockHoldersEquity.iloc[i] * 100)
                else:
                    roe.append(0)
            financial_ratios["ROE (%)"] = roe
        
        # Company value perception (P/B Ratio)
        companyValuePerception = []
        if len(marketCap) > 0 and isinstance(stockHoldersEquity, pd.Series):
            for i in range(len(marketCap)):
                if stockHoldersEquity.iloc[i] != 0:
                    companyValuePerception.append(marketCap[i] / stockHoldersEquity.iloc[i])
                else:
                    companyValuePerception.append(0)
        
        # Combine all financial data
        financials = {
            "EBIT": ebit,
            "EBITDA": ebitda,
            "Gross Profit": grossProfit,
            "Net Income": netIncome,
            "Research And Development": researchAndDevelopment,
            "Total Revenue": totalRevenue,
            "Ordinary Shares": ordinaryShares,
            "Stockholders Equity": stockHoldersEquity,
            "Total Assets": totalAssets,
            "Total Debt": totalDebt,
            "MarketCap": marketCap,
            "Company value perception": companyValuePerception
        }
        
        # Create financial data dataframe
        dfTicker = pd.DataFrame(data=financials, index=ordinaryShares.index)
        
        # Create financial ratios dataframe
        dfRatios = pd.DataFrame(data=financial_ratios, index=ordinaryShares.index) if financial_ratios else None
        
        # Scale for visualization
        scaleTicker = percentIncrease(dfTicker)
        
        # Get current company information
        info = stock.info
        companyName = info.get('longName', ticker)
        
        return [financials, scaleTicker, stockPriceHistory, companyName, dfRatios]
    
    except Exception as e:
        return [None, None, None, f"Error: An unexpected issue occurred with '{ticker}': {str(e)}", None]

# Function to generate technical indicators
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

# Function to fetch portfolio data
def get_portfolio_data(tickers, period="1y"):
    try:
        data = yf.download(tickers, period=period)
        
        # Extract closing prices
        if isinstance(tickers, list) and len(tickers) > 1:
            close_data = data['Close']
        else:
            close_data = pd.DataFrame(data['Close'])
            close_data.columns = [tickers]
        
        return close_data
    
    except Exception as e:
        st.error(f"Error fetching portfolio data: {str(e)}")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="Enhanced Stock Analyzer", page_icon="📈", layout="wide")
    
    st.title("📈 Enhanced Stock Analysis Tool")
    
    # Sidebar for app mode
    st.sidebar.header("Analysis Mode")
    app_mode = st.sidebar.radio("Select Mode", ["Single Stock Analysis", "Portfolio Analysis"])
    
    if app_mode == "Single Stock Analysis":
        single_stock_analysis()
    else:
        portfolio_analysis()

def single_stock_analysis():
    # Sidebar controls
    st.sidebar.header("Stock Settings")
    
    # Stock selection
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", "AAPL")
    
    # Date range selection
    st.sidebar.subheader("Time Period")
    time_period = st.sidebar.selectbox(
        "Select Time Period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Max"],
        index=3
    )
    
    # Map to yfinance format
    period_map = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }
    period = period_map[time_period]
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_technical = st.sidebar.checkbox("Show Technical Indicators", True)
    show_ratios = st.sidebar.checkbox("Show Financial Ratios", True)
    
    # Technical indicator selection
    if show_technical:
        tech_indicators = st.sidebar.multiselect(
            "Select Technical Indicators",
            ["Moving Averages", "RSI", "MACD", "Bollinger Bands"],
            default=["Moving Averages", "RSI"]
        )
    
    # Analyze button
    if st.sidebar.button("Analyze Stock"):
        if ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                # Fetch financial data
                financials, scale_ticker, stock_price_history, company_name, ratios_df = get_financials(ticker)
                
                if financials is None:
                    st.error(company_name)  # Display error message
                else:
                    # Fetch technical data
                    tech_data = generate_technical_indicators(ticker, period) if show_technical else None
                    
                    # Graph 1: Stock Price History
                    st.header(f"{company_name} ({ticker}) Analysis")
                    
                    # Create tabs for different analyses
                    tab1, tab2, tab3, tab4 = st.tabs(["Stock Performance", "Financial Analysis", "Technical Indicators", "Financial Ratios"])
                    
                    with tab1:
                        st.subheader("Stock Price History")
                        plt.figure(figsize=(10, 6))
                        plt.plot(stock_price_history.index, stock_price_history, 'b-', linewidth=2)
                        plt.title(f"{time_period} Stock Price History for {company_name}")
                        plt.xlabel("Date")
                        plt.ylabel("Stock Price (USD)")
                        plt.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(plt.gcf())
                        plt.clf()
                        
                        # Performance metrics
                        if len(stock_price_history) > 0:
                            current_price = stock_price_history.iloc[-1]
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # 1-month return
                                month_ago_idx = max(0, len(stock_price_history) - 22)
                                month_ago_price = stock_price_history.iloc[month_ago_idx]
                                month_change = ((current_price / month_ago_price) - 1) * 100
                                st.metric("1-Month Return", f"{month_change:.2f}%")
                            
                            with col2:
                                # 3-month return
                                three_month_idx = max(0, len(stock_price_history) - 66)
                                three_month_price = stock_price_history.iloc[three_month_idx]
                                three_month_change = ((current_price / three_month_price) - 1) * 100
                                st.metric("3-Month Return", f"{three_month_change:.2f}%")
                            
                            with col3:
                                # Total period return
                                start_price = stock_price_history.iloc[0]
                                total_change = ((current_price / start_price) - 1) * 100
                                st.metric(f"{time_period} Return", f"{total_change:.2f}%")
                    
                    with tab2:
                        st.subheader("Financial Analysis")
                        
                        # Create financial subtabs
                        fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Revenue & Profit", "EBIT & EBITDA", "Equity & Market Cap"])
                        
                        with fin_tab1:
                            # Graph: Revenue and Profit metrics
                            plt.figure(figsize=(10, 6))
                            metrics = ["Total Revenue", "Gross Profit", "Net Income"]
                            for metric in metrics:
                                if metric in scale_ticker.columns:
                                    plt.plot(scale_ticker.index, scale_ticker[metric], marker='o', label=metric)
                            plt.title(f"Revenue and Profit Trends for {company_name}")
                            plt.xlabel("Date")
                            plt.ylabel("Percentage Change")
                            plt.legend()
                            plt.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(plt.gcf())
                            plt.clf()
                            
                            st.write(
                                "- **Total Revenue**: Total amount of money from all operations before costs.\n"
                                "- **Gross Profit**: Revenue minus cost of goods sold.\n"
                                "- **Net Income**: Final profit after all expenses."
                            )
                        
                        with fin_tab2:
                            # Graph: EBIT and EBITDA metrics
                            plt.figure(figsize=(10, 6))
                            for metric in ["EBIT", "EBITDA"]:
                                if metric in scale_ticker.columns:
                                    plt.plot(scale_ticker.index, scale_ticker[metric], marker='o', label=metric)
                            plt.title(f"EBIT and EBITDA Trends for {company_name}")
                            plt.xlabel("Date")
                            plt.ylabel("Percentage Change")
                            plt.legend()
                            plt.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(plt.gcf())
                            plt.clf()
                            
                            st.write(
                                "- **EBIT**: Earnings Before Interest & Taxes. Evaluates core profitability.\n"
                                "- **EBITDA**: EBIT plus Depreciation & Amortization. Measures profitability before non-cash expenses."
                            )
                        
                        with fin_tab3:
                            # Graph: Equity and Market Cap
                            plt.figure(figsize=(10, 6))
                            for metric in ["Stockholders Equity", "MarketCap", "Ordinary Shares"]:
                                if metric in scale_ticker.columns:
                                    plt.plot(scale_ticker.index, scale_ticker[metric], marker='o', label=metric)
                            plt.title(f"Equity and Market Trends for {company_name}")
                            plt.xlabel("Date")
                            plt.ylabel("Percentage Change")
                            plt.legend()
                            plt.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(plt.gcf())
                            plt.clf()
                            
                            # Company value perception chart
                            if "Company value perception" in financials and financials["Company value perception"]:
                                plt.figure(figsize=(10, 6))
                                plt.plot(scale_ticker.index, financials["Company value perception"], 
                                        marker='o', color='purple', label="Company Value Perception")
                                plt.title(f"Company Value Perception for {company_name}")
                                plt.xlabel("Date")
                                plt.ylabel("Market Cap / Equity Ratio")
                                plt.grid(True, linestyle='--', alpha=0.7)
                                st.pyplot(plt.gcf())
                                plt.clf()
                                
                                st.write(
                                    "**Company Value Perception**: Market Cap divided by Stockholders Equity. "
                                    "Shows how the market values the company relative to its book value."
                                )
                    
                    with tab3:
                        if show_technical and tech_data is not None:
                            st.subheader("Technical Indicators")
                            
                            # Moving Averages
                            if "Moving Averages" in tech_indicators:
                                st.write("### Price and Moving Averages")
                                plt.figure(figsize=(10, 6))
                                plt.plot(tech_data.index, tech_data['Close'], label='Price', linewidth=2)
                                plt.plot(tech_data.index, tech_data['MA50'], label='50-Day MA', linewidth=1.5)
                                plt.plot(tech_data.index, tech_data['MA200'], label='200-Day MA', linewidth=1.5)
                                plt.title(f"Price and Moving Averages for {company_name}")
                                plt.xlabel("Date")
                                plt.ylabel("Price (USD)")
                                plt.legend()
                                plt.grid(True, linestyle='--', alpha=0.7)
                                st.pyplot(plt.gcf())
                                plt.clf()
                                
                                # Moving Average crossover signals
                                if len(tech_data) >= 200:
                                    last_close = tech_data['Close'].iloc[-1]
                                    last_ma50 = tech_data['MA50'].iloc[-1]
                                    last_ma200 = tech_data['MA200'].iloc[-1]
                                    
                                    # Golden/Death Cross
                                    ma50_cross_ma200 = (tech_data['MA50'].shift(1) <= tech_data['MA200'].shift(1)) & (tech_data['MA50'] > tech_data['MA200'])
                                    ma50_uncross_ma200 = (tech_data['MA50'].shift(1) >= tech_data['MA200'].shift(1)) & (tech_data['MA50'] < tech_data['MA200'])
                                    
                                    golden_cross = ma50_cross_ma200.iloc[-20:].any()
                                    death_cross = ma50_uncross_ma200.iloc[-20:].any()
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if golden_cross:
                                            st.success("Recent Golden Cross detected (50-day MA crossed above 200-day MA)")
                                        elif death_cross:
                                            st.warning("Recent Death Cross detected (50-day MA crossed below 200-day MA)")
                                    
                                    with col2:
                                        if last_close > last_ma50 and last_close > last_ma200:
                                            st.success("Price is above both moving averages (Bullish)")
                                        elif last_close < last_ma50 and last_close < last_ma200:
                                            st.warning("Price is below both moving averages (Bearish)")
                                        else:
                                            st.info("Price is between moving averages (Neutral)")
                            
                            # RSI
                            if "RSI" in tech_indicators:
                                st.write("### Relative Strength Index (RSI)")
                                plt.figure(figsize=(10, 5))
                                plt.plot(tech_data.index, tech_data['RSI'], color='purple', linewidth=1.5)
                                plt.axhline(y=70, color='r', linestyle='--', alpha=0.7)
                                plt.axhline(y=30, color='g', linestyle='--', alpha=0.7)
                                plt.axhline(y=50, color='gray', linestyle='-', alpha=0.2)
                                plt.title(f"RSI for {company_name}")
                                plt.xlabel("Date")
                                plt.ylabel("RSI")
                                plt.ylim(0, 100)
                                plt.grid(True, linestyle='--', alpha=0.7)
                                st.pyplot(plt.gcf())
                                plt.clf()
                                
                                # RSI interpretation
                                latest_rsi = tech_data['RSI'].iloc[-1]
                                st.metric("Current RSI", f"{latest_rsi:.2f}")
                                
                                if latest_rsi > 70:
                                    st.warning("RSI above 70 indicates **overbought** conditions. The stock may be overvalued.")
                                elif latest_rsi < 30:
                                    st.success("RSI below 30 indicates **oversold** conditions. The stock may be undervalued.")
                                else:
                                    st.info(f"RSI at {latest_rsi:.2f} is in the neutral zone (between 30-70).")
                            
                            # MACD
                            if "MACD" in tech_indicators:
                                st.write("### Moving Average Convergence Divergence (MACD)")
                                
                                # Create figure with two subplots
                                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
                                
                                # Plot MACD and Signal lines
                                ax1.plot(tech_data.index, tech_data['MACD'], label='MACD', color='blue', linewidth=1.5)
                                ax1.plot(tech_data.index, tech_data['Signal'], label='Signal Line', color='red', linewidth=1.5)
                                ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                                ax1.set_title(f"MACD for {company_name}")
                                ax1.set_ylabel("MACD")
                                ax1.legend()
                                ax1.grid(True, linestyle='--', alpha=0.7)
                                
                                # Plot MACD Histogram
                                histogram = tech_data['MACD_Histogram']
                                colors = ['green' if val >= 0 else 'red' for val in histogram]
                                
                                ax2.bar(tech_data.index, histogram, color=colors, alpha=0.7)
                                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                                ax2.set_xlabel("Date")
                                ax2.set_ylabel("Histogram")
                                ax2.grid(True, linestyle='--', alpha=0.7)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.clf()
                                
                                # MACD interpretation
                                macd = tech_data['MACD'].iloc[-1]
                                signal = tech_data['Signal'].iloc[-1]
                                
                                st.metric("MACD", f"{macd:.4f}")
                                st.metric("Signal", f"{signal:.4f}")
                                
                                if macd > signal:
                                    st.info("MACD is above Signal Line (Bullish momentum)")
                                else:
                                    st.info("MACD is below Signal Line (Bearish momentum)")
                            
                            # Bollinger Bands
                            if "Bollinger Bands" in tech_indicators:
                                st.write("### Bollinger Bands")
                                plt.figure(figsize=(10, 6))
                                
                                plt.plot(tech_data.index, tech_data['Close'], label='Price', color='blue', linewidth=1.5)
                                plt.plot(tech_data.index, tech_data['MA20'], label='20-Day MA', color='orange', linewidth=1.5)
                                plt.plot(tech_data.index, tech_data['Upper_Band'], label='Upper Band', color='green', linestyle='--', linewidth=1)
                                plt.plot(tech_data.index, tech_data['Lower_Band'], label='Lower Band', color='red', linestyle='--', linewidth=1)
                                
                                # Fill between the bands
                                plt.fill_between(tech_data.index, tech_data['Upper_Band'], tech_data['Lower_Band'], color='gray', alpha=0.1)
                                
                                plt.title(f"Bollinger Bands for {company_name}")
                                plt.xlabel("Date")
                                plt.ylabel("Price (USD)")
                                plt.legend()
                                plt.grid(True, linestyle='--', alpha=0.7)
                                st.pyplot(plt.gcf())
                                plt.clf()
                                
                                # Bollinger Bands interpretation
                                latest_close = tech_data['Close'].iloc[-1]
                                latest_upper = tech_data['Upper_Band'].iloc[-1]
                                latest_lower = tech_data['Lower_Band'].iloc[-1]
                                
                                st.metric("Current Price", f"${latest_close:.2f}")
                                st.metric("Upper Band", f"${latest_upper:.2f}")
                                st.metric("Lower Band", f"${latest_lower:.2f}")
                                
                                if latest_close > latest_upper:
                                    st.warning("Price is above the upper Bollinger Band. The stock may be overbought.")
                                elif latest_close < latest_lower:
                                    st.success("Price is below the lower Bollinger Band. The stock may be oversold.")
                                else:
                                    band_position = (latest_close - latest_lower) / (latest_upper - latest_lower) * 100
                                    st.info(f"Price is within the Bollinger Bands, at {band_position:.1f}% of the band range.")
                        else:
                            if show_technical:
                                st.warning("Not enough data for technical analysis. Try extending the time period.")
                            else:
                                st.info("Enable 'Show Technical Indicators' in the sidebar to view this section.")
                    
                    with tab4:
                        if show_ratios and ratios_df is not None and not ratios_df.empty:
                            st.subheader("Financial Ratios")
                            
                            # P/E Ratio and P/B Ratio
                            if "P/E Ratio" in ratios_df.columns:
                                st.write("### Valuation Ratios")
                                plt.figure(figsize=(10, 6))
                                
                                plt.plot(ratios_df.index, ratios_df["P/E Ratio"], marker='o', label="P/E Ratio", color='blue')
                                
                                if "Company value perception" in financials and financials["Company value perception"]:
                                    plt.plot(ratios_df.index, financials["Company value perception"], marker='s', label="P/B Ratio", color='green')
                                
                                plt.title(f"Valuation Ratios for {company_name}")
                                plt.xlabel("Date")
                                plt.ylabel("Ratio Value")
                                plt.legend()
                                plt.grid(True, linestyle='--', alpha=0.7)
                                st.pyplot(plt.gcf())
                                plt.clf()
                                
                                st.write(
                                    "- **P/E Ratio**: Price to Earnings ratio - measures company's current share price relative to its earnings per share.\n"
                                    "- **P/B Ratio**: Price to Book ratio - compares a company's market value to its book value."
                                )
                            
                            # ROE and Debt to Equity
                            if "ROE (%)" in ratios_df.columns or "Debt to Equity" in ratios_df.columns:
                                st.write("### Performance and Solvency Ratios")
                                plt.figure(figsize=(10, 6))
                                
                                if "ROE (%)" in ratios_df.columns:
                                    plt.plot(ratios_df.index, ratios_df["ROE (%)"], marker='o', label="Return on Equity (%)", color='purple')
                                
                                if "Debt to Equity" in ratios_df.columns:
                                    plt.plot(ratios_df.index, ratios_df["Debt to Equity"], marker='s', label="Debt to Equity", color='orange')
                                
                                plt.title(f"Performance and Solvency Ratios for {company_name}")
                                plt.xlabel("Date")
                                plt.ylabel("Ratio Value")
                                plt.legend()
                                plt.grid(True, linestyle='--', alpha=0.7)
                                st.pyplot(plt.gcf())
                                plt.clf()
                                
                                st.write(
                                    "- **Return on Equity (ROE)**: Net income as a percentage of shareholder equity. Measures profitability.\n"
                                    "- **Debt to Equity**: Total liabilities divided by shareholder equity. Measures financial leverage."
                                )
                            
                            # Raw ratio data
                            with st.expander("View Raw Ratio Data"):
                                st.dataframe(ratios_df)
                                
                        else:
                            if show_ratios:
                                st.warning("Financial ratio data is not available for this stock.")
                            else:
                                st.info("Enable 'Show Financial Ratios' in the sidebar to view this section.")
        else:
            st.error("Please enter a valid stock ticker.")

def portfolio_analysis():
    # Sidebar controls
    st.sidebar.header("Portfolio Settings")
    
    # Portfolio tickers
    default_tickers = "AAPL,MSFT,GOOG,AMZN"
    portfolio_tickers = st.sidebar.text_area("Enter Ticker Symbols (comma separated):", default_tickers)
    
    # Clean and split the tickers
    tickers = [ticker.strip().upper() for ticker in portfolio_tickers.split(",") if ticker.strip()]
    
    # Date range selection
    st.sidebar.subheader("Time Period")
    time_period = st.sidebar.selectbox(
        "Select Time Period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
        index=3
    )
    
    # Map to yfinance format
    period_map = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    period = period_map[time_period]
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    normalize_prices = st.sidebar.checkbox("Normalize Prices", True)
    show_statistics = st.sidebar.checkbox("Show Statistics", True)
    show_correlation = st.sidebar.checkbox("Show Correlation", True)
    
    # Analyze button
    if st.sidebar.button("Analyze Portfolio"):
        if tickers:
            with st.spinner("Analyzing portfolio..."):
                # Fetch portfolio data
                portfolio_data = get_portfolio_data(tickers, period)
                
                if portfolio_data is None or portfolio_data.empty:
                    st.error("Could not fetch portfolio data. Please check the ticker symbols.")
                else:
                    st.header("Portfolio Analysis")
                    st.write(f"Analyzing {len(tickers)} stocks: {', '.join(tickers)}")
                    
                    # Create tabs for different analyses
                    tab1, tab2, tab3 = st.tabs(["Price Comparison", "Performance Statistics", "Correlation Analysis"])
                    
                    with tab1:
                        st.subheader("Price Comparison")
                        
                        # Process the data for visualization
                        if normalize_prices:
                            # Normalize to the first day's price (percentage change from start)
                            normalized_data = portfolio_data.copy()
                            for column in normalized_data.columns:
                                first_price = normalized_data[column].iloc[0]
                                if first_price != 0:
                                    normalized_data[column] = (normalized_data[column] / first_price) * 100
                            
                            plot_data = normalized_data
                            y_label = "Normalized Price (%)"
                            title = "Normalized Price Comparison (First Day = 100%)"
                        else:
                            plot_data = portfolio_data
                            y_label = "Price (USD)"
                            title = "Stock Price Comparison"
                        
                        # Create plot
                        plt.figure(figsize=(12, 8))
                        
                        for column in plot_data.columns:
                            plt.plot(plot_data.index, plot_data[column], linewidth=2, label=column)
                        
                        plt.title(title)
                        plt.xlabel("Date")
                        plt.ylabel(y_label)
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.xticks(rotation=45)
                        
                        st.pyplot(plt.gcf())
                        plt.clf()
                        
                        # Calculate returns for each stock
                        returns_data = {}
                        
                        for column in portfolio_data.columns:
                            stock_data = portfolio_data[column]
                            
                            # Calculate various return periods
                            start_price = stock_data.iloc[0]
                            end_price = stock_data.iloc[-1]
                            total_return = ((end_price / start_price) - 1) * 100
                            
                            # 1-month return (or max available)
                            month_idx = max(0, len(stock_data) - 22)
                            month_return = ((end_price / stock_data.iloc[month_idx]) - 1) * 100
                            
                            returns_data[column] = {
                                "1-Month Return": f"{month_return:.2f}%",
                                "Total Period Return": f"{total_return:.2f}%",
                                "Current Price": f"${end_price:.2f}"
                            }
                        
                        # Create a table of returns
                        returns_df = pd.DataFrame(returns_data).T
                        st.dataframe(returns_df)
                        
                        # Highlight best and worst performers
                        total_returns = {col: float(returns_data[col]["Total Period Return"].strip('%')) for col in portfolio_data.columns}
                        best_performer = max(total_returns.items(), key=lambda x: x[1])
                        worst_performer = min(total_returns.items(), key=lambda x: x[1])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"Best Performer: **{best_performer[0]}** with a return of **{best_performer[1]:.2f}%**")
                        
                        with col2:
                            st.warning(f"Worst Performer: **{worst_performer[0]}** with a return of **{worst_performer[1]:.2f}%**")
                    
                    with tab2:
                        if show_statistics:
                            st.subheader("Performance Statistics")
                            
                            # Calculate performance metrics
                            metrics = {}
                            
                            for column in portfolio_data.columns:
                                stock_data = portfolio_data[column]
                                daily_returns = stock_data.pct_change().dropna()
                                
                                # Skip if not enough data
                                if len(daily_returns) < 5:
                                    continue
                                
                                # Calculate metrics
                                total_return = (stock_data.iloc[-1] / stock_data.iloc[0] - 1) * 100
                                annualized_return = daily_returns.mean() * 252 * 100
                                volatility = daily_returns.std() * np.sqrt(252) * 100
                                sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
                                
                                # Calculate maximum drawdown
                                cum_returns = (1 + daily_returns).cumprod()
                                running_max = cum_returns.cummax()
                                drawdown = (cum_returns / running_max) - 1
                                max_drawdown = drawdown.min() * 100
                                
                                metrics[column] = {
                                    "Total Return (%)": total_return,
                                    "Annual Return (%)": annualized_return,
                                    "Volatility (%)": volatility,
                                    "Sharpe Ratio": sharpe_ratio,
                                    "Max Drawdown (%)": max_drawdown
                                }
                            
                            # Create a dataframe from the metrics
                            metrics_df = pd.DataFrame(metrics).T
                            
                            # Display the dataframe
                            st.dataframe(metrics_df)
                            
                            # Description of metrics
                            st.write(
                                "- **Total Return**: Overall percentage return for the entire period.\n"
                                "- **Annual Return**: Annualized percentage return.\n"
                                "- **Volatility**: Annualized standard deviation of returns (risk).\n"
                                "- **Sharpe Ratio**: Risk-adjusted return (higher is better).\n"
                                "- **Max Drawdown**: Maximum percentage decline from a peak to a trough."
                            )
                            
                            # Risk/Return plot
                            plt.figure(figsize=(10, 8))
                            
                            for ticker in metrics_df.index:
                                annual_return = metrics_df.loc[ticker, "Annual Return (%)"]
                                volatility = metrics_df.loc[ticker, "Volatility (%)"]
                                
                                plt.scatter(volatility, annual_return, s=100, label=ticker)
                                plt.annotate(ticker, (volatility, annual_return), xytext=(5, 5), textcoords='offset points')
                            
                            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                            plt.title("Risk/Return Profile")
                            plt.xlabel("Volatility (Risk) %")
                            plt.ylabel("Annual Return %")
                            plt.grid(True, linestyle='--', alpha=0.7)
                            plt.legend()
                            
                            st.pyplot(plt.gcf())
                            plt.clf()
                        else:
                            st.info("Enable 'Show Statistics' in the sidebar to view performance statistics.")
                    
                    with tab3:
                        if show_correlation:
                            st.subheader("Correlation Analysis")
                            
                            # Calculate correlation on returns (not prices)
                            returns = portfolio_data.pct_change().dropna()
                            
                            if len(returns) < 2:
                                st.warning("Not enough data to calculate correlations.")
                            else:
                                correlation = returns.corr()
                                
                                # Display the correlation matrix
                                st.write("### Correlation Matrix")
                                st.dataframe(correlation.style.background_gradient(cmap='coolwarm'))
                                
                                # Correlation heatmap
                                plt.figure(figsize=(10, 8))
                                plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
                                plt.colorbar(label='Correlation Coefficient')
                                
                                # Add ticker labels
                                tickers_list = correlation.columns
                                plt.xticks(range(len(tickers_list)), tickers_list, rotation=45)
                                plt.yticks(range(len(tickers_list)), tickers_list)
                                
                                # Add correlation values in the cells
                                for i in range(len(tickers_list)):
                                    for j in range(len(tickers_list)):
                                        plt.text(j, i, f"{correlation.iloc[i, j]:.2f}", 
                                               ha="center", va="center", 
                                               color="white" if abs(correlation.iloc[i, j]) > 0.5 else "black")
                                
                                plt.title("Return Correlation Heatmap")
                                plt.tight_layout()
                                
                                st.pyplot(plt.gcf())
                                plt.clf()
                                
                                # Correlation interpretation
                                avg_corr = correlation.values[np.triu_indices_from(correlation.values, k=1)].mean()
                                
                                st.write(f"Portfolio Average Correlation: **{avg_corr:.4f}**")
                                
                                if avg_corr > 0.7:
                                    st.warning("High average correlation indicates low diversification. Consider adding assets from different sectors.")
                                elif avg_corr > 0.3:
                                    st.info("Moderate average correlation suggests some diversification, but there is room for improvement.")
                                else:
                                    st.success("Low average correlation indicates good diversification, which helps reduce overall risk.")
                        else:
                            st.info("Enable 'Show Correlation' in the sidebar to view correlation analysis.")
        else:
            st.error("Please enter at least one ticker symbol.")

if __name__ == "__main__":
    main()
