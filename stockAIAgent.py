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
    
    tech_indicators = []
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
                    if not stock_price_history.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=stock_price_history.index, y=stock_price_history, name="Price", line=dict(color="blue")))
                        fig.update_layout(title=f"{time_period} Stock Price History", xaxis_title="Date", yaxis_title="Price (USD)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Generate and provide download for chart
                        try:
                            buffer = io.BytesIO()
                            fig.write_image(buffer, format="png")
                            buffer.seek(0)
                            st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_price.png", "image/png")
                        except Exception as e:
                            st.warning(f"Could not generate downloadable chart: {str(e)}")
                    else:
                        st.warning("No stock price history available")
                
                with tab2:
                    st.subheader("Financial Analysis")
                    if not scale_ticker.empty:
                        fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Revenue & Profit", "EBIT & EBITDA", "Equity & Market Cap"])
                        
                        with fin_tab1:
                            available_metrics = [m for m in ["Total Revenue", "Gross Profit", "Net Income"] if m in scale_ticker.columns]
                            if available_metrics:
                                fig = go.Figure()
                                for metric in available_metrics:
                                    fig.add_trace(go.Scatter(x=scale_ticker.index, y=scale_ticker[metric], name=metric, mode='lines+markers'))
                                fig.update_layout(title="Revenue and Profit Trends", xaxis_title="Date", yaxis_title="Percentage Change")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                try:
                                    buffer = io.BytesIO()
                                    fig.write_image(buffer, format="png")
                                    buffer.seek(0)
                                    st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_revenue_profit.png", "image/png")
                                    
                                    csv_data = scale_ticker[available_metrics].to_csv()
                                    st.download_button("Download Data", csv_data, f"{ticker}_revenue_profit.csv", "text/csv")
                                except Exception as e:
                                    st.warning(f"Could not generate downloadable content: {str(e)}")
                            else:
                                st.warning("No revenue and profit data available")
                        
                        with fin_tab2:
                            available_metrics = [m for m in ["EBIT", "EBITDA"] if m in scale_ticker.columns]
                            if available_metrics:
                                fig = go.Figure()
                                for metric in available_metrics:
                                    fig.add_trace(go.Scatter(x=scale_ticker.index, y=scale_ticker[metric], name=metric, mode='lines+markers'))
                                fig.update_layout(title="EBIT and EBITDA Trends", xaxis_title="Date", yaxis_title="Percentage Change")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                try:
                                    buffer = io.BytesIO()
                                    fig.write_image(buffer, format="png")
                                    buffer.seek(0)
                                    st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_ebit_ebitda.png", "image/png")
                                except Exception as e:
                                    st.warning(f"Could not generate downloadable chart: {str(e)}")
                            else:
                                st.warning("No EBIT/EBITDA data available")
                        
                        with fin_tab3:
                            available_metrics = [m for m in ["Stockholders Equity", "MarketCap"] if m in scale_ticker.columns]
                            if available_metrics:
                                fig = go.Figure()
                                for metric in available_metrics:
                                    fig.add_trace(go.Scatter(x=scale_ticker.index, y=scale_ticker[metric], name=metric, mode='lines+markers'))
                                fig.update_layout(title="Equity and Market Trends", xaxis_title="Date", yaxis_title="Percentage Change")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                try:
                                    buffer = io.BytesIO()
                                    fig.write_image(buffer, format="png")
                                    buffer.seek(0)
                                    st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_equity_marketcap.png", "image/png")
                                except Exception as e:
                                    st.warning(f"Could not generate downloadable chart: {str(e)}")
                                
                                if "Company value perception" in financials and isinstance(financials["Company value perception"], list) and len(financials["Company value perception"]) > 0:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=scale_ticker.index, y=financials["Company value perception"], name="P/B Ratio", mode='lines+markers', line=dict(color="purple")))
                                    fig.update_layout(title="Company Value Perception", xaxis_title="Date", yaxis_title="Market Cap / Equity Ratio")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    try:
                                        buffer = io.BytesIO()
                                        fig.write_image(buffer, format="png")
                                        buffer.seek(0)
                                        st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_pb_ratio.png", "image/png")
                                    except Exception as e:
                                        st.warning(f"Could not generate downloadable chart: {str(e)}")
                            else:
                                st.warning("No equity/market cap data available")
                    else:
                        st.warning("No financial data available to analyze")
                
                with tab3:
                    if show_technical and tech_data is not None and not tech_data.empty:
                        st.subheader("Technical Indicators")
                        if "Moving Averages" in tech_indicators:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Close'], name="Price"))
                            if 'MA50' in tech_data.columns:
                                fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA50'], name="MA50"))
                            if 'MA200' in tech_data.columns:
                                fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA200'], name="MA200"))
                            fig.update_layout(title="Price and Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            try:
                                buffer = io.BytesIO()
                                fig.write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_bollinger.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                    else:
                        st.warning("No technical indicator data available")
                
                with tab4:
                    if show_ratios and ratios_df is not None and not ratios_df.empty:
                        st.subheader("Financial Ratios")
                        
                        # P/E and P/B Ratios Chart
                        has_pe = "P/E Ratio" in ratios_df.columns
                        has_pb = "Company value perception" in financials and isinstance(financials["Company value perception"], list) and len(financials["Company value perception"]) > 0
                        
                        if has_pe or has_pb:
                            fig = go.Figure()
                            if has_pe:
                                fig.add_trace(go.Scatter(x=ratios_df.index, y=ratios_df["P/E Ratio"], name="P/E Ratio", mode='lines+markers'))
                            if has_pb:
                                fig.add_trace(go.Scatter(x=ratios_df.index, y=financials["Company value perception"], name="P/B Ratio", mode='lines+markers'))
                            fig.update_layout(title="Valuation Ratios", xaxis_title="Date", yaxis_title="Ratio Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            try:
                                buffer = io.BytesIO()
                                fig.write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_valuation_ratios.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                        
                        # ROE and Debt-to-Equity Chart
                        has_roe = "ROE (%)" in ratios_df.columns
                        has_dte = "Debt to Equity" in ratios_df.columns
                        
                        if has_roe or has_dte:
                            fig = go.Figure()
                            if has_roe:
                                fig.add_trace(go.Scatter(x=ratios_df.index, y=ratios_df["ROE (%)"], name="ROE (%)", mode='lines+markers'))
                            if has_dte:
                                fig.add_trace(go.Scatter(x=ratios_df.index, y=ratios_df["Debt to Equity"], name="Debt to Equity", mode='lines+markers'))
                            fig.update_layout(title="Performance and Solvency Ratios", xaxis_title="Date", yaxis_title="Ratio Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            try:
                                buffer = io.BytesIO()
                                fig.write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_performance_ratios.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                        
                        try:
                            csv = ratios_df.to_csv()
                            st.download_button("Download Ratio Data", csv, f"{ticker}_ratios.csv", "text/csv")
                        except Exception as e:
                            st.warning(f"Could not generate downloadable data: {str(e)}")
                    else:
                        st.warning("No financial ratio data available")

def portfolio_analysis():
    st.sidebar.header("Portfolio Settings")
    default_tickers = "AAPL,MSFT,GOOG,AMZN"
    portfolio_tickers = st.sidebar.text_area("Enter Ticker Symbols (comma separated):", default_tickers)
    tickers = [ticker.strip().upper() for ticker in portfolio_tickers.split(",") if ticker.strip()]
    
    period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    time_period = st.sidebar.selectbox("Select Time Period", list(period_map.keys()), index=3)
    period = period_map[time_period]
    
    st.sidebar.subheader("Analysis Options")
    normalize_prices = st.sidebar.checkbox("Normalize Prices", True)
    show_statistics = st.sidebar.checkbox("Show Statistics", True)
    show_correlation = st.sidebar.checkbox("Show Correlation", True)
    
    if st.sidebar.button("Analyze Portfolio"):
        if not tickers:
            st.error("Please enter at least one ticker symbol.")
            return
            
        with st.spinner("Analyzing portfolio..."):
            portfolio_data = get_portfolio_data(tickers, period)
            if portfolio_data is None or portfolio_data.empty:
                st.error("Could not fetch portfolio data.")
            else:
                st.header("Portfolio Analysis")
                tab1, tab2, tab3 = st.tabs(["Price Comparison", "Performance Statistics", "Correlation Analysis"])
                
                with tab1:
                    st.subheader("Price Comparison")
                    
                    # Create plot data with error handling
                    if normalize_prices:
                        # Handle the case where some stocks might start with 0
                        first_valid = portfolio_data.iloc[0].copy()
                        for col in first_valid.index:
                            if first_valid[col] == 0:
                                mask = portfolio_data[col] != 0
                                if mask.any():
                                    first_valid[col] = portfolio_data.loc[mask, col].iloc[0]
                        
                        plot_data = portfolio_data.div(first_valid) * 100
                    else:
                        plot_data = portfolio_data
                    
                    fig = go.Figure()
                    for column in plot_data.columns:
                        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data[column], name=column))
                    y_title = "Normalized Price (%)" if normalize_prices else "Price (USD)"
                    fig.update_layout(title="Portfolio Price Comparison", xaxis_title="Date", yaxis_title=y_title)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    try:
                        buffer = io.BytesIO()
                        fig.write_image(buffer, format="png")
                        buffer.seek(0)
                        st.download_button("Download Chart", buffer.getvalue(), "portfolio_price.png", "image/png")
                        csv = portfolio_data.to_csv()
                        st.download_button("Download Data", csv, "portfolio_data.csv", "text/csv")
                    except Exception as e:
                        st.warning(f"Could not generate downloadable content: {str(e)}")
                
                with tab2:
                    if show_statistics:
                        st.subheader("Performance Statistics")
                        metrics = {}
                        
                        for column in portfolio_data.columns:
                            stock_data = portfolio_data[column]
                            # Skip if no data or less than 5 valid data points
                            if stock_data.isnull().all() or stock_data.count() < 5:
                                continue
                                
                            try:
                                daily_returns = stock_data.pct_change().dropna()
                                if len(daily_returns) < 5:
                                    continue
                                
                                # Calculate metrics with appropriate error handling
                                start_price = stock_data.iloc[0]
                                end_price = stock_data.iloc[-1]
                                
                                # Avoid division by zero
                                if start_price == 0:
                                    total_return = 0
                                else:
                                    total_return = (end_price / start_price - 1) * 100
                                
                                annualized_return = daily_returns.mean() * 252 * 100
                                volatility = daily_returns.std() * np.sqrt(252) * 100
                                sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
                                
                                # Calculate drawdown with error handling
                                cum_returns = (1 + daily_returns).cumprod()
                                running_max = cum_returns.cummax()
                                # Avoid division by zero
                                drawdown = (cum_returns / running_max) - 1
                                max_drawdown = drawdown.min() * 100
                                
                                metrics[column] = {
                                    "Total Return (%)": total_return,
                                    "Annual Return (%)": annualized_return,
                                    "Volatility (%)": volatility,
                                    "Sharpe Ratio": sharpe_ratio,
                                    "Max Drawdown (%)": max_drawdown
                                }
                            except Exception as e:
                                st.warning(f"Error calculating metrics for {column}: {str(e)}")
                        
                        if metrics:
                            metrics_df = pd.DataFrame(metrics).T
                            st.dataframe(metrics_df)
                            
                            # Create risk/return scatter plot
                            if len(metrics_df) > 0:
                                fig = go.Figure()
                                for ticker in metrics_df.index:
                                    try:
                                        volatility = metrics_df.loc[ticker, "Volatility (%)"]
                                        annual_return = metrics_df.loc[ticker, "Annual Return (%)"]
                                        if pd.notna(volatility) and pd.notna(annual_return):
                                            fig.add_trace(go.Scatter(
                                                x=[volatility], 
                                                y=[annual_return], 
                                                mode="markers+text", 
                                                name=ticker, 
                                                text=[ticker], 
                                                textposition="top center"
                                            ))
                                    except Exception:
                                        continue
                                
                                fig.update_layout(title="Risk/Return Profile", xaxis_title="Volatility (Risk) %", yaxis_title="Annual Return %")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                try:
                                    buffer = io.BytesIO()
                                    fig.write_image(buffer, format="png")
                                    buffer.seek(0)
                                    st.download_button("Download Chart", buffer.getvalue(), "portfolio_risk_return.png", "image/png")
                                    csv = metrics_df.to_csv()
                                    st.download_button("Download Data", csv, "portfolio_stats.csv", "text/csv")
                                except Exception as e:
                                    st.warning(f"Could not generate downloadable content: {str(e)}")
                        else:
                            st.warning("Could not calculate performance metrics.")
                
                with tab3:
                    if show_correlation and len(portfolio_data.columns) > 1:
                        st.subheader("Correlation Analysis")
                        try:
                            returns = portfolio_data.pct_change().dropna()
                            if len(returns) < 2:
                                st.warning("Not enough data to calculate correlations.")
                            else:
                                correlation = returns.corr()
                                st.dataframe(correlation.style.background_gradient(cmap='coolwarm'))
                                
                                fig = go.Figure(data=go.Heatmap(
                                    z=correlation.values,
                                    x=correlation.columns,
                                    y=correlation.index,
                                    colorscale='RdBu',
                                    zmin=-1,
                                    zmax=1,
                                    text=[[f"{val:.2f}" for val in row] for row in correlation.values],
                                    texttemplate="%{text}"
                                ))
                                fig.update_layout(title="Return Correlation Heatmap", width=600, height=600)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                try:
                                    buffer = io.BytesIO()
                                    fig.write_image(buffer, format="png")
                                    buffer.seek(0)
                                    st.download_button("Download Chart", buffer.getvalue(), "portfolio_correlation.png", "image/png")
                                    csv = correlation.to_csv()
                                    st.download_button("Download Data", csv, "portfolio_correlation.csv", "text/csv")
                                except Exception as e:
                                    st.warning(f"Could not generate downloadable content: {str(e)}")
                        except Exception as e:
                            st.error(f"Error calculating correlations: {str(e)}")
                    elif len(portfolio_data.columns) <= 1:
                        st.warning("Need at least two stocks to calculate correlations.")

if __name__ == "__main__":
    main().write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_ma.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                        
                        if "RSI" in tech_indicators and 'RSI' in tech_data.columns:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['RSI'], name="RSI", line=dict(color="purple")))
                            fig.add_hline(y=70, line_dash="dash", line_color="red")
                            fig.add_hline(y=30, line_dash="dash", line_color="green")
                            fig.update_layout(title="Relative Strength Index (RSI)", xaxis_title="Date", yaxis_title="RSI", yaxis_range=[0, 100])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            try:
                                buffer = io.BytesIO()
                                fig.write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_rsi.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                        
                        if "MACD" in tech_indicators and all(col in tech_data.columns for col in ['MACD', 'Signal', 'MACD_Histogram']):
                            fig = make_subplots(rows=2, cols=1, subplot_titles=("MACD", "Histogram"), vertical_spacing=0.1)
                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MACD'], name="MACD", line=dict(color="blue")), row=1, col=1)
                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Signal'], name="Signal", line=dict(color="red")), row=1, col=1)
                            
                            # Create colors for histogram
                            colors = ["green" if x >= 0 else "red" for x in tech_data['MACD_Histogram']]
                            fig.add_trace(go.Bar(x=tech_data.index, y=tech_data['MACD_Histogram'], name="Histogram", marker_color=colors), row=2, col=1)
                            fig.update_layout(height=600, title_text="MACD Analysis")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            try:
                                buffer = io.BytesIO()
                                fig.write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_macd.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                        
                        if "Bollinger Bands" in tech_indicators and all(col in tech_data.columns for col in ['Close', 'MA20', 'Upper_Band', 'Lower_Band']):
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Close'], name="Price", line=dict(color="blue")))
                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA20'], name="MA20", line=dict(color="orange")))
                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Upper_Band'], name="Upper Band", line=dict(color="green", dash="dash")))
                            fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Lower_Band'], name="Lower Band", line=dict(color="red", dash="dash")))
                            fig.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price (USD)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            try:
                                buffer = io.BytesIO()
                                fig
