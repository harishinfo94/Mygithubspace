import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import matplotlib.dates as mdates

# Set page configuration
st.set_page_config(
    page_title="Enhanced Stock Analysis Tool",
    page_icon="📈",
    layout="wide"
)

# Helper function to normalize data for comparison
def percentIncrease(df):
    """Normalize the data for better visualization"""
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
    
    dfPercents = pd.DataFrame(data=dfPercents, index=df.index)
    return dfPercents

# Function to fetch financial data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # Get basic info
        info = stock.info
        company_name = info.get('longName', ticker)
        
        # Financial statements
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Extract key metrics
        ebit = financials.loc["EBIT"] if "EBIT" in financials.index else pd.Series("N/A", index=financials.columns)
        ebitda = financials.loc["EBITDA"] if "EBITDA" in financials.index else pd.Series("N/A", index=financials.columns)
        gross_profit = financials.loc["Gross Profit"] if "Gross Profit" in financials.index else pd.Series("N/A", index=financials.columns)
        net_income = financials.loc["Net Income"] if "Net Income" in financials.index else pd.Series("N/A", index=financials.columns)
        rd = financials.loc["Research And Development"] if "Research And Development" in financials.index else pd.Series("N/A", index=financials.columns)
        total_revenue = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else pd.Series("N/A", index=financials.columns)
        
        # Balance sheet items
        ordinary_shares = balance_sheet.loc["Ordinary Shares Number"] if "Ordinary Shares Number" in balance_sheet.index else pd.Series("N/A", index=balance_sheet.columns)
        stockholders_equity = balance_sheet.loc["Stockholders Equity"] if "Stockholders Equity" in balance_sheet.index else pd.Series("N/A", index=balance_sheet.columns)
        total_assets = balance_sheet.loc["Total Assets"] if "Total Assets" in balance_sheet.index else pd.Series("N/A", index=balance_sheet.columns)
        total_debt = balance_sheet.loc["Total Debt"] if "Total Debt" in balance_sheet.index else pd.Series("N/A", index=balance_sheet.columns)
        
        # Cash flow items
        free_cash_flow = cash_flow.loc["Free Cash Flow"] if "Free Cash Flow" in cash_flow.index else pd.Series("N/A", index=cash_flow.columns)
        operating_cash_flow = cash_flow.loc["Operating Cash Flow"] if "Operating Cash Flow" in cash_flow.index else pd.Series("N/A", index=cash_flow.columns)
        
        # Stock price history
        stock_price_history = stock.history(period="5y")["Close"]
        
        # Calculate market cap
        market_cap = []
        for day in ordinary_shares.index:
            try:
                start_date = day - timedelta(days=3)
                end_date = day + timedelta(days=3)
                stock_price = np.average(stock.history(start=start_date, end=end_date)["Close"].values)
                market_cap.append(ordinary_shares.loc[day] * stock_price)
            except:
                market_cap.append(0)
        
        # Combine all financial data
        financials_data = {
            "EBIT": ebit,
            "EBITDA": ebitda,
            "Gross Profit": gross_profit,
            "Net Income": net_income,
            "Research And Development": rd,
            "Total Revenue": total_revenue,
            "Ordinary Shares": ordinary_shares,
            "Stockholders Equity": stockholders_equity,
            "Total Assets": total_assets,
            "Total Debt": total_debt,
            "Free Cash Flow": free_cash_flow,
            "Operating Cash Flow": operating_cash_flow,
            "MarketCap": market_cap,
        }
        
        # Calculate financial ratios
        ratios = {}
        
        # Return on Equity (ROE)
        if not isinstance(net_income.iloc[0], str) and not isinstance(stockholders_equity.iloc[0], str):
            roe = []
            for i in range(len(net_income)):
                if stockholders_equity.iloc[i] != 0:
                    roe.append(net_income.iloc[i] / stockholders_equity.iloc[i] * 100)  # as percentage
                else:
                    roe.append(0)
            ratios["Return on Equity"] = roe
            
        # Debt to Equity Ratio
        if not isinstance(total_debt.iloc[0], str) and not isinstance(stockholders_equity.iloc[0], str):
            debt_to_equity = []
            for i in range(len(total_debt)):
                if stockholders_equity.iloc[i] != 0:
                    debt_to_equity.append(total_debt.iloc[i] / stockholders_equity.iloc[i])
                else:
                    debt_to_equity.append(0)
            ratios["Debt to Equity"] = debt_to_equity
            
        # Profit Margin
        if not isinstance(net_income.iloc[0], str) and not isinstance(total_revenue.iloc[0], str):
            profit_margin = []
            for i in range(len(net_income)):
                if total_revenue.iloc[i] != 0:
                    profit_margin.append(net_income.iloc[i] / total_revenue.iloc[i] * 100)  # as percentage
                else:
                    profit_margin.append(0)
            ratios["Profit Margin"] = profit_margin
        
        # P/E Ratio (Price to Earnings)
        if len(market_cap) > 0 and not isinstance(net_income.iloc[0], str):
            pe_ratio = []
            for i in range(len(market_cap)):
                if net_income.iloc[i] != 0:
                    pe_ratio.append(market_cap[i] / net_income.iloc[i])
                else:
                    pe_ratio.append(0)
            ratios["P/E Ratio"] = pe_ratio
        
        # Company Value Perception (P/B Ratio)
        if len(market_cap) > 0 and not isinstance(stockholders_equity.iloc[0], str):
            company_value_perception = []
            for i in range(len(market_cap)):
                if stockholders_equity.iloc[i] != 0:
                    company_value_perception.append(market_cap[i] / stockholders_equity.iloc[i])
                else:
                    company_value_perception.append(0)
            ratios["P/B Ratio"] = company_value_perception
            financials_data["Company value perception"] = company_value_perception
            
        # Current information
        current_info = {
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "beta": info.get("beta", 0),
            "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            "trailing_pe": info.get("trailingPE", 0),
            "forward_pe": info.get("forwardPE", 0),
            "price_to_book": info.get("priceToBook", 0),
            "52_week_high": info.get("fiftyTwoWeekHigh", 0),
            "52_week_low": info.get("fiftyTwoWeekLow", 0)
        }
        
        # Create dataframe and scale for visualization
        df_ticker = pd.DataFrame(data=financials_data, index=ordinary_shares.index)
        scale_ticker = percentIncrease(df_ticker)
        
        ratios_df = pd.DataFrame(data=ratios, index=ordinary_shares.index) if ratios else None
        
        return {
            "financials": financials_data,
            "ratios": ratios,
            "scaled_data": scale_ticker,
            "stock_price": stock_price_history,
            "company_name": company_name,
            "current_info": current_info,
            "ratios_df": ratios_df
        }
    
    except Exception as e:
        return {"error": f"Error: An unexpected issue occurred with '{ticker}': {str(e)}"}

# Function to generate technical indicators
@st.cache_data(ttl=3600)
def generate_technical_indicators(ticker, start_date=None, end_date=None):
    """Calculate technical indicators for the stock"""
    try:
        stock = yf.Ticker(ticker)
        
        if start_date and end_date:
            hist = stock.history(start=start_date, end=end_date)
        else:
            hist = stock.history(period="1y")
        
        if len(hist) < 14:  # Need at least 14 days for RSI
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
        window = 20
        hist['MA20'] = hist['Close'].rolling(window=window).mean()
        hist['STD20'] = hist['Close'].rolling(window=window).std()
        hist['Upper_Band'] = hist['MA20'] + (hist['STD20'] * 2)
        hist['Lower_Band'] = hist['MA20'] - (hist['STD20'] * 2)
        
        # Calculate ATR (Average True Range)
        high_low = hist['High'] - hist['Low']
        high_close = abs(hist['High'] - hist['Close'].shift())
        low_close = abs(hist['Low'] - hist['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        hist['ATR'] = true_range.rolling(14).mean()
        
        return hist
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None

# Function to get multiple stock prices for portfolio analysis
@st.cache_data(ttl=3600)
def get_portfolio_data(tickers, start_date=None, end_date=None, period="1y"):
    """Get price data for multiple tickers"""
    try:
        if start_date and end_date:
            data = yf.download(tickers, start=start_date, end=end_date)
        else:
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

# Function to calculate portfolio performance metrics
def calculate_portfolio_metrics(portfolio_data):
    """Calculate performance metrics for a portfolio"""
    metrics = {}
    
    if portfolio_data is None or portfolio_data.empty:
        return metrics
        
    # Calculate daily returns
    daily_returns = portfolio_data.pct_change().dropna()
    
    # Check if we have enough data
    if len(daily_returns) < 20:
        return metrics
    
    # For each stock
    for column in daily_returns.columns:
        stock_returns = daily_returns[column]
        
        # Calculate annualized volatility (standard deviation of returns * sqrt(252))
        volatility = stock_returns.std() * np.sqrt(252)
        
        # Calculate annualized return
        annual_return = stock_returns.mean() * 252
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Calculate maximum drawdown
        cum_returns = (1 + stock_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Store metrics
        metrics[column] = {
            "annualized_return": annual_return * 100,  # as percentage
            "annualized_volatility": volatility * 100,  # as percentage
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown * 100,  # as percentage
            "total_return": (portfolio_data[column].iloc[-1] / portfolio_data[column].iloc[0] - 1) * 100  # as percentage
        }
    
    return metrics

# Main UI function
def main():
    # Title and description
    st.title("📈 Enhanced Stock Analysis Tool")
    st.markdown("Analyze stocks with technical indicators, financial ratios, and portfolio comparison")
    
    # Sidebar for app mode selection
    st.sidebar.header("Analysis Mode")
    app_mode = st.sidebar.radio("Select Mode", ["Single Stock Analysis", "Portfolio Analysis"])
    
    if app_mode == "Single Stock Analysis":
        single_stock_analysis()
    else:
        portfolio_analysis()

# Single Stock Analysis UI
def single_stock_analysis():
    # Sidebar controls
    st.sidebar.header("Stock Settings")
    
    # Stock selection
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", "AAPL")
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    use_custom_dates = st.sidebar.checkbox("Use Custom Date Range")
    
    if use_custom_dates:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", date.today() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", date.today())
    else:
        time_period = st.sidebar.selectbox(
            "Select Time Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Max"],
            index=3
        )
        # Convert to yfinance format
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
        start_date = None
        end_date = None
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_technical = st.sidebar.checkbox("Technical Indicators", True)
    show_financials = st.sidebar.checkbox("Financial Analysis", True)
    show_ratios = st.sidebar.checkbox("Financial Ratios", True)
    
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
            # Loading indicator
            with st.spinner(f"Analyzing {ticker}..."):
                # Fetch financial data
                stock_data = get_financials(ticker)
                
                if "error" in stock_data:
                    st.error(stock_data["error"])
                else:
                    # Fetch technical data with date range if specified
                    if use_custom_dates:
                        tech_data = generate_technical_indicators(ticker, start_date, end_date)
                        # Also update stock price data for the custom range
                        stock = yf.Ticker(ticker)
                        stock_price_history = stock.history(start=start_date, end=end_date)["Close"]
                    else:
                        tech_data = generate_technical_indicators(ticker)
                        stock_price_history = stock_data["stock_price"]
                    
                    # Display company info
                    display_company_info(stock_data)
                    
                    # Create main tabs
                    tab_list = ["Stock Performance"]
                    if show_technical:
                        tab_list.append("Technical Analysis")
                    if show_financials:
                        tab_list.append("Financial Analysis")
                    if show_ratios:
                        tab_list.append("Financial Ratios")
                    
                    tabs = st.tabs(tab_list)
                    
                    # Stock Performance Tab
                    with tabs[0]:
                        display_stock_performance(stock_data, stock_price_history)
                    
                    # Technical Analysis Tab
                    if show_technical:
                        with tabs[tab_list.index("Technical Analysis")]:
                            if tech_data is not None and not tech_data.empty:
                                display_technical_analysis(tech_data, stock_data["company_name"], tech_indicators)
                            else:
                                st.warning("Not enough data for technical analysis. Try extending the date range.")
                    
                    # Financial Analysis Tab
                    if show_financials:
                        with tabs[tab_list.index("Financial Analysis")]:
                            display_financial_analysis(stock_data)
                    
                    # Financial Ratios Tab
                    if show_ratios:
                        with tabs[tab_list.index("Financial Ratios")]:
                            display_financial_ratios(stock_data)
        else:
            st.error("Please enter a valid stock ticker.")

# Portfolio Analysis UI
def portfolio_analysis():
    st.sidebar.header("Portfolio Settings")
    
    # Portfolio tickers
    default_tickers = "AAPL,MSFT,GOOG,AMZN"
    portfolio_tickers = st.sidebar.text_area("Enter Ticker Symbols (comma separated):", default_tickers)
    
    # Clean and split the tickers
    tickers = [ticker.strip().upper() for ticker in portfolio_tickers.split(",") if ticker.strip()]
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    use_custom_dates = st.sidebar.checkbox("Use Custom Date Range", value=False)
    
    if use_custom_dates:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", date.today() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", date.today())
        period = None
    else:
        time_period = st.sidebar.selectbox(
            "Select Time Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
            index=3
        )
        # Convert to yfinance format
        period_map = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        period = period_map[time_period]
        start_date = None
        end_date = None
    
    # Analysis options
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["Price Comparison", "Performance Metrics", "Correlation Analysis"]
    )
    
    # Normalize prices option
    normalize_prices = st.sidebar.checkbox("Normalize Prices", value=True)
    
    # Analyze button
    if st.sidebar.button("Analyze Portfolio"):
        if tickers:
            with st.spinner("Analyzing portfolio..."):
                # Fetch portfolio data
                if use_custom_dates:
                    portfolio_data = get_portfolio_data(tickers, start_date, end_date)
                else:
                    portfolio_data = get_portfolio_data(tickers, period=period)
                
                if portfolio_data is None or portfolio_data.empty:
                    st.error("Could not fetch portfolio data. Please check the ticker symbols.")
                else:
                    st.header("Portfolio Analysis")
                    st.write(f"Analyzing {len(tickers)} stocks: {', '.join(tickers)}")
                    
                    if analysis_type == "Price Comparison":
                        display_portfolio_price_comparison(portfolio_data, normalize_prices)
                    elif analysis_type == "Performance Metrics":
                        display_portfolio_performance_metrics(portfolio_data)
                    else:  # Correlation Analysis
                        display_portfolio_correlation(portfolio_data)
        else:
            st.error("Please enter at least one ticker symbol.")

# Display company information
def display_company_info(stock_data):
    company_name = stock_data["company_name"]
    current_info = stock_data["current_info"]
    
    # Header with company info
    st.header(f"{company_name} ({current_info['sector']} - {current_info['industry']})")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Market Cap", f"${current_info['market_cap']/1e9:.2f}B" if current_info['market_cap'] else "N/A")
    
    with col2:
        st.metric("P/E (TTM)", f"{current_info['trailing_pe']:.2f}" if current_info['trailing_pe'] else "N/A")
    
    with col3:
        st.metric("Dividend Yield", f"{current_info['dividend_yield']:.2f}%" if current_info['dividend_yield'] else "0.00%")
    
    with col4:
        st.metric("Beta", f"{current_info['beta']:.2f}" if current_info['beta'] else "N/A")

# Display stock performance analysis
def display_stock_performance(stock_data, stock_price_history):
    company_name = stock_data["company_name"]
    
    # Stock price chart
    st.subheader("Stock Price History")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_price_history.index, stock_price_history, 'b-', linewidth=2)
    ax.set_title(f"Stock Price History for {company_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price (USD)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    st.pyplot(fig)
    
    # Calculate basic stats
    if len(stock_price_history) > 0:
        current_price = stock_price_history.iloc[-1]
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 1 month return
            month_ago_idx = max(0, len(stock_price_history) - 22)
            month_ago_price = stock_price_history.iloc[month_ago_idx]
            month_change = ((current_price / month_ago_price) - 1) * 100
            st.metric("1-Month Return", f"{month_change:.2f}%", 
                     delta=f"{month_change:.2f}%")
        
        with col2:
            # YTD return
            start_of_year = datetime(stock_price_history.index[-1].year, 1, 1)
            ytd_idx = stock_price_history.index.get_indexer([start_of_year], method='nearest')[0]
            ytd_price = stock_price_history.iloc[ytd_idx]
            ytd_change = ((current_price / ytd_price) - 1) * 100
            st.metric("YTD Return", f"{ytd_change:.2f}%", 
                     delta=f"{ytd_change:.2f}%")
        
        with col3:
            # 1 year return (or max available)
            year_ago_idx = max(0, len(stock_price_history) - 252)
            year_ago_price = stock_price_history.iloc[year_ago_idx]
            year_change = ((current_price / year_ago_price) - 1) * 100
            st.metric("1-Year Return", f"{year_change:.2f}%", 
                     delta=f"{year_change:.2f}%")

# Display technical analysis
def display_technical_analysis(tech_data, company_name, selected_indicators):
    st.subheader("Technical Indicators")
    
    # Moving Averages
    if "Moving Averages" in selected_indicators:
        st.write("### Price and Moving Averages")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(tech_data.index, tech_data['Close'], label='Price', linewidth=2)
        ax.plot(tech_data.index, tech_data['MA50'], label='50-Day MA', linewidth=1.5)
        ax.plot(tech_data.index, tech_data['MA200'], label='200-Day MA', linewidth=1.5)
        ax.set_title(f"Price and Moving Averages for {company_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gcf().autofmt_xdate()
        
        st.pyplot(fig)
        
        # Trading signals based on moving averages
        last_close = tech_data['Close'].iloc[-1]
        last_ma50 = tech_data['MA50'].iloc[-1]
        last_ma200 = tech_data['MA200'].iloc[-1]
        
        signal_col1, signal_col2 = st.columns(2)
        
        with signal_col1:
            # Golden Cross / Death Cross
            ma50_cross_ma200 = (tech_data['MA50'].shift(1) <= tech_data['MA200'].shift(1)) & (tech_data['MA50'] > tech_data['MA200'])
            ma50_uncross_ma200 = (tech_data['MA50'].shift(1) >= tech_data['MA200'].shift(1)) & (tech_data['MA50'] < tech_data['MA200'])
            
            golden_cross = ma50_cross_ma200.iloc[-20:].any()
            death_cross = ma50_uncross_ma200.iloc[-20:].any()
            
            if golden_cross:
                st.success("Recent Golden Cross detected (50-day MA crossed above 200-day MA)")
            elif death_cross:
                st.warning("Recent Death Cross detected (50-day MA crossed below 200-day MA)")
        
        with signal_col2:
            # Price relationship to MAs
            if last_close > last_ma50 and last_close > last_ma200:
                st.success("Price is above both 50-day and 200-day Moving Averages (Bullish)")
            elif last_close < last_ma50 and last_close < last_ma200:
                st.warning("Price is below both 50-day and 200-day Moving Averages (Bearish)")
            elif last_close > last_ma50 and last_close < last_ma200:
                st.info("Price is above 50-day MA but below 200-day MA (Mixed)")
            else:
                st.info("Price is below 50-day MA but above 200-day MA (Mixed)")
    
    # RSI
    if "RSI" in selected_indicators:
        st.write("### Relative Strength Index (RSI)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tech_data.index, tech_data['RSI'], color='purple', linewidth=1.5)
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.7)
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.7)
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.2)
        ax.set_title(f"RSI for {company_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("RSI")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gcf().autofmt_xdate()
        
        st.pyplot(fig)
        
        # RSI interpretation
        latest_rsi = tech_data['RSI'].iloc[-1]
        
        st.metric("Current RSI", f"{latest_rsi:.2f}")
        
        if latest_rsi > 70:
            st.warning("RSI above 70 indicates **overbought** conditions. The stock may be overvalued and could experience a pullback.")
        elif latest_rsi < 30:
            st.success("RSI below 30 indicates **oversold** conditions. The stock may be undervalued and could experience a bounce.")
        else:
            st.info(f"RSI at {latest_rsi:.2f} is in the neutral zone (between 30-70).")
    
    # MACD
    if "MACD" in selected_indicators:
        st.write("### Moving Average Convergence Divergence (MACD)")
        
        # Create figure with two subplots (MACD on top, histogram on bottom)
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
        pos_hist = histogram.copy()
        neg_hist = histogram.copy()
        pos_hist[pos_hist < 0] = 0
        neg_hist[neg_hist > 0] = 0
        
        ax2.bar(tech_data.index, pos_hist, color='green', alpha=0.5, label='Positive')
        ax2.bar(tech_data.index, neg_hist, color='red', alpha=0.5, label='Negative')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Histogram")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # MACD interpretation
        macd = tech_data['MACD'].iloc[-1]
        signal = tech_data['Signal'].iloc[-1]
        histogram_value = tech_data['MACD_Histogram'].iloc[-1]
        
        st.metric("MACD Value", f"{macd:.4f}", delta=f"{histogram_value:.4f}")
        
        # Check for MACD crossover
        macd_cross_signal = (tech_data['MACD'].shift(1) <= tech_data['Signal'].shift(1)) & (tech_data['MACD'] > tech_data['Signal'])
        signal_cross_macd = (tech_data['MACD'].shift(1) >= tech_data['Signal'].shift(1)) & (tech_data['MACD'] < tech_data['Signal'])
        
        bullish_cross = macd_cross_signal.iloc[-5:].any()  # Recent bullish crossover
        bearish_cross = signal_cross_macd.iloc[-5:].any()  # Recent bearish crossover
        
        if bullish_cross:
            st.success("Recent **bullish** MACD crossover detected (MACD crossed above Signal Line)")
        elif bearish_cross:
            st.warning("Recent **bearish** MACD crossover detected (MACD crossed below Signal Line)")
        
        if macd > signal:
            st.info("MACD is currently above the Signal Line, suggesting bullish momentum.")
        else:
            st.info("MACD is currently below the Signal Line, suggesting bearish momentum.")
    
    # Bollinger Bands
    if "Bollinger Bands" in selected_indicators:
        st.write("### Bollinger Bands")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(tech_data.index, tech_data['Close'], label='Price', color='blue', linewidth=1.5)
        ax.plot(tech_data.index, tech_data['MA20'], label='20-Day MA', color='orange', linewidth=1.5)
        ax.plot(tech_data.index, tech_data['Upper_Band'], label='Upper Band', color='green', linestyle='--', linewidth=1)
        ax.plot(tech_data.index, tech_data['Lower_Band'], label='Lower Band', color='red', linestyle='--', linewidth=1)
        
        # Fill between the bands
        ax.fill_between(tech_data.index, tech_data['Upper_Band'], tech_data['Lower_Band'], color='gray', alpha=0.1)
        
        ax.set_title(f"Bollinger Bands for {company_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gcf().autofmt_xdate()
        
        st.pyplot(fig)
        
        # Bollinger Bands interpretation
        latest_close = tech_data['Close'].iloc[-1]
        latest_upper = tech_data['Upper_Band'].iloc[-1]
        latest_lower = tech_data['Lower_Band'].iloc[-1]
        latest_middle = tech_data['MA20'].iloc[-1]
        
        # Calculate Bandwidth and %B
        bandwidth = (latest_upper - latest_lower) / latest_middle * 100
        percent_b = (latest_close - latest_lower) / (latest_upper - latest_lower) if (latest_upper - latest_lower) > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bandwidth (%)", f"{bandwidth:.2f}%")
        with col2:
            st.metric("%B", f"{percent_b:.2f}")
        
        if latest_close > latest_upper:
            st.warning("Price is above the upper Bollinger Band. The stock may be overbought.")
        elif latest_close < latest_lower:
            st.success("Price is below the lower Bollinger Band. The stock may be oversold.")
        else:
            position = (latest_close - latest_lower) / (latest_upper - latest_lower) * 100
            st.info(f"Price is within the Bollinger Bands, at {position:.1f}% of the band range.")
            
        # Bollinger Band squeeze (low bandwidth) indicates potential for volatility increase
        if bandwidth < 10:  # This threshold can be adjusted
            st.warning("Bollinger Band squeeze detected (low bandwidth). This may indicate a potential increase in volatility soon.")
