import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from openai import OpenAI
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time

# Set page configuration
st.set_page_config(
    page_title="AI Stock Investment Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .positive {
        color: green;
    }
    .negative {
        color: red;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to normalize data for comparison
@st.cache_data(ttl=3600)  # Cache data for 1 hour
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
    
    dfPercents = pd.DataFrame(data=dfPercents, index=df.index)
    return dfPercents

# Function to fetch financial data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch basic info
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        
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
        total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest"] if "Total Liabilities Net Minority Interest" in balance_sheet.index else pd.Series("N/A", index=balance_sheet.columns)
        
        # Cash flow items
        free_cash_flow = cash_flow.loc["Free Cash Flow"] if "Free Cash Flow" in cash_flow.index else pd.Series("N/A", index=cash_flow.columns)
        
        # Stock price history
        stock_price_history = stock.history(period="5y")["Close"]
        
        # Calculate market cap over time
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
            "Total Liabilities": total_liabilities,
            "Free Cash Flow": free_cash_flow,
            "MarketCap": market_cap
        }
        
        # Calculate P/E ratio if available
        current_price = stock_price_history.iloc[-1] if not stock_price_history.empty else 0
        latest_eps = info.get('trailingEPS', 0)
        pe_ratio = current_price / latest_eps if latest_eps and latest_eps != 0 else 0
        
        # Calculate Company Value Perception
        if len(market_cap) > 0 and not isinstance(stockholders_equity.iloc[0], str):
            company_value_perception = [mc / se if se != 0 else 0 for mc, se in zip(market_cap, stockholders_equity)]
            financials_data["Company Value Perception"] = company_value_perception
        
        # Create dataframe and scale for visualization
        df_ticker = pd.DataFrame(data=financials_data, index=ordinary_shares.index)
        scale_ticker = percentIncrease(df_ticker)
        
        # Additional company information
        company_info = {
            "name": company_name,
            "ticker": ticker,
            "sector": sector,
            "industry": industry,
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": pe_ratio,
            "dividend_yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            "beta": info.get('beta', 'N/A'),
            "52w_high": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52w_low": info.get('fiftyTwoWeekLow', 'N/A'),
            "description": info.get('longBusinessSummary', 'No description available.')
        }
        
        return {
            "financials": financials_data,
            "scaled_data": scale_ticker,
            "stock_price": stock_price_history,
            "company_info": company_info,
            "raw_data": {
                "financials": financials,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow
            }
        }
    
    except Exception as e:
        return {"error": f"Error: An unexpected issue occurred with '{ticker}': {str(e)}"}

# Function to calculate performance metrics
def calculate_performance_metrics(stock_data):
    """Calculate key performance metrics from stock price history"""
    if len(stock_data) < 30:
        return {}
        
    # Calculate returns
    daily_returns = stock_data.pct_change().dropna()
    
    # Calculate metrics
    metrics = {
        "volatility": daily_returns.std() * (252 ** 0.5),  # Annualized volatility
        "sharpe_ratio": daily_returns.mean() / daily_returns.std() * (252 ** 0.5) if daily_returns.std() != 0 else 0,
        "max_drawdown": (stock_data / stock_data.cummax() - 1).min(),
        "last_month_return": (stock_data.iloc[-1] / stock_data.iloc[-22] - 1) if len(stock_data) >= 22 else 0,
        "ytd_return": (stock_data.iloc[-1] / stock_data.iloc[0] - 1)
    }
    
    return metrics

# Function to generate technical indicators
def generate_technical_indicators(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    
    # Calculate 50-day and 200-day moving averages
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
    hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = hist['EMA12'] - hist['EMA26']
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    
    return hist

# Function to calculate growth rates
def calculate_growth_rates(financials_data):
    """Calculate YoY growth rates for key financial metrics"""
    growth_rates = {}
    
    # Calculate growth for common metrics
    for key in ["Total Revenue", "Net Income", "EBITDA"]:
        if key in financials_data and isinstance(financials_data[key], pd.Series):
            values = financials_data[key]
            if len(values) >= 2:
                # Calculate year-over-year growth rates
                yoy_growth = []
                for i in range(1, len(values)):
                    if values.iloc[i-1] != 0:
                        growth = (values.iloc[i] - values.iloc[i-1]) / abs(values.iloc[i-1]) * 100
                        yoy_growth.append(growth)
                    else:
                        yoy_growth.append(0)
                
                # Add the growth rates to our dictionary
                growth_rates[f"{key}_growth"] = yoy_growth
    
    return growth_rates

# Main UI function
def main():
    # Sidebar
    st.sidebar.markdown("<h1 class='main-header'>📈 Stock Analyzer</h1>", unsafe_allow_html=True)
    
    # Stock selection
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", "AAPL")
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe:",
        ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "5 Years"]
    )
    
    # Additional options
    show_technical = st.sidebar.checkbox("Show Technical Indicators", True)
    advanced_analysis = st.sidebar.checkbox("Show Advanced Analysis", False)
    dark_mode = st.sidebar.checkbox("Dark Mode", False)
    
    # Apply dark mode if selected
    if dark_mode:
        st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        .metric-card {
            background-color: #262730;
            color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Analyze button
    if st.sidebar.button("Analyze Stock"):
        if ticker:
            # Show main container
            main_container = st.container()
            
            with main_container:
                # Show loading spinner
                with st.spinner(f"Analyzing {ticker}..."):
                    # Fetch data
                    data = get_financials(ticker)
                    
                    if "error" in data:
                        st.error(data["error"])
                        return
                    
                    # Display company information
                    company_info = data["company_info"]
                    
                    # Header with company info
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"<h1 class='main-header'>{company_info['name']} ({company_info['ticker']})</h1>", unsafe_allow_html=True)
                        st.markdown(f"**Sector:** {company_info['sector']} | **Industry:** {company_info['industry']}")
                    
                    with col2:
                        # Current stock price
                        current_price = data["stock_price"].iloc[-1] if not data["stock_price"].empty else "N/A"
                        price_change = data["stock_price"].iloc[-1] - data["stock_price"].iloc[-2] if len(data["stock_price"]) > 1 else 0
                        price_change_pct = (price_change / data["stock_price"].iloc[-2]) * 100 if len(data["stock_price"]) > 1 and data["stock_price"].iloc[-2] != 0 else 0
                        
                        change_color = "positive" if price_change >= 0 else "negative"
                        change_symbol = "↑" if price_change >= 0 else "↓"
                        
                        st.markdown(f"<h2>${current_price:.2f} <span class='{change_color}'>{change_symbol} ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)</span></h2>", unsafe_allow_html=True)
                    
                    # Company description with "Read more" expander
                    with st.expander("Company Description"):
                        st.write(company_info["description"])
                    
                    # Key financial metrics in cards
                    st.markdown("<h2 class='sub-header'>Key Metrics</h2>", unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Market Cap", f"${company_info['market_cap']/1e9:.2f}B" if isinstance(company_info['market_cap'], (int, float)) else "N/A")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("P/E Ratio", f"{company_info['pe_ratio']:.2f}" if company_info['pe_ratio'] else "N/A")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Dividend Yield", f"{company_info['dividend_yield']:.2f}%" if company_info['dividend_yield'] else "0.00%")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Beta", f"{company_info['beta']:.2f}" if isinstance(company_info['beta'], (int, float)) else "N/A")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Create tabs for different analysis sections
                    tab1, tab2, tab3, tab4 = st.tabs(["Stock Performance", "Financial Analysis", "Technical Indicators", "Growth Analysis"])
                    
                    with tab1:
                        st.subheader("Stock Price History")
                        # Use Plotly for interactive charts
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data["stock_price"].index,
                            y=data["stock_price"].values,
                            mode='lines',
                            name=ticker,
                            line=dict(color='royalblue', width=2)
                        ))
                        
                        # Add competitor data if requested
                        if compare_competitors and selected_competitors:
                            for comp in selected_competitors:
                                try:
                                    comp_data = yf.Ticker(comp).history(period="5y")["Close"]
                                    # Normalize to percentage change for fair comparison
                                    comp_data_normalized = (comp_data / comp_data.iloc[0]) * 100
                                    ticker_data_normalized = (data["stock_price"] / data["stock_price"].iloc[0]) * 100
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=ticker_data_normalized.index,
                                        y=ticker_data_normalized.values,
                                        mode='lines',
                                        name=ticker,
                                        line=dict(color='royalblue', width=2)
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=comp_data_normalized.index,
                                        y=comp_data_normalized.values,
                                        mode='lines',
                                        name=comp,
                                        line=dict(width=1.5)
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"{ticker} vs Competitors (Normalized %)",
                                        xaxis_title="Date",
                                        yaxis_title="% Change from Start",
                                        legend_title="Stocks",
                                        hovermode="x unified"
                                    )
                                except:
                                    st.warning(f"Could not load data for {comp}")
                        
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Stock Price (USD)",
                            hovermode="x unified",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with tab2:
                        # Financial Analysis Tab
                        st.subheader("Financial Performance")
                        
                        # Create subtabs for different financial metrics
                        fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Revenue & Profit", "Balance Sheet", "Cash Flow"])
                        
                        with fin_tab1:
                            # Revenue and Profit metrics
                            fig = go.Figure()
                            metrics = ["Total Revenue", "Gross Profit", "Net Income"]
                            
                            for metric in metrics:
                                if metric in data["scaled_data"].columns:
                                    fig.add_trace(go.Scatter(
                                        x=data["scaled_data"].index,
                                        y=data["scaled_data"][metric],
                                        mode='lines+markers',
                                        name=metric
                                    ))
                            
                            fig.update_layout(
                                title="Revenue and Profit Trends (% Change)",
                                xaxis_title="Date",
                                yaxis_title="Percentage Change",
                                hovermode="x unified",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display raw data in expandable section
                            with st.expander("View Raw Data"):
                                if isinstance(data["financials"]["Total Revenue"], pd.Series):
                                    df_display = pd.DataFrame({
                                        "Date": data["financials"]["Total Revenue"].index,
                                        "Total Revenue": data["financials"]["Total Revenue"].values,
                                        "Gross Profit": data["financials"]["Gross Profit"].values if isinstance(data["financials"]["Gross Profit"], pd.Series) else ['N/A'] * len(data["financials"]["Total Revenue"]),
                                        "Net Income": data["financials"]["Net Income"].values if isinstance(data["financials"]["Net Income"], pd.Series) else ['N/A'] * len(data["financials"]["Total Revenue"])
                                    })
                                    st.dataframe(df_display)
                        
                        with fin_tab2:
                            # Balance Sheet metrics
                            fig = go.Figure()
                            metrics = ["Stockholders Equity", "Total Assets", "Total Liabilities"]
                            
                            for metric in metrics:
                                if metric in data["scaled_data"].columns:
                                    fig.add_trace(go.Scatter(
                                        x=data["scaled_data"].index,
                                        y=data["scaled_data"][metric],
                                        mode='lines+markers',
                                        name=metric
                                    ))
                            
                            fig.update_layout(
                                title="Balance Sheet Metrics (% Change)",
                                xaxis_title="Date",
                                yaxis_title="Percentage Change",
                                hovermode="x unified",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with fin_tab3:
                            # Cash Flow metrics
                            if "Free Cash Flow" in data["scaled_data"].columns:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=data["scaled_data"].index,
                                    y=data["scaled_data"]["Free Cash Flow"],
                                    mode='lines+markers',
                                    name="Free Cash Flow"
                                ))
                                
                                fig.update_layout(
                                    title="Free Cash Flow Trend (% Change)",
                                    xaxis_title="Date",
                                    yaxis_title="Percentage Change",
                                    hovermode="x unified",
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Free Cash Flow data not available for this stock")
                        
                    with tab3:
                        if show_technical:
                            st.subheader("Technical Indicators")
                            
                            # Get technical indicators
                            tech_data = generate_technical_indicators(ticker)
                            
                            # Price with Moving Averages
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['Close'],
                                mode='lines',
                                name='Close Price',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['MA50'],
                                mode='lines',
                                name='50-Day MA',
                                line=dict(color='orange', width=1.5)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['MA200'],
                                mode='lines',
                                name='200-Day MA',
                                line=dict(color='red', width=1.5)
                            ))
                            
                            fig.update_layout(
                                title="Price and Moving Averages",
                                xaxis_title="Date",
                                yaxis_title="Price (USD)",
                                hovermode="x unified",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # RSI Chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['RSI'],
                                mode='lines',
                                name='RSI',
                                line=dict(color='purple', width=1.5)
                            ))
                            
                            # Add overbought and oversold lines
                            fig.add_shape(
                                type="line",
                                x0=tech_data.index[0],
                                y0=70,
                                x1=tech_data.index[-1],
                                y1=70,
                                line=dict(color="red", width=1, dash="dash")
                            )
                            
                            fig.add_shape(
                                type="line",
                                x0=tech_data.index[0],
                                y0=30,
                                x1=tech_data.index[-1],
                                y1=30,
                                line=dict(color="green", width=1, dash="dash")
                            )
                            
                            fig.update_layout(
                                title="Relative Strength Index (RSI)",
                                xaxis_title="Date",
                                yaxis_title="RSI",
                                yaxis=dict(range=[0, 100]),
                                hovermode="x unified",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # MACD Chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['MACD'],
                                mode='lines',
                                name='MACD',
                                line=dict(color='blue', width=1.5)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['Signal'],
                                mode='lines',
                                name='Signal Line',
                                line=dict(color='red', width=1.5)
                            ))
                            
                            # Add bar chart for MACD histogram
                            histogram = tech_data['MACD'] - tech_data['Signal']
                            colors = ['green' if val >= 0 else 'red' for val in histogram]
                            
                            fig.add_trace(go.Bar(
                                x=tech_data.index,
                                y=histogram,
                                name='Histogram',
                                marker_color=colors
                            ))
                            
                            fig.update_layout(
                                title="Moving Average Convergence Divergence (MACD)",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                hovermode="x unified",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Enable 'Show Technical Indicators' in the sidebar to view this section")
                    
                    with tab4:
                        st.subheader("Growth Analysis")
                        
                        # Calculate growth rates
                        growth_rates = calculate_growth_rates(data["financials"])
                        
                        if growth_rates:
                            # Create a growth rate chart
                            fig = go.Figure()
                            
                            for key, values in growth_rates.items():
                                # Get the metric name without "_growth"
                                metric_name = key.replace("_growth", "")
                                
                                # Create index for the growth rates (need to offset by 1 period)
                                if isinstance(data["financials"][metric_name], pd.Series):
                                    indices = data["financials"][metric_name].index[1:]
                                    if len(indices) == len(values):
                                        fig.add_trace(go.Bar(
                                            x=indices,
                                            y=values,
                                            name=f"{metric_name} Growth %"
                                        ))
                            
                            fig.update_layout(
                                title="Year-over-Year Growth Rates",
                                xaxis_title="Date",
                                yaxis_title="Growth Rate (%)",
                                hovermode="x unified",
                                height=500,
                                barmode='group'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add a table with the growth rates
                            with st.expander("View Growth Rate Data"):
                                growth_df = pd.DataFrame(growth_rates)
                                if isinstance(data["financials"]["Total Revenue"], pd.Series):
                                    growth_df.index = data["financials"]["Total Revenue"].index[1:]
                                st.dataframe(growth_df)
                            
                            # Calculate performance metrics
                            perf_metrics = calculate_performance_metrics(data["stock_price"])
                            
                            if perf_metrics:
                                st.subheader("Stock Performance Metrics")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("YTD Return", f"{perf_metrics.get('ytd_return', 0)*100:.2f}%")
                                    
                                with col2:
                                    st.metric("Annualized Volatility", f"{perf_metrics.get('volatility', 0)*100:.2f}%")
                                    
                                with col3:
                                    st.metric("Max Drawdown", f"{perf_metrics.get('max_drawdown', 0)*100:.2f}%")
                                
                                # Add a chart showing the rolling volatility
                                st.subheader("Rolling Volatility (30-Day Window)")
                                
                                # Calculate rolling volatility
                                if len(data["stock_price"]) > 30:
                                    returns = data["stock_price"].pct_change().dropna()
                                    rolling_vol = returns.rolling(window=30).std() * (252 ** 0.5) * 100  # Annualized and in %
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=rolling_vol.index,
                                        y=rolling_vol.values,
                                        mode='lines',
                                        name='30-Day Rolling Volatility',
                                        line=dict(color='purple', width=1.5)
                                    ))
                                    
                                    fig.update_layout(
                                        xaxis_title="Date",
                                        yaxis_title="Annualized Volatility (%)",
                                        hovermode="x unified",
                                        height=300
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Not enough data for rolling volatility calculation")
                        else:
                            st.info("Insufficient data to calculate growth rates")
