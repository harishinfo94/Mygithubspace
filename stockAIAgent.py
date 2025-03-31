import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis Tool",
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
            "MarketCap": market_cap,
        }
        
        # Calculate Company Value Perception if possible
        if len(market_cap) > 0 and not isinstance(stockholders_equity.iloc[0], str):
            company_value_perception = []
            for i in range(len(market_cap)):
                if stockholders_equity.iloc[i] != 0:
                    company_value_perception.append(market_cap[i] / stockholders_equity.iloc[i])
                else:
                    company_value_perception.append(0)
            financials_data["Company value perception"] = company_value_perception
        
        # Create dataframe and scale for visualization
        df_ticker = pd.DataFrame(data=financials_data, index=ordinary_shares.index)
        scale_ticker = percentIncrease(df_ticker)
        
        return [financials_data, scale_ticker, stock_price_history, company_name]
    
    except Exception as e:
        return [None, None, None, f"Error: An unexpected issue occurred with '{ticker}': {str(e)}"]

# Function to generate technical indicators
def generate_technical_indicators(ticker):
    """Calculate technical indicators for the stock"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    
    # Calculate moving averages
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    
    # Calculate RSI
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    return hist

# Main UI function
def main():
    # Title
    st.title("📈 Stock Analysis Tool")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Stock selection
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", "AAPL")
    
    # Analysis options
    show_technical = st.sidebar.checkbox("Show Technical Indicators", True)
    
    # Analyze button
    if st.sidebar.button("Analyze Stock"):
        if ticker:
            # Loading indicator
            with st.spinner(f"Analyzing {ticker}..."):
                # Fetch data
                financials, scale_ticker, stock_price_history, company_name = get_financials(ticker)
                
                if financials is None:
                    st.error(company_name)  # Display error message
                else:
                    # Display company name
                    st.header(f"{company_name} ({ticker})")
                    
                    # Create tabs
                    tab1, tab2, tab3 = st.tabs(["Stock Performance", "Financial Analysis", "Technical Indicators"])
                    
                    with tab1:
                        # Stock price chart
                        st.subheader("Stock Price History")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(stock_price_history.index, stock_price_history, 'b-', linewidth=2)
                        ax.set_title(f"5-Year Stock Price History for {company_name}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Stock Price (USD)")
                        ax.grid(True, linestyle='--', alpha=0.7)
                        st.pyplot(fig)
                        
                        # Calculate basic stats
                        if len(stock_price_history) > 0:
                            current_price = stock_price_history.iloc[-1]
                            year_ago_price = stock_price_history.iloc[-252] if len(stock_price_history) >= 252 else stock_price_history.iloc[0]
                            year_change = ((current_price / year_ago_price) - 1) * 100
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Current Price", f"${current_price:.2f}")
                            with col2:
                                st.metric("1-Year Change", f"{year_change:.2f}%", 
                                         delta=f"{'+' if year_change > 0 else ''}{year_change:.2f}%")
                    
                    with tab2:
                        # Financial metrics tabs
                        fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Revenue & Profit", "EBIT & EBITDA", "Valuation"])
                        
                        with fin_tab1:
                            # Revenue and profit metrics
                            st.subheader("Revenue and Profit Trends")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            for metric in ["Total Revenue", "Gross Profit", "Net Income"]:
                                if metric in scale_ticker.columns:
                                    ax.plot(scale_ticker.index, scale_ticker[metric], marker='o', label=metric)
                            ax.set_title(f"Revenue and Profit Trends for {company_name}")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Percentage Change")
                            ax.legend()
                            ax.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                            
                            st.write(
                                "- **Total Revenue**: Revenue from all operations before costs/expenses.\n"
                                "- **Gross Profit**: Revenue minus Cost of Goods Sold.\n"
                                "- **Net Income**: Final profit after all expenses."
                            )
                        
                        with fin_tab2:
                            # EBIT and EBITDA metrics
                            st.subheader("EBIT and EBITDA Trends")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            for metric in ["EBIT", "EBITDA"]:
                                if metric in scale_ticker.columns:
                                    ax.plot(scale_ticker.index, scale_ticker[metric], marker='o', label=metric)
                            ax.set_title(f"EBIT and EBITDA Trends for {company_name}")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Percentage Change")
                            ax.legend()
                            ax.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                            
                            st.write(
                                "- **EBIT**: Earnings Before Interest & Taxes\n"
                                "- **EBITDA**: Earnings Before Interest, Taxes, Depreciation & Amortization"
                            )
                        
                        with fin_tab3:
                            # Valuation metrics
                            st.subheader("Market Cap and Company Value Perception")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            for metric in ["MarketCap", "Stockholders Equity"]:
                                if metric in scale_ticker.columns:
                                    ax.plot(scale_ticker.index, scale_ticker[metric], marker='o', label=metric)
                            ax.set_title(f"Market Cap and Equity Trends for {company_name}")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Percentage Change")
                            ax.legend()
                            ax.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                            
                            # Company value perception chart (if available)
                            if "Company value perception" in financials:
                                st.subheader("Company Value Perception")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(scale_ticker.index, financials["Company value perception"], 
                                       marker='o', color='purple', label="Company Value Perception")
                                ax.set_title(f"Market Cap / Equity Ratio for {company_name}")
                                ax.set_xlabel("Date")
                                ax.set_ylabel("Ratio")
                                ax.grid(True, linestyle='--', alpha=0.7)
                                st.pyplot(fig)
                                
                                st.write(
                                    "**Company Value Perception**: Market Cap divided by Stockholders Equity. "
                                    "Shows how the market values the company relative to its book value."
                                )
                    
                    with tab3:
                        if show_technical:
                            # Get technical indicators
                            tech_data = generate_technical_indicators(ticker)
                            
                            # Price and Moving Averages
                            st.subheader("Price and Moving Averages")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(tech_data.index, tech_data['Close'], label='Price', linewidth=2)
                            ax.plot(tech_data.index, tech_data['MA50'], label='50-Day MA', linewidth=1.5)
                            ax.plot(tech_data.index, tech_data['MA200'], label='200-Day MA', linewidth=1.5)
                            ax.set_title(f"Price and Moving Averages for {company_name}")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Price (USD)")
                            ax.legend()
                            ax.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                            
                            # RSI Chart
                            st.subheader("Relative Strength Index (RSI)")
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(tech_data.index, tech_data['RSI'], color='purple', linewidth=1.5)
                            ax.axhline(y=70, color='r', linestyle='--', alpha=0.7)
                            ax.axhline(y=30, color='g', linestyle='--', alpha=0.7)
                            ax.set_title(f"RSI for {company_name}")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("RSI")
                            ax.set_ylim(0, 100)
                            ax.grid(True, linestyle='--', alpha=0.7)
                            st.pyplot(fig)
                            
                            st.write(
                                "- **RSI > 70**: Stock may be overbought (potential sell signal)\n"
                                "- **RSI < 30**: Stock may be oversold (potential buy signal)"
                            )
                        else:
                            st.info("Enable 'Show Technical Indicators' in the sidebar to view this section")
        else:
            st.error("Please enter a valid stock ticker.")

# Run the app
if __name__ == "__main__":
    main()
