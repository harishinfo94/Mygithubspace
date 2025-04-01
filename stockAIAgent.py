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
    
    # Strategy specific parameters
    if selected_strategy == "sma_crossover":
        col1, col2 = st.columns(2)
        with col1:
            short_ma = st.number_input("Short MA Period", 5, 100, 50)
        with col2:
            long_ma = st.number_input("Long MA Period", 50, 300, 200)
    elif selected_strategy == "rsi":
        col1, col2 = st.columns(2)
        with col1:
            rsi_period = st.number_input("RSI Period", 5, 30, 14)
        with col2:
            overbought = st.number_input("Overbought Level", 50, 90, 70)
            oversold = st.number_input("Oversold Level", 10, 50, 30)
    elif selected_strategy == "bollinger_bands":
        col1, col2 = st.columns(2)
        with col1:
            sma_period = st.number_input("SMA Period", 5, 50, 20)
        with col2:
            num_std = st.number_input("Number of Standard Deviations", 1.0, 3.0, 2.0, 0.1)
    
    # Initial investment amount
    initial_investment = st.number_input("Initial Investment ($)", 1000, 1000000, 10000, 1000)
    
    if st.button("Run Backtest"):
        with st.spinner(f"Backtesting {strategy_options[selected_strategy]} on {ticker}..."):
            try:
                # Run the backtest
                backtest_results = backtest_strategy(ticker, selected_strategy, selected_period)
                
                if backtest_results:
                    st.success("Backtest completed successfully!")
                    
                    # Display strategy performance
                    st.subheader("Strategy Performance")
                    
                    # Strategy metrics
                    metrics = backtest_results["metrics"]
                    
                    # Calculate investment results
                    final_strategy_value = initial_investment * (1 + metrics["total_return"])
                    final_market_value = initial_investment * (1 + metrics["market_return"])
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Strategy Return", f"{metrics['total_return']:.2%}", 
                                 f"{metrics['total_return'] - metrics['market_return']:.2%}")
                    with col2:
                        st.metric("Buy & Hold Return", f"{metrics['market_return']:.2%}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    
                    # Portfolio value comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Final Strategy Value", f"${final_strategy_value:.2f}", 
                                 f"${final_strategy_value - final_market_value:.2f}")
                    with col2:
                        st.metric("Final Buy & Hold Value", f"${final_market_value:.2f}")
                    
                    # Plot performance chart
                    data = backtest_results["data"]
                    
                    # Create performance chart
                    fig = go.Figure()
                    
                    # Add price chart
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        name="Price",
                        opacity=0.7
                    ))
                    
                    # Add strategy specific indicators
                    if selected_strategy == "sma_crossover":
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['SMA50'],
                            name="SMA50",
                            line=dict(width=1, dash="dash")
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['SMA200'],
                            name="SMA200",
                            line=dict(width=1, dash="dash")
                        ))
                    elif selected_strategy == "rsi":
                        # Create subplot for RSI
                        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True,
                                          subplot_titles=["Price", "RSI"])
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            name="Price",
                            opacity=0.7
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['RSI'],
                            name="RSI",
                            line=dict(color="purple")
                        ), row=2, col=1)
                        
                        # Add overbought/oversold lines
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    elif selected_strategy == "bollinger_bands":
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['MA20'],
                            name="SMA20",
                            line=dict(color="blue", width=1)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Upper_Band'],
                            name="Upper Band",
                            line=dict(color="green", width=1, dash="dash")
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Lower_Band'],
                            name="Lower Band",
                            line=dict(color="red", width=1, dash="dash")
                        ))
                    
                    fig.update_layout(
                        title=f"{ticker} Price with {strategy_options[selected_strategy]}",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create portfolio value chart
                    fig = go.Figure()
                    
                    # Convert cumulative returns to portfolio value
                    strategy_value = data['Cum_Strategy'] * initial_investment
                    market_value = data['Cum_Market'] * initial_investment
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=strategy_value,
                        name=strategy_options[selected_strategy],
                        line=dict(color="green", width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=market_value,
                        name="Buy & Hold",
                        line=dict(color="blue", width=2)
                    ))
                    
                    fig.update_layout(
                        title="Portfolio Value Comparison",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display signal analysis
                    st.subheader("Signal Analysis")
                    
                    # Extract signals from the data
                    signals = data[data['Signal'] != 0].copy()
                    
                    # Convert to buy/sell signals
                    signals['Action'] = signals['Signal'].map({1: 'Buy', -1: 'Sell'})
                    
                    # Show signals table
                    if not signals.empty:
                        signals_df = pd.DataFrame({
                            'Date': signals.index,
                            'Price': signals['Close'],
                            'Action': signals['Action']
                        })
                        
                        st.dataframe(signals_df)
                        
                        # Show trade statistics
                        buys = len(signals[signals['Action'] == 'Buy'])
                        sells = len(signals[signals['Action'] == 'Sell'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Trades", buys + sells)
                        with col2:
                            st.metric("Buy Signals", buys)
                        with col3:
                            st.metric("Sell Signals", sells)
                        
                        # Calculate trade performance
                        if buys > 0 and sells > 0:
                            # Find buy-sell pairs
                            trades = []
                            in_position = False
                            entry_price = 0
                            entry_date = None
                            
                            for idx, row in signals_df.iterrows():
                                if row['Action'] == 'Buy' and not in_position:
                                    entry_price = row['Price']
                                    entry_date = row['Date']
                                    in_position = True
                                elif row['Action'] == 'Sell' and in_position:
                                    exit_price = row['Price']
                                    exit_date = row['Date']
                                    profit_pct = (exit_price / entry_price - 1) * 100
                                    trades.append({
                                        'Entry Date': entry_date,
                                        'Entry Price': entry_price,
                                        'Exit Date': exit_date,
                                        'Exit Price': exit_price,
                                        'Profit (%)': profit_pct
                                    })
                                    in_position = False
                            
                            if trades:
                                trades_df = pd.DataFrame(trades)
                                
                                # Display trade performance
                                st.subheader("Trade Performance")
                                st.dataframe(trades_df)
                                
                                # Calculate statistics
                                winning_trades = len(trades_df[trades_df['Profit (%)'] > 0])
                                losing_trades = len(trades_df[trades_df['Profit (%)'] <= 0])
                                win_rate = winning_trades / len(trades_df) * 100
                                
                                avg_win = trades_df[trades_df['Profit (%)'] > 0]['Profit (%)'].mean() if winning_trades > 0 else 0
                                avg_loss = trades_df[trades_df['Profit (%)'] <= 0]['Profit (%)'].mean() if losing_trades > 0 else 0
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Win Rate", f"{win_rate:.1f}%")
                                    st.metric("Average Win", f"{avg_win:.2f}%")
                                with col2:
                                    st.metric("Total Completed Trades", len(trades_df))
                                    st.metric("Average Loss", f"{avg_loss:.2f}%")
                    else:
                        st.info("No trading signals were generated during the backtest period.")
                    
                    # Strategy explanation
                    st.subheader("Strategy Explanation")
                    
                    if selected_strategy == "sma_crossover":
                        st.markdown("""
                        **Moving Average Crossover (50/200)**
                        
                        This strategy uses two moving averages:
                        - 50-day Moving Average (shorter-term trend)
                        - 200-day Moving Average (longer-term trend)
                        
                        **Trading Rules:**
                        - **Buy Signal:** When the 50-day MA crosses above the 200-day MA (Golden Cross)
                        - **Sell Signal:** When the 50-day MA crosses below the 200-day MA (Death Cross)
                        
                        **Strengths:**
                        - Follows medium to long-term trends
                        - Filters out short-term market noise
                        - Simple to understand and implement
                        
                        **Weaknesses:**
                        - Lag - signals occur after trends have already started
                        - Can generate false signals in choppy markets
                        - Underperforms in sideways or range-bound markets
                        """)
                    elif selected_strategy == "rsi":
                        st.markdown("""
                        **RSI Overbought/Oversold Strategy**
                        
                        This strategy uses the Relative Strength Index (RSI) to identify potential reversals:
                        
                        **Trading Rules:**
                        - **Buy Signal:** When RSI falls below 30 (oversold) and then rises back above 30
                        - **Sell Signal:** When RSI rises above 70 (overbought) and then falls back below 70
                        
                        **Strengths:**
                        - Good at identifying potential reversal points
                        - Can generate profits in sideways or range-bound markets
                        - Provides clear entry and exit signals
                        
                        **Weaknesses:**
                        - Can lead to premature exits during strong trends
                        - May generate false signals during strong trends
                        - Requires proper timing to be effective
                        """)
                    elif selected_strategy == "bollinger_bands":
                        st.markdown("""
                        **Bollinger Bands Strategy**
                        
                        This strategy uses Bollinger Bands to identify potential reversals:
                        
                        **Trading Rules:**
                        - **Buy Signal:** When price touches or drops below the lower band
                        - **Sell Signal:** When price touches or rises above the upper band
                        
                        **Strengths:**
                        - Adapts to market volatility automatically
                        - Identifies potential reversal points
                        - Works well in range-bound markets
                        
                        **Weaknesses:**
                        - Can generate false signals during strong trends
                        - May result in early exits during extended trends
                        - Less effective in low-volatility environments
                        """)
                    
                    # Disclaimer
                    st.warning("""
                    **Disclaimer:** Past performance is not indicative of future results. This backtesting tool is for educational purposes only. 
                    The results shown do not account for transaction costs, slippage, or taxes. Always conduct your own research before making investment decisions.
                    """)
                else:
                    st.error("Could not complete backtest. Please try a different stock or strategy.")
            except Exception as e:
                st.error(f"Error during backtesting: {str(e)}")

def economic_indicators_tab():
    st.header("Economic Indicators Analysis")
    
    # Get economic indicators
    with st.spinner("Fetching economic data..."):
        eco_data = get_economic_indicators()
        
        if not eco_data.empty:
            st.success("Economic data loaded successfully!")
            
            # Display economic indicators
            st.subheader("Key Economic Indicators")
            st.dataframe(eco_data)
            
            # Create visualizations
            st.subheader("Economic Trends")
            
            # Allow user to select indicators to visualize
            selected_indicators = st.multiselect("Select indicators to visualize", eco_data.columns, default=eco_data.columns.tolist()[:2])
            
            if selected_indicators:
                # Create trend chart
                fig = go.Figure()
                
                for indicator in selected_indicators:
                    fig.add_trace(go.Scatter(
                        x=eco_data.index,
                        y=eco_data[indicator],
                        mode='lines+markers',
                        name=indicator
                    ))
                
                fig.update_layout(
                    title="Economic Indicator Trends",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add correlation matrix
                if len(selected_indicators) > 1:
                    st.subheader("Correlation Between Indicators")
                    
                    corr = eco_data[selected_indicators].corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorscale='RdBu',
                        zmin=-1,
                        zmax=1,
                        text=[[f"{val:.2f}" for val in row] for row in corr.values],
                        texttemplate="%{text}"
                    ))
                    
                    fig.update_layout(title="Correlation Matrix")
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            csv = eco_data.to_csv()
            st.download_button("Download Economic Data", csv, "economic_indicators.csv", "text/csv")
            
            # Economic outlook
            st.subheader("Economic Outlook")
            
            # Calculate trend direction for key indicators
            if "GDP Growth" in eco_data.columns and "Inflation" in eco_data.columns:
                recent_gdp = eco_data["GDP Growth"].iloc[-3:].mean()
                gdp_trend = eco_data["GDP Growth"].iloc[-3:].is_monotonic_increasing
                
                recent_inflation = eco_data["Inflation"].iloc[-3:].mean()
                inflation_trend = eco_data["Inflation"].iloc[-3:].is_monotonic_increasing
                
                # Simple economic outlook
                if recent_gdp > 2 and not inflation_trend:
                    outlook = "Positive"
                    color = "green"
                elif recent_gdp < 0 or (recent_inflation > 5 and inflation_trend):
                    outlook = "Negative"
                    color = "red"
                else:
                    outlook = "Neutral"
                    color = "orange"
                
                st.markdown(f"""
                <div style="background-color:{color}; padding:15px; border-radius:10px; text-align:center;">
                    <h2 style="color:white; margin:0;">Economic Outlook: {outlook}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                **Recent GDP Growth:** {recent_gdp:.1f}% ({gdp_trend and 'Rising' or 'Stable/Falling'})  
                **Recent Inflation:** {recent_inflation:.1f}% ({inflation_trend and 'Rising' or 'Stable/Falling'})
                
                **Analysis:** This is a simplified economic outlook based on recent trends in GDP growth and inflation. 
                A more comprehensive analysis would include other factors such as unemployment, consumer sentiment, 
                housing market data, industrial production, and leading economic indicators.
                """)
            else:
                st.info("Insufficient data to generate economic outlook.")
            
            # Correlation with major indices
            st.subheader("Economic Indicators vs. Market Performance")
            
            # Allow user to select market index
            index_options = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"}
            selected_index = st.selectbox("Select Market Index", list(index_options.keys()), format_func=lambda x: index_options[x])
            
            if selected_index:
                try:
                    # Get index data
                    index_data = yf.download(selected_index, start=eco_data.index[0], end=eco_data.index[-1])
                    
                    if not index_data.empty:
                        # Resample to monthly for comparison with economic data
                        monthly_index = index_data["Close"].resample('M').last()
                        
                        # Calculate returns
                        monthly_returns = monthly_index.pct_change() * 100
                        
                        # Merge with economic data
                        comparison_data = pd.DataFrame(monthly_returns)
                        comparison_data.columns = [index_options[selected_index]]
                        
                        # Align dates (use only common dates)
                        common_dates = comparison_data.index.intersection(eco_data.index)
                        
                        if len(common_dates) > 0:
                            aligned_index = comparison_data.loc[common_dates]
                            aligned_eco = eco_data.loc[common_dates]
                            
                            # Create correlation matrix
                            combined_data = pd.concat([aligned_index, aligned_eco], axis=1)
                            corr = combined_data.corr()
                            
                            # Display correlation with index
                            index_corr = corr.iloc[0, 1:].sort_values(ascending=False)
                            
                            st.markdown(f"#### Correlation Between {index_options[selected_index]} and Economic Indicators")
                            
                            # Create bar chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=index_corr.index,
                                y=index_corr.values,
                                marker_color=['green' if x > 0 else 'red' for x in index_corr.values]
                            ))
                            
                            fig.update_layout(
                                title=f"Correlation with {index_options[selected_index]} Returns",
                                xaxis_title="Economic Indicator",
                                yaxis_title="Correlation Coefficient",
                                yaxis=dict(range=[-1, 1])
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **How to interpret:**
                            - **Positive correlation:** The indicator tends to move in the same direction as the market
                            - **Negative correlation:** The indicator tends to move in the opposite direction of the market
                            - **Stronger correlation (closer to +1 or -1):** More consistent relationship
                            - **Weaker correlation (closer to 0):** Less consistent relationship
                            
                            Remember that correlation does not imply causation. Economic indicators and market performance may be influenced by common factors rather than directly affecting each other.
                            """)
                        else:
                            st.warning("Insufficient overlapping data between economic indicators and market data.")
                    else:
                        st.warning(f"Could not fetch data for {index_options[selected_index]}.")
                except Exception as e:
                    st.error(f"Error fetching market data: {str(e)}")
        else:
            st.error("Could not fetch economic data.")

if __name__ == "__main__":
    main()                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Download option
                                csv = cash_flow.to_csv()
                                st.download_button("Download Cash Flow Statement", csv, f"{ticker}_cash_flow.csv", "text/csv")
                            else:
                                st.warning("Cash flow data not available.")
                        
                        with fin_tab4:
                            # Key Financial Ratios
                            st.subheader("Key Financial Ratios")
                            
                            # Calculate ratios using financial statements
                            ratios = {}
                            
                            # Make sure we have the necessary data
                            income_stmt = stock.income_stmt if not hasattr(stock, 'income_stmt') or not stock.income_stmt.empty else None
                            balance_sheet = stock.balance_sheet if not hasattr(stock, 'balance_sheet') or not stock.balance_sheet.empty else None
                            cash_flow = stock.cashflow if not hasattr(stock, 'cashflow') or not stock.cashflow.empty else None
                            
                            if income_stmt is not None and balance_sheet is not None:
                                # Create common periods list (intersection of all statements)
                                periods = income_stmt.columns.intersection(balance_sheet.columns)
                                if cash_flow is not None:
                                    periods = periods.intersection(cash_flow.columns)
                                
                                # Create multi-index DataFrame for better organization
                                if not periods.empty:
                                    ratio_categories = {
                                        "Profitability": [],
                                        "Liquidity": [],
                                        "Solvency": [],
                                        "Efficiency": [],
                                        "Valuation": []
                                    }
                                    
                                    # --- Profitability Ratios ---
                                    
                                    # Return on Assets (ROA)
                                    if "Net Income" in income_stmt.index and "Total Assets" in balance_sheet.index:
                                        roa = {}
                                        for period in periods:
                                            roa[period] = income_stmt.loc["Net Income", period] / balance_sheet.loc["Total Assets", period]
                                        ratio_categories["Profitability"].append(("Return on Assets (ROA)", roa))
                                    
                                    # Return on Equity (ROE)
                                    if "Net Income" in income_stmt.index and "Stockholders Equity" in balance_sheet.index:
                                        roe = {}
                                        for period in periods:
                                            roe[period] = income_stmt.loc["Net Income", period] / balance_sheet.loc["Stockholders Equity", period]
                                        ratio_categories["Profitability"].append(("Return on Equity (ROE)", roe))
                                    
                                    # Net Profit Margin
                                    if "Net Income" in income_stmt.index and "Total Revenue" in income_stmt.index:
                                        npm = {}
                                        for period in periods:
                                            npm[period] = income_stmt.loc["Net Income", period] / income_stmt.loc["Total Revenue", period]
                                        ratio_categories["Profitability"].append(("Net Profit Margin", npm))
                                    
                                    # Gross Margin
                                    if "Gross Profit" in income_stmt.index and "Total Revenue" in income_stmt.index:
                                        gm = {}
                                        for period in periods:
                                            gm[period] = income_stmt.loc["Gross Profit", period] / income_stmt.loc["Total Revenue", period]
                                        ratio_categories["Profitability"].append(("Gross Margin", gm))
                                    
                                    # --- Liquidity Ratios ---
                                    
                                    # Current Ratio
                                    if "Current Assets" in balance_sheet.index and "Current Liabilities" in balance_sheet.index:
                                        cr = {}
                                        for period in periods:
                                            cr[period] = balance_sheet.loc["Current Assets", period] / balance_sheet.loc["Current Liabilities", period]
                                        ratio_categories["Liquidity"].append(("Current Ratio", cr))
                                    
                                    # Quick Ratio
                                    if "Current Assets" in balance_sheet.index and "Current Liabilities" in balance_sheet.index and "Inventory" in balance_sheet.index:
                                        qr = {}
                                        for period in periods:
                                            qr[period] = (balance_sheet.loc["Current Assets", period] - balance_sheet.loc["Inventory", period]) / balance_sheet.loc["Current Liabilities", period]
                                        ratio_categories["Liquidity"].append(("Quick Ratio", qr))
                                    
                                    # --- Solvency Ratios ---
                                    
                                    # Debt-to-Equity
                                    if "Total Debt" in balance_sheet.index and "Stockholders Equity" in balance_sheet.index:
                                        de = {}
                                        for period in periods:
                                            de[period] = balance_sheet.loc["Total Debt", period] / balance_sheet.loc["Stockholders Equity", period]
                                        ratio_categories["Solvency"].append(("Debt-to-Equity", de))
                                    
                                    # Debt-to-Assets
                                    if "Total Debt" in balance_sheet.index and "Total Assets" in balance_sheet.index:
                                        da = {}
                                        for period in periods:
                                            da[period] = balance_sheet.loc["Total Debt", period] / balance_sheet.loc["Total Assets", period]
                                        ratio_categories["Solvency"].append(("Debt-to-Assets", da))
                                    
                                    # Interest Coverage Ratio
                                    if "EBIT" in income_stmt.index and "Interest Expense" in income_stmt.index:
                                        ic = {}
                                        for period in periods:
                                            # Avoid division by zero
                                            if income_stmt.loc["Interest Expense", period] != 0:
                                                ic[period] = income_stmt.loc["EBIT", period] / abs(income_stmt.loc["Interest Expense", period])
                                            else:
                                                ic[period] = float('inf')  # or some other placeholder
                                        ratio_categories["Solvency"].append(("Interest Coverage", ic))
                                    
                                    # --- Efficiency Ratios ---
                                    
                                    # Asset Turnover
                                    if "Total Revenue" in income_stmt.index and "Total Assets" in balance_sheet.index:
                                        at = {}
                                        for period in periods:
                                            at[period] = income_stmt.loc["Total Revenue", period] / balance_sheet.loc["Total Assets", period]
                                        ratio_categories["Efficiency"].append(("Asset Turnover", at))
                                    
                                    # --- Valuation Ratios ---
                                    # Note: These typically require market data, which we already have in info
                                    
                                    # Display ratios by category
                                    for category, ratios_list in ratio_categories.items():
                                        if ratios_list:
                                            st.markdown(f"##### {category} Ratios")
                                            
                                            # Create DataFrame for this category
                                            category_data = {}
                                            for name, values in ratios_list:
                                                category_data[name] = pd.Series(values)
                                            
                                            category_df = pd.DataFrame(category_data).T
                                            
                                            # Format as percentages where appropriate
                                            percentage_ratios = [
                                                "Return on Assets (ROA)", "Return on Equity (ROE)",
                                                "Net Profit Margin", "Gross Margin", "Debt-to-Assets"
                                            ]
                                            
                                            # Format display
                                            formatted_df = category_df.copy()
                                            for idx in formatted_df.index:
                                                if idx in percentage_ratios:
                                                    formatted_df.loc[idx] = formatted_df.loc[idx].map(lambda x: f"{x:.2%}")
                                                else:
                                                    formatted_df.loc[idx] = formatted_df.loc[idx].map(lambda x: f"{x:.2f}")
                                            
                                            st.dataframe(formatted_df)
                                            
                                            # Visualize ratios
                                            if len(category_df) > 0:
                                                # Transpose for better visualization
                                                plot_df = category_df.T
                                                
                                                fig = go.Figure()
                                                for col in plot_df.columns:
                                                    fig.add_trace(go.Scatter(
                                                        x=plot_df.index,
                                                        y=plot_df[col],
                                                        mode='lines+markers',
                                                        name=col
                                                    ))
                                                
                                                fig.update_layout(
                                                    title=f"{category} Ratios Over Time",
                                                    xaxis_title="Period",
                                                    yaxis_title="Ratio Value"
                                                )
                                                
                                                st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No common periods found across financial statements.")
                            else:
                                st.warning("Insufficient financial statement data to calculate ratios.")
                    
                    tab_index += 1
                
                # Growth Metrics Tab
                if show_growth_metrics:
                    with tabs[tab_index]:
                        st.subheader("Growth Metrics Analysis")
                        
                        # Get financial data
                        income_stmt = stock.income_stmt if not hasattr(stock, 'income_stmt') or not stock.income_stmt.empty else None
                        quarterly_income = stock.quarterly_income_stmt if not hasattr(stock, 'quarterly_income_stmt') or not stock.quarterly_income_stmt.empty else None
                        
                        if income_stmt is not None or quarterly_income is not None:
                            # Create tabs for annual vs quarterly
                            growth_tab1, growth_tab2 = st.tabs(["Annual Growth", "Quarterly Growth"])
                            
                            with growth_tab1:
                                if income_stmt is not None:
                                    # Calculate annual growth rates
                                    annual_growth = income_stmt.pct_change(axis=1) * 100
                                    
                                    # Key metrics to track
                                    key_metrics = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EPS", "EBITDA"]
                                    available_metrics = [m for m in key_metrics if m in annual_growth.index]
                                    
                                    if available_metrics:
                                        # Extract data for key metrics
                                        key_growth = annual_growth.loc[available_metrics]
                                        
                                        # Create growth chart
                                        fig = go.Figure()
                                        
                                        for metric in key_growth.index:
                                            fig.add_trace(go.Bar(
                                                x=key_growth.columns,
                                                y=key_growth.loc[metric],
                                                name=metric
                                            ))
                                        
                                        fig.update_layout(
                                            title="Annual Growth Rates",
                                            xaxis_title="Year",
                                            yaxis_title="Growth Rate (%)",
                                            barmode='group'
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Growth trends
                                        st.subheader("Growth Trends")
                                        
                                        # Create line chart for trends
                                        fig = go.Figure()
                                        
                                        for metric in key_growth.index:
                                            fig.add_trace(go.Scatter(
                                                x=key_growth.columns,
                                                y=key_growth.loc[metric],
                                                mode='lines+markers',
                                                name=metric
                                            ))
                                        
                                        fig.update_layout(
                                            title="Growth Rate Trends",
                                            xaxis_title="Year",
                                            yaxis_title="Growth Rate (%)"
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Show raw data
                                        st.markdown("#### Annual Growth Rates (%)")
                                        st.dataframe(key_growth)
                                        
                                        # Download option
                                        csv = key_growth.to_csv()
                                        st.download_button("Download Annual Growth Data", csv, f"{ticker}_annual_growth.csv", "text/csv")
                                    else:
                                        st.warning("No key metrics available for growth calculation.")
                                else:
                                    st.warning("Annual income statement data not available.")
                            
                            with growth_tab2:
                                if quarterly_income is not None:
                                    # Calculate quarterly growth rates
                                    quarterly_growth = quarterly_income.pct_change(axis=1) * 100
                                    
                                    # Key metrics to track (same as annual)
                                    key_metrics = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EPS", "EBITDA"]
                                    available_metrics = [m for m in key_metrics if m in quarterly_growth.index]
                                    
                                    if available_metrics:
                                        # Extract data for key metrics
                                        key_growth = quarterly_growth.loc[available_metrics]
                                        
                                        # Create growth chart
                                        fig = go.Figure()
                                        
                                        for metric in key_growth.index:
                                            fig.add_trace(go.Bar(
                                                x=key_growth.columns,
                                                y=key_growth.loc[metric],
                                                name=metric
                                            ))
                                        
                                        fig.update_layout(
                                            title="Quarterly Growth Rates",
                                            xaxis_title="Quarter",
                                            yaxis_title="Growth Rate (%)",
                                            barmode='group'
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Show raw data
                                        st.markdown("#### Quarterly Growth Rates (%)")
                                        st.dataframe(key_growth)
                                        
                                        # Download option
                                        csv = key_growth.to_csv()
                                        st.download_button("Download Quarterly Growth Data", csv, f"{ticker}_quarterly_growth.csv", "text/csv")
                                    else:
                                        st.warning("No key metrics available for quarterly growth calculation.")
                                else:
                                    st.warning("Quarterly income statement data not available.")
                            
                            # Year-over-Year Quarterly Growth
                            if quarterly_income is not None:
                                st.subheader("Year-over-Year Quarterly Growth")
                                
                                # Get quarters
                                quarters = quarterly_income.columns
                                
                                # Calculate YoY growth for each quarter
                                yoy_growth = {}
                                
                                # Key metrics
                                key_metrics = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
                                available_metrics = [m for m in key_metrics if m in quarterly_income.index]
                                
                                # Calculate YoY growth for each available metric
                                for metric in available_metrics:
                                    metric_data = quarterly_income.loc[metric]
                                    yoy_values = []
                                    yoy_periods = []
                                    
                                    # Compare each quarter with the same quarter from previous year
                                    for i in range(4, len(quarters)):
                                        current_quarter = metric_data[quarters[i]]
                                        year_ago_quarter = metric_data[quarters[i-4]]
                                        
                                        if year_ago_quarter != 0:
                                            yoy_change = (current_quarter / year_ago_quarter - 1) * 100
                                            yoy_values.append(yoy_change)
                                            yoy_periods.append(quarters[i])
                                    
                                    if yoy_values:
                                        yoy_growth[metric] = pd.Series(yoy_values, index=yoy_periods)
                                
                                if yoy_growth:
                                    # Convert to DataFrame
                                    yoy_df = pd.DataFrame(yoy_growth)
                                    
                                    # Visualize YoY growth
                                    fig = go.Figure()
                                    
                                    for column in yoy_df.columns:
                                        fig.add_trace(go.Bar(
                                            x=yoy_df.index,
                                            y=yoy_df[column],
                                            name=column
                                        ))
                                    
                                    fig.update_layout(
                                        title="Year-over-Year Quarterly Growth",
                                        xaxis_title="Quarter",
                                        yaxis_title="YoY Growth Rate (%)",
                                        barmode='group'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show raw data
                                    st.markdown("#### Year-over-Year Quarterly Growth Rates (%)")
                                    st.dataframe(yoy_df)
                                    
                                    # Download option
                                    csv = yoy_df.to_csv()
                                    st.download_button("Download YoY Quarterly Growth Data", csv, f"{ticker}_yoy_quarterly_growth.csv", "text/csv")
                                else:
                                    st.warning("Insufficient quarterly data for YoY comparison.")
                            
                            # Growth vs Industry
                            st.subheader("Growth vs. Industry")
                            st.info("This feature would compare the company's growth rates with industry averages. This requires additional industry data sources.")
                            
                            # Growth Projections (Analyst Estimates)
                            st.subheader("Growth Projections")
                            
                            # Get analyst data
                            try:
                                earnings_trend = stock.earnings_trend
                                if earnings_trend is not None and 'Growth' in earnings_trend.index:
                                    # Extract growth projections
                                    growth_data = earnings_trend.loc['Growth']
                                    
                                    # Display growth projections
                                    st.dataframe(growth_data)
                                    
                                    # Visualize growth projections
                                    if '+5y' in growth_data.columns:
                                        # Extract relevant columns for visualization
                                        viz_columns = [c for c in ['-5y', '-1y', '0y', '+1y', '+5y'] if c in growth_data.columns]
                                        
                                        # Available metrics
                                        available_metrics = [m for m in ['Revenue', 'Earnings'] if m in growth_data.index]
                                        
                                        if available_metrics:
                                            # Create chart
                                            fig = go.Figure()
                                            
                                            for metric in available_metrics:
                                                # Convert percentage strings to float values
                                                values = []
                                                for col in viz_columns:
                                                    val = growth_data.loc[metric, col]
                                                    if isinstance(val, str) and '%' in val:
                                                        values.append(float(val.replace('%', '')))
                                                    elif isinstance(val, (int, float)):
                                                        values.append(val * 100 if val < 1 else val)
                                                    else:
                                                        values.append(0)
                                                
                                                fig.add_trace(go.Scatter(
                                                    x=viz_columns,
                                                    y=values,
                                                    mode='lines+markers',
                                                    name=metric
                                                ))
                                            
                                            fig.update_layout(
                                                title="Earnings & Revenue Growth Projections",
                                                xaxis_title="Time Period",
                                                yaxis_title="Growth Rate (%)"
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("Growth projection data format not as expected.")
                                else:
                                    st.warning("Analyst growth projections not available.")
                            except Exception as e:
                                st.warning(f"Could not retrieve growth projections: {str(e)}")
                        else:
                            st.warning("Financial statement data not available for growth analysis.")
                    
                    tab_index += 1
                
                # News & Sentiment Tab
                if show_news_sentiment:
                    with tabs[tab_index]:
                        st.subheader("News & Sentiment Analysis")
                        
                        # Get news data
                        news_data = get_stock_news(ticker, limit=10)
                        
                        if news_data:
                            # Display news with sentiment analysis
                            st.markdown("### Recent News Articles")
                            
                            # Calculate overall sentiment
                            avg_sentiment = sum(item['sentiment'] for item in news_data) / len(news_data)
                            
                            # Display sentiment gauge
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = (avg_sentiment + 1) / 2 * 100,  # Convert from [-1,1] to [0,100]
                                title = {'text': "Overall News Sentiment"},
                                gauge = {
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 33], 'color': "red"},
                                        {'range': [33, 67], 'color': "yellow"},
                                        {'range': [67, 100], 'color': "green"}
                                    ]
                                }
                            ))
                            
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display individual news items
                            for i, news in enumerate(news_data):
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"**{news['title']}**")
                                    st.markdown(f"*{news['publisher']} - {news['published'].strftime('%Y-%m-%d')}*")
                                    st.markdown(f"[Read more]({news['link']})")
                                with col2:
                                    sentiment = news['sentiment']
                                    color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "orange"
                                    st.markdown(
                                        f"""
                                        <div style="background-color:{color}; padding:10px; border-radius:5px;">
                                            <h4 style="margin:0; color:white; text-align:center;">{sentiment:.2f}</h4>
                                            <p style="margin:0; color:white; text-align:center;">Sentiment</p>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                                st.markdown("---")
                            
                            # Sentiment over time
                            st.subheader("Sentiment Trend")
                            
                            # Extract dates and sentiment scores
                            dates = [item['published'] for item in news_data]
                            sentiments = [item['sentiment'] for item in news_data]
                            
                            # Sort by date
                            date_sentiment = sorted(zip(dates, sentiments), key=lambda x: x[0])
                            dates = [item[0] for item in date_sentiment]
                            sentiments = [item[1] for item in date_sentiment]
                            
                            # Create sentiment trend chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=dates,
                                y=sentiments,
                                mode='lines+markers',
                                marker=dict(
                                    color=['green' if s > 0 else 'red' for s in sentiments],
                                    size=10
                                ),
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            
                            fig.update_layout(
                                title="News Sentiment Trend",
                                xaxis_title="Date",
                                yaxis_title="Sentiment Score",
                                yaxis=dict(range=[-1, 1])
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **How to interpret sentiment scores:**
                            - **+1.0 to +0.3:** Strong positive sentiment
                            - **+0.3 to +0.1:** Slightly positive sentiment
                            - **+0.1 to -0.1:** Neutral sentiment
                            - **-0.1 to -0.3:** Slightly negative sentiment
                            - **-0.3 to -1.0:** Strong negative sentiment
                            
                            News sentiment can be a leading indicator of stock price movements, but should be used in conjunction with other analysis methods for best results.
                            """)
                        else:
                            st.warning("No recent news data available.")
                    
                    tab_index += 1
            except Exception as e:
                st.error(f"Error performing advanced analysis: {str(e)}")
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")

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
        with st.spinner("Screening stocks... This may take a moment."):
            try:
                results = stock_screener(criteria)
                
                if not results.empty:
                    st.success(f"Found {len(results)} stocks matching your criteria.")
                    
                    # Display results
                    st.dataframe(results)
                    
                    # Visualization of results
                    if len(results) > 0:
                        st.subheader("Screening Results Visualization")
                        
                        # Ask user which metric to visualize
                        viz_options = [col for col in results.columns if col not in ['ticker', 'name', 'sector', 'industry']]
                        selected_metric = st.selectbox("Select metric to visualize", viz_options)
                        
                        if selected_metric in results.columns:
                            # Sort by selected metric
                            sorted_results = results.sort_values(selected_metric, ascending=False)
                            
                            # Create bar chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=sorted_results['ticker'],
                                y=sorted_results[selected_metric],
                                text=sorted_results[selected_metric],
                                textposition='auto',
                                marker_color='blue'
                            ))
                            
                            fig.update_layout(
                                title=f"{selected_metric} by Stock",
                                xaxis_title="Stock Ticker",
                                yaxis_title=selected_metric
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Sector breakdown
                            if 'sector' in results.columns:
                                sector_counts = results['sector'].value_counts()
                                
                                fig = go.Figure(data=[go.Pie(
                                    labels=sector_counts.index,
                                    values=sector_counts.values,
                                    hole=0.4
                                )])
                                
                                fig.update_layout(title="Sector Breakdown of Screened Stocks")
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button("Download Screening Results", csv, "stock_screening_results.csv", "text/csv")
                else:
                    st.warning("No stocks found matching your criteria. Try relaxing some parameters.")
            except Exception as e:
                st.error(f"Error running stock screener: {str(e)}")

def trading_strategies_tab():
    st.header("Trading Strategy Backtesting")
                                    - Higher volume on up days suggests strong buying pressure
                                - Higher volume on down days suggests strong selling pressure
                                - Low volume during price movements may indicate weak trends
                                - Volume spikes often occur at trend reversal points or breakouts
                                """)
                            
                            # Trading signals based on technical analysis
                            st.subheader("Technical Analysis Signals")
                            
                            # Calculate signals
                            signals = []
                            
                            # Moving Average signals
                            last_ma50 = tech_data['MA50'].iloc[-1]
                            last_ma200 = tech_data['MA200'].iloc[-1]
                            prev_ma50 = tech_data['MA50'].iloc[-2]
                            prev_ma200 = tech_data['MA200'].iloc[-2]
                            
                            if last_ma50 > last_ma200 and prev_ma50 <= prev_ma200:
                                signals.append({
                                    "Signal": "Golden Cross",
                                    "Type": "Bullish",
                                    "Description": "50-day MA crossed above 200-day MA",
                                    "Strength": "Strong"
                                })
                            elif last_ma50 < last_ma200 and prev_ma50 >= prev_ma200:
                                signals.append({
                                    "Signal": "Death Cross",
                                    "Type": "Bearish",
                                    "Description": "50-day MA crossed below 200-day MA",
                                    "Strength": "Strong"
                                })
                            
                            # RSI signals
                            last_rsi = tech_data['RSI'].iloc[-1]
                            if last_rsi > 70:
                                signals.append({
                                    "Signal": "RSI Overbought",
                                    "Type": "Bearish",
                                    "Description": f"RSI is {last_rsi:.2f}, indicating overbought conditions",
                                    "Strength": "Moderate"
                                })
                            elif last_rsi < 30:
                                signals.append({
                                    "Signal": "RSI Oversold",
                                    "Type": "Bullish",
                                    "Description": f"RSI is {last_rsi:.2f}, indicating oversold conditions",
                                    "Strength": "Moderate"
                                })
                            
                            # MACD signals
                            last_macd = tech_data['MACD'].iloc[-1]
                            last_signal = tech_data['Signal'].iloc[-1]
                            prev_macd = tech_data['MACD'].iloc[-2]
                            prev_signal = tech_data['Signal'].iloc[-2]
                            
                            if last_macd > last_signal and prev_macd <= prev_signal:
                                signals.append({
                                    "Signal": "MACD Bullish Crossover",
                                    "Type": "Bullish",
                                    "Description": "MACD line crossed above Signal line",
                                    "Strength": "Moderate"
                                })
                            elif last_macd < last_signal and prev_macd >= prev_signal:
                                signals.append({
                                    "Signal": "MACD Bearish Crossover",
                                    "Type": "Bearish",
                                    "Description": "MACD line crossed below Signal line",
                                    "Strength": "Moderate"
                                })
                            
                            # Bollinger Band signals
                            last_close = tech_data['Close'].iloc[-1]
                            last_upper = tech_data['Upper_Band'].iloc[-1]
                            last_lower = tech_data['Lower_Band'].iloc[-1]
                            
                            if last_close >= last_upper:
                                signals.append({
                                    "Signal": "Bollinger Band Upper Touch",
                                    "Type": "Bearish",
                                    "Description": "Price touched or exceeded upper Bollinger Band",
                                    "Strength": "Weak"
                                })
                            elif last_close <= last_lower:
                                signals.append({
                                    "Signal": "Bollinger Band Lower Touch",
                                    "Type": "Bullish",
                                    "Description": "Price touched or dropped below lower Bollinger Band",
                                    "Strength": "Weak"
                                })
                            
                            # Stochastic signals
                            last_k = tech_data['%K'].iloc[-1]
                            last_d = tech_data['%D'].iloc[-1]
                            prev_k = tech_data['%K'].iloc[-2]
                            prev_d = tech_data['%D'].iloc[-2]
                            
                            if last_k > last_d and prev_k <= prev_d and last_k < 20:
                                signals.append({
                                    "Signal": "Stochastic Bullish Crossover (Oversold)",
                                    "Type": "Bullish",
                                    "Description": "%K crossed above %D in oversold territory",
                                    "Strength": "Strong"
                                })
                            elif last_k < last_d and prev_k >= prev_d and last_k > 80:
                                signals.append({
                                    "Signal": "Stochastic Bearish Crossover (Overbought)",
                                    "Type": "Bearish",
                                    "Description": "%K crossed below %D in overbought territory",
                                    "Strength": "Strong"
                                })
                            
                            # ADX signals
                            last_adx = tech_data['ADX'].iloc[-1]
                            if last_adx > 25:
                                # Determine if it's an uptrend or downtrend
                                if last_ma50 > last_ma200:
                                    signals.append({
                                        "Signal": "Strong Uptrend",
                                        "Type": "Bullish",
                                        "Description": f"ADX is {last_adx:.2f}, indicating a strong trend (uptrend based on MA)",
                                        "Strength": "Moderate"
                                    })
                                else:
                                    signals.append({
                                        "Signal": "Strong Downtrend",
                                        "Type": "Bearish",
                                        "Description": f"ADX is {last_adx:.2f}, indicating a strong trend (downtrend based on MA)",
                                        "Strength": "Moderate"
                                    })
                            
                            # Display signals if any
                            if signals:
                                # Group by signal type
                                bullish_signals = [s for s in signals if s["Type"] == "Bullish"]
                                bearish_signals = [s for s in signals if s["Type"] == "Bearish"]
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("### Bullish Signals")
                                    if bullish_signals:
                                        for signal in bullish_signals:
                                            st.markdown(f"""
                                            **{signal['Signal']}** ({signal['Strength']})  
                                            {signal['Description']}
                                            """)
                                    else:
                                        st.info("No bullish signals detected.")
                                
                                with col2:
                                    st.markdown("### Bearish Signals")
                                    if bearish_signals:
                                        for signal in bearish_signals:
                                            st.markdown(f"""
                                            **{signal['Signal']}** ({signal['Strength']})  
                                            {signal['Description']}
                                            """)
                                    else:
                                        st.info("No bearish signals detected.")
                                
                                # Overall assessment
                                bullish_count = len(bullish_signals)
                                bearish_count = len(bearish_signals)
                                
                                # Weight signals by strength
                                bullish_score = sum([3 if s["Strength"] == "Strong" else 2 if s["Strength"] == "Moderate" else 1 for s in bullish_signals])
                                bearish_score = sum([3 if s["Strength"] == "Strong" else 2 if s["Strength"] == "Moderate" else 1 for s in bearish_signals])
                                
                                st.markdown("### Technical Analysis Summary")
                                
                                # Create gauge chart for technical outlook
                                if bullish_score + bearish_score > 0:
                                    tech_score = (bullish_score - bearish_score) / (bullish_score + bearish_score)
                                    tech_score_normalized = (tech_score + 1) / 2  # Convert from [-1, 1] to [0, 1]
                                    
                                    fig = go.Figure(go.Indicator(
                                        mode = "gauge+number",
                                        value = tech_score_normalized * 100,
                                        title = {'text': "Technical Outlook"},
                                        gauge = {
                                            'axis': {'range': [0, 100]},
                                            'bar': {'color': "darkblue"},
                                            'steps': [
                                                {'range': [0, 33], 'color': "red"},
                                                {'range': [33, 67], 'color': "yellow"},
                                                {'range': [67, 100], 'color': "green"}
                                            ],
                                            'threshold': {
                                                'line': {'color': "black", 'width': 4},
                                                'thickness': 0.75,
                                                'value': 50
                                            }
                                        }
                                    ))
                                    
                                    fig.update_layout(height=300)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Interpretation text
                                    if tech_score_normalized > 0.67:
                                        st.success("The technical analysis indicates a **bullish** outlook. Multiple technical indicators are showing positive signals.")
                                    elif tech_score_normalized < 0.33:
                                        st.error("The technical analysis indicates a **bearish** outlook. Multiple technical indicators are showing negative signals.")
                                    else:
                                        st.info("The technical analysis indicates a **neutral** outlook. There are mixed signals or the indicators are not showing strong trends.")
                                else:
                                    st.info("No significant technical signals detected at this time.")
                            else:
                                st.info("No significant technical signals detected at this time.")
                        else:
                            st.warning("Could not generate technical indicators.")
                    
                    tab_index += 1
                
                # Industry Comparison Tab
                if show_industry_comparison:
                    with tabs[tab_index]:
                        st.subheader("Industry Comparison")
                        
                        # Get peers data
                        peer_data = industry_comparison(ticker)
                        
                        if peer_data is not None and not peer_data.empty:
                            # Display peer comparison data
                            st.dataframe(peer_data)
                            
                            # Create comparison visualizations
                            st.subheader("Key Metrics Comparison")
                            
                            # Select metrics to visualize
                            metrics_to_compare = st.multiselect(
                                "Select metrics to compare",
                                options=[col for col in peer_data.columns if col not in ['name']],
                                default=['pe_ratio', 'forward_pe', 'price_to_book', 'profit_margin']
                            )
                            
                            if metrics_to_compare:
                                # Create radar chart
                                if len(metrics_to_compare) >= 3:
                                    # Prepare data for radar chart
                                    # Normalize values between 0 and 1 for each metric
                                    radar_data = peer_data[metrics_to_compare].copy()
                                    
                                    # Handle NaN values
                                    radar_data = radar_data.fillna(0)
                                    
                                    for col in radar_data.columns:
                                        if radar_data[col].max() != radar_data[col].min():
                                            # For metrics where lower is better (P/E, etc.), invert the normalization
                                            if col in ['pe_ratio', 'forward_pe', 'price_to_book']:
                                                # Check if min is not 0 to avoid division by zero
                                                if radar_data[col].min() > 0:
                                                    # Invert so lower values get higher normalized scores
                                                    radar_data[col] = 1 - (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())
                                                else:
                                                    # Regular normalization but cap at reasonable values
                                                    radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())
                                                    radar_data[col] = 1 - radar_data[col]  # Invert
                                            else:
                                                # For metrics where higher is better (margins, etc.)
                                                radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())
                                    
                                    # Create radar chart
                                    fig = go.Figure()
                                    
                                    # Add company of interest first (with different color)
                                    company_values = radar_data.loc[ticker].tolist()
                                    theta = metrics_to_compare
                                    
                                    fig.add_trace(go.Scatterpolar(
                                        r=company_values,
                                        theta=theta,
                                        fill='toself',
                                        name=ticker,
                                        line_color='red'
                                    ))
                                    
                                    # Add peer companies
                                    for peer in radar_data.index:
                                        if peer != ticker:
                                            peer_values = radar_data.loc[peer].tolist()
                                            
                                            fig.add_trace(go.Scatterpolar(
                                                r=peer_values,
                                                theta=theta,
                                                fill='toself',
                                                name=peer,
                                                opacity=0.6
                                            ))
                                    
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 1]
                                            )
                                        ),
                                        title="Relative Metrics Comparison (Normalized)",
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.markdown("""
                                    **What this chart shows:** Comparison of key metrics across peer companies, normalized to a 0-1 scale.
                                    
                                    **How to interpret:**
                                    - For valuation metrics (P/E, Forward P/E, P/B): Higher values on the chart indicate *better* valuation (i.e., lower actual ratio)
                                    - For operational metrics (margins, ROE): Higher values indicate better performance
                                    - The company being analyzed is highlighted in red
                                    - Larger area generally indicates better overall metrics
                                    """)
                                
                                # Create individual bar charts for each selected metric
                                for metric in metrics_to_compare:
                                    metric_data = peer_data.dropna(subset=[metric]).sort_values(metric)
                                    
                                    if not metric_data.empty:
                                        fig = go.Figure()
                                        
                                        fig.add_trace(go.Bar(
                                            x=metric_data.index,
                                            y=metric_data[metric],
                                            marker_color=['red' if x == ticker else 'blue' for x in metric_data.index]
                                        ))
                                        
                                        # Format title using proper names
                                        metric_names = {
                                            'pe_ratio': 'P/E Ratio',
                                            'forward_pe': 'Forward P/E',
                                            'price_to_book': 'Price-to-Book',
                                            'peg_ratio': 'PEG Ratio',
                                            'profit_margin': 'Profit Margin',
                                            'beta': 'Beta',
                                            'dividend_yield': 'Dividend Yield',
                                            'market_cap': 'Market Cap'
                                        }
                                        
                                        metric_title = metric_names.get(metric, metric.replace('_', ' ').title())
                                        
                                        # Format y-axis for percentages
                                        percentage_metrics = ['profit_margin', 'dividend_yield']
                                        
                                        fig.update_layout(
                                            title=f"{metric_title} Comparison",
                                            xaxis_title="Company",
                                            yaxis_title=metric_title
                                        )
                                        
                                        # Format y-axis for percentages
                                        if metric in percentage_metrics:
                                            fig.update_traces(text=[f"{val:.2%}" for val in metric_data[metric]], textposition='auto')
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                # Performance comparison over time
                                st.subheader("Performance Comparison")
                                
                                # Allow user to select period
                                period_selected = st.selectbox("Select Time Period for Performance Comparison", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
                                
                                # Get price data for all peers
                                peer_tickers = peer_data.index.tolist()
                                try:
                                    peer_prices = yf.download(peer_tickers, period=period_selected)['Close']
                                    
                                    if not peer_prices.empty:
                                        # Normalize prices to 100 at the beginning
                                        normalized_prices = peer_prices.div(peer_prices.iloc[0]) * 100
                                        
                                        # Create comparison chart
                                        fig = go.Figure()
                                        
                                        # Add main ticker first with thicker line
                                        if ticker in normalized_prices.columns:
                                            fig.add_trace(go.Scatter(
                                                x=normalized_prices.index,
                                                y=normalized_prices[ticker],
                                                name=ticker,
                                                line=dict(color='red', width=3)
                                            ))
                                        
                                        # Add peers
                                        for peer in normalized_prices.columns:
                                            if peer != ticker:
                                                fig.add_trace(go.Scatter(
                                                    x=normalized_prices.index,
                                                    y=normalized_prices[peer],
                                                    name=peer,
                                                    line=dict(width=1.5)
                                                ))
                                        
                                        fig.update_layout(
                                            title=f"Relative Price Performance ({period_selected})",
                                            xaxis_title="Date",
                                            yaxis_title="Normalized Price (%)",
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("Could not fetch price comparison data.")
                                except Exception as e:
                                    st.error(f"Error fetching peer price data: {str(e)}")
                            else:
                                st.warning("Please select at least one metric to compare.")
                        else:
                            st.warning("No peer comparison data available.")
                    
                    tab_index += 1
                
                # Financial Statements Tab
                if show_financial_statements:
                    with tabs[tab_index]:
                        st.subheader("Financial Statements")
                        
                        # Create subtabs for different statements
                        fin_tab1, fin_tab2, fin_tab3, fin_tab4 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Key Ratios"])
                        
                        with fin_tab1:
                            # Income Statement
                            income_stmt = stock.income_stmt
                            if not income_stmt.empty:
                                # Calculate YoY growth
                                income_growth = income_stmt.pct_change(axis=1) * 100
                                
                                # Display income statement with option to toggle between raw values and YoY growth
                                show_growth = st.checkbox("Show Year-over-Year Growth (%)", key="income_growth")
                                
                                if show_growth:
                                    st.dataframe(income_growth)
                                else:
                                    st.dataframe(income_stmt)
                                
                                # Visualize key income statement metrics
                                key_metrics = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
                                available_metrics = [m for m in key_metrics if m in income_stmt.index]
                                
                                if available_metrics:
                                    st.subheader("Key Income Statement Metrics")
                                    
                                    # Extract the data for key metrics
                                    key_data = income_stmt.loc[available_metrics]
                                    
                                    # Create bar chart
                                    fig = go.Figure()
                                    
                                    for metric in key_data.index:
                                        fig.add_trace(go.Bar(
                                            x=key_data.columns,
                                            y=key_data.loc[metric],
                                            name=metric
                                        ))
                                    
                                    fig.update_layout(
                                        title="Key Income Statement Metrics Over Time",
                                        xaxis_title="Period",
                                        yaxis_title="Amount",
                                        barmode='group'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Profit Margins
                                    if "Total Revenue" in income_stmt.index and "Net Income" in income_stmt.index:
                                        st.subheader("Profit Margins")
                                        
                                        # Calculate margins
                                        gross_margin = income_stmt.loc["Gross Profit"] / income_stmt.loc["Total Revenue"] * 100 if "Gross Profit" in income_stmt.index else None
                                        operating_margin = income_stmt.loc["Operating Income"] / income_stmt.loc["Total Revenue"] * 100 if "Operating Income" in income_stmt.index else None
                                        net_margin = income_stmt.loc["Net Income"] / income_stmt.loc["Total Revenue"] * 100
                                        
                                        # Create margin chart
                                        fig = go.Figure()
                                        
                                        if gross_margin is not None:
                                            fig.add_trace(go.Scatter(
                                                x=gross_margin.index,
                                                y=gross_margin,
                                                mode='lines+markers',
                                                name='Gross Margin'
                                            ))
                                        
                                        if operating_margin is not None:
                                            fig.add_trace(go.Scatter(
                                                x=operating_margin.index,
                                                y=operating_margin,
                                                mode='lines+markers',
                                                name='Operating Margin'
                                            ))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=net_margin.index,
                                            y=net_margin,
                                            mode='lines+markers',
                                            name='Net Margin'
                                        ))
                                        
                                        fig.update_layout(
                                            title="Profit Margins Over Time",
                                            xaxis_title="Period",
                                            yaxis_title="Margin (%)"
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                # Download option
                                csv = income_stmt.to_csv()
                                st.download_button("Download Income Statement", csv, f"{ticker}_income_statement.csv", "text/csv")
                            else:
                                st.warning("Income statement data not available.")
                        
                        with fin_tab2:
                            # Balance Sheet
                            balance_sheet = stock.balance_sheet
                            if not balance_sheet.empty:
                                st.dataframe(balance_sheet)
                                
                                # Visualize key balance sheet metrics
                                st.subheader("Assets and Liabilities")
                                
                                # Extract key metrics
                                assets = balance_sheet.loc["Total Assets"] if "Total Assets" in balance_sheet.index else None
                                liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest"] if "Total Liabilities Net Minority Interest" in balance_sheet.index else None
                                equity = balance_sheet.loc["Stockholders Equity"] if "Stockholders Equity" in balance_sheet.index else None
                                
                                if assets is not None and liabilities is not None:
                                    # Create stacked bar chart
                                    fig = go.Figure()
                                    
                                    # Add assets bars
                                    fig.add_trace(go.Bar(
                                        x=assets.index,
                                        y=assets,
                                        name='Total Assets',
                                        marker_color='blue'
                                    ))
                                    
                                    # Add liabilities bars
                                    fig.add_trace(go.Bar(
                                        x=liabilities.index,
                                        y=liabilities,
                                        name='Total Liabilities',
                                        marker_color='red'
                                    ))
                                    
                                    # Add equity bars
                                    if equity is not None:
                                        fig.add_trace(go.Bar(
                                            x=equity.index,
                                            y=equity,
                                            name='Stockholders\' Equity',
                                            marker_color='green'
                                        ))
                                    
                                    fig.update_layout(
                                        title="Assets and Liabilities Over Time",
                                        xaxis_title="Period",
                                        yaxis_title="Amount",
                                        barmode='group'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Calculate and visualize key balance sheet ratios
                                st.subheader("Balance Sheet Ratios")
                                
                                # Current Ratio
                                current_assets = balance_sheet.loc["Current Assets"] if "Current Assets" in balance_sheet.index else None
                                current_liabilities = balance_sheet.loc["Current Liabilities"] if "Current Liabilities" in balance_sheet.index else None
                                
                                # Debt-to-Equity Ratio
                                total_debt = balance_sheet.loc["Total Debt"] if "Total Debt" in balance_sheet.index else None
                                
                                # Create ratios
                                ratios_data = {}
                                
                                if current_assets is not None and current_liabilities is not None:
                                    current_ratio = current_assets / current_liabilities
                                    ratios_data["Current Ratio"] = current_ratio
                                
                                if total_debt is not None and equity is not None:
                                    debt_equity_ratio = total_debt / equity
                                    ratios_data["Debt-to-Equity"] = debt_equity_ratio
                                
                                if assets is not None and equity is not None:
                                    equity_multiplier = assets / equity
                                    ratios_data["Equity Multiplier"] = equity_multiplier
                                
                                if ratios_data:
                                    # Create chart
                                    fig = make_subplots(rows=len(ratios_data), cols=1, subplot_titles=list(ratios_data.keys()))
                                    
                                    for i, (name, values) in enumerate(ratios_data.items(), 1):
                                        fig.add_trace(
                                            go.Scatter(x=values.index, y=values, mode='lines+markers', name=name),
                                            row=i, col=1
                                        )
                                    
                                    fig.update_layout(height=300 * len(ratios_data), showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Download option
                                csv = balance_sheet.to_csv()
                                st.download_button("Download Balance Sheet", csv, f"{ticker}_balance_sheet.csv", "text/csv")
                            else:
                                st.warning("Balance sheet data not available.")
                        
                        with fin_tab3:
                            # Cash Flow Statement
                            cash_flow = stock.cashflow
                            if not cash_flow.empty:
                                st.dataframe(cash_flow)
                                
                                # Visualize key cash flow metrics
                                st.subheader("Key Cash Flow Metrics")
                                
                                # Extract key metrics
                                operating_cf = cash_flow.loc["Operating Cash Flow"] if "Operating Cash Flow" in cash_flow.index else None
                                investing_cf = cash_flow.loc["Investing Cash Flow"] if "Investing Cash Flow" in cash_flow.index else None
                                financing_cf = cash_flow.loc["Financing Cash Flow"] if "Financing Cash Flow" in cash_flow.index else None
                                free_cf = cash_flow.loc["Free Cash Flow"] if "Free Cash Flow" in cash_flow.index else None
                                
                                if operating_cf is not None:
                                    # Create cash flow chart
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Bar(
                                        x=operating_cf.index,
                                        y=operating_cf,
                                        name='Operating CF',
                                        marker_color='green'
                                    ))
                                    
                                    if investing_cf is not None:
                                        fig.add_trace(go.Bar(
                                            x=investing_cf.index,
                                            y=investing_cf,
                                            name='Investing CF',
                                            marker_color='blue'
                                        ))
                                    
                                    if financing_cf is not None:
                                        fig.add_trace(go.Bar(
                                            x=financing_cf.index,
                                            y=financing_cf,
                                            name='Financing CF',
                                            marker_color='orange'
                                        ))
                                    
                                    fig.update_layout(
                                        title="Cash Flows Over Time",
                                        xaxis_title="Period",
                                        yaxis_title="Amount"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Free Cash Flow chart
                                if free_cf is not None:
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Bar(
                                        x=free_cf.index,
                                        y=free_cf,
                                        name='Free Cash Flow',
                                        marker_color=['green' if x >= 0 else 'red' for x in free_cf]
                                    ))
                                    
                                    fig.update_layout(
                                        title="Free Cash Flow Over Time",
                                        xaxis_title="Period",
                                        yaxis_title="Amount"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=                                            name=asset
                                        ))
                                
                                fig3.update_layout(
                                    title="Efficient Frontier",
                                    xaxis_title="Expected Volatility (%)",
                                    yaxis_title="Expected Return (%)",
                                    legend=dict(x=0, y=1)
                                )
                                
                                st.plotly_chart(fig3, use_container_width=True)
                                
                                st.markdown("""
                                **What this chart shows:** The efficient frontier represents the optimal portfolios that offer the highest expected return for a defined level of risk.
                                
                                **Key components:**
                                - **Colored dots:** Simulated random portfolios with different asset combinations
                                - **Red star:** Maximum Sharpe ratio portfolio (optimal risk-adjusted return)
                                - **Green star:** Minimum volatility portfolio (lowest risk)
                                - **Blue dots:** Individual assets in your portfolio
                                
                                **How to interpret:**
                                - Portfolios on the top-left portion of the curve offer the best risk-return tradeoff
                                - The efficient frontier demonstrates the power of diversification - optimal portfolios often have better risk-return profiles than individual assets
                                - Colors indicate Sharpe ratio (risk-adjusted return) - brighter colors represent higher Sharpe ratios
                                """)
                                
                                # Display weights in table format for download
                                optimal_weights_df = pd.DataFrame.from_dict(optimal_weights, orient='index', columns=['Weight'])
                                optimal_weights_df.index.name = 'Asset'
                                
                                min_vol_weights_df = pd.DataFrame.from_dict(min_vol_weights, orient='index', columns=['Weight'])
                                min_vol_weights_df.index.name = 'Asset'
                                
                                # Prepare download buttons
                                csv_optimal = optimal_weights_df.to_csv()
                                csv_min_vol = min_vol_weights_df.to_csv()
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button("Download Optimal Weights", csv_optimal, "optimal_portfolio_weights.csv", "text/csv")
                                with col2:
                                    st.download_button("Download Min Volatility Weights", csv_min_vol, "min_vol_portfolio_weights.csv", "text/csv")
                            else:
                                st.warning("Could not optimize portfolio. Try different assets or time period.")
                        except Exception as e:
                            st.error(f"Error optimizing portfolio: {str(e)}")
                    elif len(portfolio_data.columns) <= 1:
                        st.warning("Need at least two stocks for portfolio optimization.")

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
        with st.spinner(f"Performing advanced analysis on {ticker}..."):
            try:
                # Get stock data
                stock = yf.Ticker(ticker)
                
                # Basic info
                info = stock.info
                if not info:
                    st.error("Could not fetch data for this ticker. Please check the symbol and try again.")
                    return
                
                company_name = info.get('longName', ticker)
                
                # Display company header
                st.header(f"{company_name} ({ticker}) Advanced Analysis")
                
                # Display company info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}")
                with col2:
                    market_cap = info.get('marketCap', 0)
                    if market_cap > 1e9:
                        market_cap_formatted = f"${market_cap/1e9:.2f}B"
                    else:
                        market_cap_formatted = f"${market_cap/1e6:.2f}M"
                    st.metric("Market Cap", market_cap_formatted)
                with col3:
                    st.metric("Sector", info.get('sector', 'N/A'))
                
                # Company description
                with st.expander("Company Description", expanded=False):
                    st.write(info.get('longBusinessSummary', 'No description available.'))
                
                tabs = []
                tab_labels = []
                
                if show_valuation:
                    tab_labels.append("Valuation")
                if show_advanced_technical:
                    tab_labels.append("Technical Analysis")
                if show_industry_comparison:
                    tab_labels.append("Industry Comparison")
                if show_financial_statements:
                    tab_labels.append("Financial Statements")
                if show_growth_metrics:
                    tab_labels.append("Growth Metrics")
                if show_news_sentiment:
                    tab_labels.append("News & Sentiment")
                
                tabs = st.tabs(tab_labels)
                
                tab_index = 0
                
                # Valuation Tab
                if show_valuation:
                    with tabs[tab_index]:
                        st.subheader("Comprehensive Valuation Analysis")
                        
                        # Get valuation metrics
                        valuation_metrics = {
                            "P/E Ratio": info.get('trailingPE', None),
                            "Forward P/E": info.get('forwardPE', None),
                            "PEG Ratio": info.get('pegRatio', None),
                            "Price/Sales": info.get('priceToSalesTrailing12Months', None),
                            "Price/Book": info.get('priceToBook', None),
                            "Enterprise Value/EBITDA": info.get('enterpriseToEbitda', None),
                            "Enterprise Value/Revenue": info.get('enterpriseToRevenue', None),
                            "Market Cap": info.get('marketCap', None),
                            "Enterprise Value": info.get('enterpriseValue', None)
                        }
                        
                        # Filter out None values
                        valuation_metrics = {k: v for k, v in valuation_metrics.items() if v is not None}
                        
                        if valuation_metrics:
                            # Create comparison to sector averages (placeholder values for example)
                            sector = info.get('sector', 'N/A')
                            
                            # These would be replaced with actual sector averages from a data source
                            sector_averages = {
                                "Technology": {"P/E Ratio": 25, "Forward P/E": 20, "PEG Ratio": 1.5, "Price/Sales": 4, "Price/Book": 5},
                                "Healthcare": {"P/E Ratio": 20, "Forward P/E": 18, "PEG Ratio": 1.3, "Price/Sales": 3, "Price/Book": 4},
                                "Consumer Cyclical": {"P/E Ratio": 22, "Forward P/E": 19, "PEG Ratio": 1.4, "Price/Sales": 2, "Price/Book": 3},
                                "Financial Services": {"P/E Ratio": 15, "Forward P/E": 14, "PEG Ratio": 1.2, "Price/Sales": 3, "Price/Book": 1.5},
                                "N/A": {"P/E Ratio": 20, "Forward P/E": 18, "PEG Ratio": 1.3, "Price/Sales": 3, "Price/Book": 3}
                            }
                            
                            # Get relevant sector averages
                            selected_sector_avg = sector_averages.get(sector, sector_averages["N/A"])
                            
                            # Create visualization
                            comparison_data = []
                            for metric in ["P/E Ratio", "Forward P/E", "PEG Ratio", "Price/Sales", "Price/Book"]:
                                if metric in valuation_metrics and metric in selected_sector_avg:
                                    comparison_data.append({
                                        "Metric": metric,
                                        "Stock": valuation_metrics[metric],
                                        "Sector Avg": selected_sector_avg[metric]
                                    })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                
                                # Create side-by-side bar chart
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=comparison_df["Metric"],
                                    y=comparison_df["Stock"],
                                    name=ticker,
                                    marker_color='blue'
                                ))
                                fig.add_trace(go.Bar(
                                    x=comparison_df["Metric"],
                                    y=comparison_df["Sector Avg"],
                                    name=f"{sector} Sector Average",
                                    marker_color='gray'
                                ))
                                
                                fig.update_layout(
                                    title=f"Valuation Metrics vs {sector} Sector Averages",
                                    xaxis_title="Metric",
                                    yaxis_title="Value",
                                    barmode='group'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **What this chart shows:** Comparison of key valuation metrics for this company versus the sector average.
                                
                                **How to interpret:**
                                - Higher metrics compared to sector average might indicate overvaluation
                                - Lower metrics compared to sector average might indicate undervaluation
                                - Consider company growth rates and profitability when interpreting these metrics
                                """)
                            
                            # Display all valuation metrics in a table
                            st.subheader("All Valuation Metrics")
                            
                            # Convert to DataFrame for better display
                            valuation_df = pd.DataFrame.from_dict(valuation_metrics, orient='index', columns=["Value"])
                            valuation_df.index.name = "Metric"
                            
                            # Format large values
                            for idx, row in valuation_df.iterrows():
                                if idx in ["Market Cap", "Enterprise Value"] and isinstance(row["Value"], (int, float)):
                                    if row["Value"] >= 1e9:
                                        valuation_df.at[idx, "Value"] = f"${row['Value']/1e9:.2f}B"
                                    else:
                                        valuation_df.at[idx, "Value"] = f"${row['Value']/1e6:.2f}M"
                            
                            st.dataframe(valuation_df)
                            
                            # DCF Valuation (simplified example)
                            st.subheader("Discounted Cash Flow Valuation")
                            
                            # Get historical financials
                            try:
                                cash_flow = stock.cashflow
                                if not cash_flow.empty and "Free Cash Flow" in cash_flow.index:
                                    # Extract free cash flow
                                    fcf = cash_flow.loc["Free Cash Flow"]
                                    
                                    # Calculate average FCF growth
                                    fcf_growth = fcf.pct_change().mean()
                                    
                                    # Projection period
                                    years = 5
                                    
                                    # Terminal growth rate
                                    terminal_growth = 0.02  # 2%
                                    
                                    # Discount rate (WACC - would be calculated more accurately in a real model)
                                    discount_rate = 0.1  # 10%
                                    
                                    # Last year's FCF
                                    last_fcf = fcf.iloc[0]
                                    
                                    # Project future cash flows
                                    projected_fcf = []
                                    dcf_years = []
                                    for i in range(1, years + 1):
                                        projected_fcf.append(last_fcf * (1 + fcf_growth) ** i)
                                        dcf_years.append(f"Year {i}")
                                    
                                    # Terminal value
                                    terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
                                    
                                    # Discount the cash flows
                                    present_values = []
                                    for i, fcf in enumerate(projected_fcf):
                                        present_values.append(fcf / (1 + discount_rate) ** (i + 1))
                                    
                                    # Discount terminal value
                                    present_value_terminal = terminal_value / (1 + discount_rate) ** years
                                    
                                    # Sum up to get enterprise value
                                    enterprise_value = sum(present_values) + present_value_terminal
                                    
                                    # Get net debt
                                    balance_sheet = stock.balance_sheet
                                    
                                    total_debt = balance_sheet.loc["Total Debt"].iloc[0] if "Total Debt" in balance_sheet.index else 0
                                    cash_and_equivalents = balance_sheet.loc["Cash And Cash Equivalents"].iloc[0] if "Cash And Cash Equivalents" in balance_sheet.index else 0
                                    net_debt = total_debt - cash_and_equivalents
                                    
                                    # Equity value
                                    equity_value = enterprise_value - net_debt
                                    
                                    # Shares outstanding
                                    shares_outstanding = info.get('sharesOutstanding', None)
                                    
                                    if shares_outstanding:
                                        # Fair value per share
                                        fair_value_per_share = equity_value / shares_outstanding
                                        
                                        # Current price
                                        current_price = info.get('currentPrice', info.get('regularMarketPrice', None))
                                        
                                        if current_price:
                                            # Upside/downside
                                            upside = (fair_value_per_share / current_price - 1) * 100
                                            
                                            # Display DCF results
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("DCF Fair Value", f"${fair_value_per_share:.2f}")
                                            with col2:
                                                st.metric("Current Price", f"${current_price:.2f}")
                                            with col3:
                                                st.metric("Upside/Downside", f"{upside:.1f}%", delta=f"{upside:.1f}%")
                                            
                                            # Create visualization of projected cash flows
                                            fig = go.Figure()
                                            
                                            # Add projected FCF bars
                                            fig.add_trace(go.Bar(
                                                x=dcf_years,
                                                y=projected_fcf,
                                                name="Projected FCF",
                                                marker_color='blue'
                                            ))
                                            
                                            # Add terminal value
                                            fig.add_trace(go.Bar(
                                                x=["Terminal Value"],
                                                y=[terminal_value],
                                                name="Terminal Value",
                                                marker_color='green'
                                            ))
                                            
                                            fig.update_layout(
                                                title="DCF Valuation Components",
                                                xaxis_title="Year",
                                                yaxis_title="Value"
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Display model assumptions
                                            st.markdown("**DCF Model Assumptions:**")
                                            st.markdown(f"""
                                            - FCF Growth Rate: {fcf_growth:.1%} (based on historical average)
                                            - Projection Period: {years} years
                                            - Terminal Growth Rate: {terminal_growth:.1%}
                                            - Discount Rate (WACC): {discount_rate:.1%}
                                            """)
                                            
                                            st.markdown("""
                                            **Note:** This is a simplified DCF model for illustration purposes. A comprehensive DCF analysis would involve more detailed projections and analysis of the company's financial statements, growth drivers, and industry trends.
                                            """)
                                    else:
                                        st.warning("Could not calculate fair value - shares outstanding data unavailable.")
                                else:
                                    st.warning("Free Cash Flow data unavailable for DCF valuation.")
                            except Exception as e:
                                st.error(f"Error calculating DCF valuation: {str(e)}")
                        else:
                            st.warning("No valuation metrics available.")
                    
                    tab_index += 1
                
                # Technical Analysis Tab
                if show_advanced_technical:
                    with tabs[tab_index]:
                        st.subheader("Advanced Technical Analysis")
                        
                        # Allow user to select period
                        period_selected = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
                        
                        # Get advanced technical data
                        tech_data = generate_advanced_technical(ticker, period_selected)
                        
                        if tech_data is not None and not tech_data.empty:
                            # Create price chart with all overlays
                            fig = go.Figure()
                            
                            # Add price
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['Close'],
                                name="Price",
                                line=dict(color='black', width=2)
                            ))
                            
                            # Add moving averages
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['MA50'],
                                name="MA50",
                                line=dict(color='blue', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['MA200'],
                                name="MA200",
                                line=dict(color='red', width=1)
                            ))
                            
                            # Add Bollinger Bands
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['Upper_Band'],
                                name="Upper Band",
                                line=dict(color='green', width=1, dash='dash')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['MA20'],
                                name="MA20",
                                line=dict(color='orange', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=tech_data.index,
                                y=tech_data['Lower_Band'],
                                name="Lower Band",
                                line=dict(color='purple', width=1, dash='dash')
                            ))
                            
                            # Customize layout
                            fig.update_layout(
                                title=f"{ticker} Price Chart with Indicators",
                                xaxis_title="Date",
                                yaxis_title="Price (USD)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create tabs for different indicators
                            ind_tab1, ind_tab2, ind_tab3, ind_tab4 = st.tabs(["Momentum Indicators", "Trend Indicators", "Volatility", "Volume Analysis"])
                            
                            with ind_tab1:
                                # Create RSI and Stochastic Charts
                                fig = make_subplots(rows=2, cols=1, subplot_titles=["Relative Strength Index (RSI)", "Stochastic Oscillator"])
                                
                                # RSI Chart
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['RSI'],
                                    name="RSI",
                                    line=dict(color='blue')
                                ), row=1, col=1)
                                
                                # Add overbought/oversold lines
                                fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                                fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                                
                                # Stochastic Chart
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['%K'],
                                    name="%K",
                                    line=dict(color='blue')
                                ), row=2, col=1)
                                
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['%D'],
                                    name="%D",
                                    line=dict(color='red')
                                ), row=2, col=1)
                                
                                # Add overbought/oversold lines
                                fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
                                fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
                                
                                fig.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **Momentum Indicators Analysis:**
                                
                                **RSI (Relative Strength Index):**
                                - Measures the speed and change of price movements
                                - RSI > 70 suggests overbought conditions (potential reversal down)
                                - RSI < 30 suggests oversold conditions (potential reversal up)
                                
                                **Stochastic Oscillator:**
                                - Compares current price to price range over a period
                                - %K (blue) is the main line, %D (red) is a moving average of %K
                                - Values > 80 suggest overbought conditions
                                - Values < 20 suggest oversold conditions
                                - %K crossing above %D is a bullish signal
                                - %K crossing below %D is a bearish signal
                                """)
                            
                            with ind_tab2:
                                # Create MACD and ADX Charts
                                fig = make_subplots(rows=2, cols=1, subplot_titles=["MACD", "Average Directional Index (ADX)"])
                                
                                # MACD Chart
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['MACD'],
                                    name="MACD",
                                    line=dict(color='blue')
                                ), row=1, col=1)
                                
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['Signal'],
                                    name="Signal",
                                    line=dict(color='red')
                                ), row=1, col=1)
                                
                                fig.add_trace(go.Bar(
                                    x=tech_data.index,
                                    y=tech_data['MACD_Histogram'],
                                    name="Histogram",
                                    marker_color=['green' if val >= 0 else 'red' for val in tech_data['MACD_Histogram']]
                                ), row=1, col=1)
                                
                                # ADX Chart
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['ADX'],
                                    name="ADX",
                                    line=dict(color='purple')
                                ), row=2, col=1)
                                
                                # Add threshold line
                                fig.add_hline(y=25, line_dash="dash", line_color="blue", row=2, col=1)
                                
                                fig.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **Trend Indicators Analysis:**
                                
                                **MACD (Moving Average Convergence Divergence):**
                                - Shows the relationship between two moving averages
                                - MACD Line (blue): Difference between 12-day and 26-day EMAs
                                - Signal Line (red): 9-day EMA of the MACD Line
                                - Histogram: Difference between MACD and Signal lines
                                - MACD crossing above Signal line is bullish
                                - MACD crossing below Signal line is bearish
                                - Histogram changing from negative to positive shows momentum shift
                                
                                **ADX (Average Directional Index):**
                                - Measures the strength of a trend (not direction)
                                - ADX > 25 indicates a strong trend
                                - ADX < 25 indicates a weak or absent trend
                                - Rising ADX suggests increasing trend strength
                                - Falling ADX suggests decreasing trend strength
                                """)
                            
                            with ind_tab3:
                                # Create Bollinger Bands and ATR Chart
                                fig = make_subplots(rows=2, cols=1, subplot_titles=["Bollinger Bands", "Average True Range (ATR)"])
                                
                                # Bollinger Bands
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['Close'],
                                    name="Price",
                                    line=dict(color='black')
                                ), row=1, col=1)
                                
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['MA20'],
                                    name="MA20",
                                    line=dict(color='blue')
                                ), row=1, col=1)
                                
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['Upper_Band'],
                                    name="Upper Band",
                                    line=dict(color='green', dash='dash')
                                ), row=1, col=1)
                                
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['Lower_Band'],
                                    name="Lower Band",
                                    line=dict(color='red', dash='dash')
                                ), row=1, col=1)
                                
                                # ATR (Average True Range)
                                # Calculate ATR
                                tr1 = tech_data['High'] - tech_data['Low']
                                tr2 = abs(tech_data['High'] - tech_data['Close'].shift())
                                tr3 = abs(tech_data['Low'] - tech_data['Close'].shift())
                                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                                atr = tr.rolling(window=14).mean()
                                
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=atr,
                                    name="ATR (14)",
                                    line=dict(color='purple')
                                ), row=2, col=1)
                                
                                fig.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **Volatility Indicators Analysis:**
                                
                                **Bollinger Bands:**
                                - Consists of a middle band (20-day MA) and two outer bands (±2 standard deviations)
                                - Bands widen during high volatility and narrow during low volatility
                                - Price reaching the upper band may indicate overbought conditions
                                - Price reaching the lower band may indicate oversold conditions
                                - "Bollinger Band squeeze" (narrowing bands) often precedes significant price movements
                                
                                **ATR (Average True Range):**
                                - Measures market volatility
                                - Higher ATR indicates higher volatility
                                - Lower ATR indicates lower volatility
                                - Often used to set stop-loss levels or position sizing
                                - Not directional - only measures volatility magnitude
                                """)
                            
                            with ind_tab4:
                                # Create OBV Chart
                                fig = make_subplots(rows=2, cols=1, subplot_titles=["On-Balance Volume (OBV)", "Volume"])
                                
                                # OBV Chart
                                fig.add_trace(go.Scatter(
                                    x=tech_data.index,
                                    y=tech_data['OBV'],
                                    name="OBV",
                                    line=dict(color='blue')
                                ), row=1, col=1)
                                
                                # Volume Chart
                                fig.add_trace(go.Bar(
                                    x=tech_data.index,
                                    y=tech_data['Volume'],
                                    name="Volume",
                                    marker_color=['green' if tech_data['Close'][i] > tech_data['Close'][i-1] else 'red' for i in range(1, len(tech_data['Close']))] + ['blue']
                                ), row=2, col=1)
                                
                                fig.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **Volume Analysis:**
                                
                                **On-Balance Volume (OBV):**
                                - Cumulative indicator that adds volume on up days and subtracts on down days
                                - Used to confirm price trends or spot divergences
                                - Rising OBV with rising price confirms uptrend
                                - Falling OBV with falling price confirms downtrend
                                - OBV rising while price falling suggests potential bullish reversal
                                - OBV falling while price rising suggests potential bearish reversal
                                
                                **Volume:**
                                - Green bars indicate volume on up days
                                - Red bars indicate volume on down days
                                - Higher volume on up days suggests strong buying                        st.dataframe(peer_data)
                        
                        # Create peer comparison charts
                        st.subheader("Valuation Comparison")
                        
                        # P/E Ratio comparison
                        if 'pe_ratio' in peer_data.columns:
                            pe_data = peer_data.dropna(subset=['pe_ratio']).sort_values('pe_ratio')
                            if not pe_data.empty:
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=pe_data.index,
                                    y=pe_data['pe_ratio'],
                                    marker_color=['red' if x == ticker else 'blue' for x in pe_data.index]
                                ))
                                fig.update_layout(title="P/E Ratio Comparison", xaxis_title="Company", yaxis_title="P/E Ratio")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **What this chart shows:** Comparison of Price-to-Earnings (P/E) ratios across industry peers.
                                
                                **How to interpret:**
                                - Lower P/E ratio may indicate a more attractive valuation relative to peers
                                - Higher P/E ratio might suggest higher growth expectations or overvaluation
                                - Consider why a company's P/E differs significantly from its peers (growth prospects, risks, etc.)
                                - The current stock being analyzed is highlighted in red
                                """)
                        
                        # Market Cap comparison
                        if 'market_cap' in peer_data.columns:
                            market_cap_data = peer_data.dropna(subset=['market_cap']).sort_values('market_cap', ascending=False)
                            if not market_cap_data.empty:
                                # Convert to billions for better display
                                market_cap_billions = market_cap_data['market_cap'] / 1e9
                                
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=market_cap_data.index,
                                    y=market_cap_billions,
                                    marker_color=['red' if x == ticker else 'blue' for x in market_cap_data.index]
                                ))
                                fig.update_layout(title="Market Capitalization Comparison", xaxis_title="Company", yaxis_title="Market Cap (Billions $)")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **What this chart shows:** Comparison of market capitalizations across industry peers.
                                
                                **How to interpret:**
                                - Larger market cap typically indicates established companies with more stable revenue/earnings
                                - Smaller market cap may indicate greater growth potential but also higher risk
                                - Consider relative size when comparing other metrics like P/E ratio
                                - The current stock being analyzed is highlighted in red
                                """)
                        
                        # Profit Margin comparison
                        if 'profit_margin' in peer_data.columns:
                            margin_data = peer_data.dropna(subset=['profit_margin']).sort_values('profit_margin', ascending=False)
                            if not margin_data.empty:
                                # Convert to percentage for better display
                                profit_margin_pct = margin_data['profit_margin'] * 100
                                
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=margin_data.index,
                                    y=profit_margin_pct,
                                    marker_color=['red' if x == ticker else 'blue' for x in margin_data.index]
                                ))
                                fig.update_layout(title="Profit Margin Comparison", xaxis_title="Company", yaxis_title="Profit Margin (%)")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **What this chart shows:** Comparison of profit margins across industry peers.
                                
                                **How to interpret:**
                                - Higher profit margins generally indicate more efficient operations or stronger pricing power
                                - Lower margins might indicate intense competition or less efficient operations
                                - Consistent margins significantly above peers may indicate competitive advantages
                                - The current stock being analyzed is highlighted in red
                                """)
                        
                        # Download peer comparison data
                        csv = peer_data.to_csv()
                        st.download_button("Download Peer Comparison Data", csv, f"{ticker}_peer_comparison.csv", "text/csv")
                    else:
                        st.warning("No peer comparison data available")

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
        if not tickers:
            st.error("Please enter at least one ticker symbol.")
            return
            
        with st.spinner("Analyzing portfolio..."):
            portfolio_data = get_portfolio_data(tickers, period)
            if portfolio_data is None or portfolio_data.empty:
                st.error("Could not fetch portfolio data.")
            else:
                st.header("Portfolio Analysis")
                tab1, tab2, tab3, tab4 = st.tabs(["Price Comparison", "Performance Statistics", "Correlation Analysis", "Portfolio Optimization"])
                
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
                    
                    # Add explanations
                    st.markdown("""
                    **What this chart shows:** Price comparison of multiple stocks over time, allowing you to see relative performance.
                    
                    **How to interpret:** 
                    - **Normalized view (if selected):** All stocks start at 100%, making it easy to compare percentage gains/losses
                    - **Non-normalized view:** Shows actual stock prices, useful for comparing price levels
                    - **Relative performance:** Stocks rising faster than others may indicate stronger company performance or sector trends
                    - **Correlation:** Stocks moving together suggest similar market factors affecting them
                    - **Divergence:** Stocks moving in opposite directions might indicate company-specific factors rather than market trends
                    - **Volatility:** Stocks with larger swings may be riskier but potentially offer higher returns
                    """)
                    
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
                                
                                # Additional metrics
                                # Calculate beta against a benchmark (assuming SPY is the benchmark)
                                try:
                                    spy_data = yf.download('SPY', period=period)['Close']
                                    spy_returns = spy_data.pct_change().dropna()
                                    
                                    # Align data
                                    aligned_data = pd.DataFrame({
                                        'stock': daily_returns,
                                        'market': spy_returns
                                    }).dropna()
                                    
                                    if len(aligned_data) > 5:
                                        covariance = aligned_data['stock'].cov(aligned_data['market'])
                                        market_variance = aligned_data['market'].var()
                                        beta = covariance / market_variance if market_variance != 0 else 1
                                    else:
                                        beta = 1
                                except:
                                    beta = 1
                                
                                # Calculate information ratio
                                try:
                                    excess_returns = aligned_data['stock'] - aligned_data['market']
                                    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
                                except:
                                    information_ratio = 0
                                
                                metrics[column] = {
                                    "Total Return (%)": total_return,
                                    "Annual Return (%)": annualized_return,
                                    "Volatility (%)": volatility,
                                    "Sharpe Ratio": sharpe_ratio,
                                    "Max Drawdown (%)": max_drawdown,
                                    "Beta": beta,
                                    "Information Ratio": information_ratio
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
                                        sharpe = metrics_df.loc[ticker, "Sharpe Ratio"]
                                        if pd.notna(volatility) and pd.notna(annual_return):
                                            # Size bubble by Sharpe ratio
                                            size = abs(sharpe) * 15 if not pd.isna(sharpe) else 15
                                            fig.add_trace(go.Scatter(
                                                x=[volatility], 
                                                y=[annual_return], 
                                                mode="markers+text", 
                                                name=ticker, 
                                                text=[ticker], 
                                                textposition="top center",
                                                marker=dict(
                                                    size=size,
                                                    color='green' if sharpe > 0 else 'red'
                                                )
                                            ))
                                    except Exception:
                                        continue
                                
                                # Add quadrant lines
                                fig.add_shape(type="line", x0=0, y0=0, x1=max(metrics_df["Volatility (%)"]) * 1.1, y1=0, line=dict(color="Black", width=1, dash="dash"))
                                fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=max(metrics_df["Annual Return (%)"]) * 1.1, line=dict(color="Black", width=1, dash="dash"))
                                
                                fig.update_layout(
                                    title="Risk/Return Profile", 
                                    xaxis_title="Volatility (Risk) %", 
                                    yaxis_title="Annual Return %",
                                    annotations=[
                                        dict(x=max(metrics_df["Volatility (%)"]) * 0.25, y=max(metrics_df["Annual Return (%)"]) * 0.75, text="Low Risk / High Return", showarrow=False),
                                        dict(x=max(metrics_df["Volatility (%)"]) * 0.75, y=max(metrics_df["Annual Return (%)"]) * 0.75, text="High Risk / High Return", showarrow=False),
                                        dict(x=max(metrics_df["Volatility (%)"]) * 0.25, y=max(metrics_df["Annual Return (%)"]) * 0.25, text="Low Risk / Low Return", showarrow=False),
                                        dict(x=max(metrics_df["Volatility (%)"]) * 0.75, y=max(metrics_df["Annual Return (%)"]) * 0.25, text="High Risk / Low Return", showarrow=False),
                                    ]
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add explanations
                                st.markdown("""
                                **What this chart shows:** The risk-return relationship for each stock in your portfolio, plotted as volatility (risk) vs. annual return.
                                
                                **Key terms:**
                                - **Annual Return (%):** The average yearly return of the stock
                                - **Volatility (%):** How much the stock price fluctuates (standard deviation of returns)
                                - **Bubble size:** Represents Sharpe ratio (larger is better)
                                - **Bubble color:** Green for positive Sharpe ratio, red for negative
                                
                                **How to interpret:** 
                                - **Top-left quadrant (low risk, high return):** Ideal investments
                                - **Top-right quadrant (high risk, high return):** Growth or speculative investments
                                - **Bottom-left quadrant (low risk, low return):** Conservative or defensive investments
                                - **Bottom-right quadrant (high risk, low return):** Problematic investments to reconsider
                                
                                - A well-diversified portfolio typically contains a mix of investments with different risk-return profiles
                                """)
                                
                                # Total & Drawdown chart
                                fig2 = make_subplots(rows=2, cols=1, subplot_titles=("Total Return", "Maximum Drawdown"))
                                
                                # Sort for better visualization
                                sorted_returns = metrics_df.sort_values("Total Return (%)", ascending=False)
                                
                                # Total Return bars
                                fig2.add_trace(
                                    go.Bar(
                                        x=sorted_returns.index,
                                        y=sorted_returns["Total Return (%)"],
                                        marker_color=["green" if x >= 0 else "red" for x in sorted_returns["Total Return (%)"]],
                                        name="Total Return"
                                    ),
                                    row=1, col=1
                                )
                                
                                # Max Drawdown bars
                                sorted_drawdowns = metrics_df.sort_values("Max Drawdown (%)")
                                fig2.add_trace(
                                    go.Bar(
                                        x=sorted_drawdowns.index,
                                        y=sorted_drawdowns["Max Drawdown (%)"],
                                        marker_color="red",
                                        name="Max Drawdown"
                                    ),
                                    row=2, col=1
                                )
                                
                                fig2.update_layout(height=600, showlegend=False)
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                # Additional metrics charts
                                fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Beta", "Sharpe Ratio"))
                                
                                # Beta chart
                                sorted_beta = metrics_df.sort_values("Beta")
                                fig3.add_trace(
                                    go.Bar(
                                        x=sorted_beta.index,
                                        y=sorted_beta["Beta"],
                                        marker_color="blue",
                                        name="Beta"
                                    ),
                                    row=1, col=1
                                )
                                
                                # Add reference line for market beta = 1
                                fig3.add_shape(
                                    type="line",
                                    x0=-0.5, 
                                    y0=1, 
                                    x1=len(sorted_beta)-0.5, 
                                    y1=1,
                                    line=dict(color="red", width=2, dash="dash"),
                                    row=1, col=1
                                )
                                
                                # Sharpe Ratio chart
                                sorted_sharpe = metrics_df.sort_values("Sharpe Ratio", ascending=False)
                                fig3.add_trace(
                                    go.Bar(
                                        x=sorted_sharpe.index,
                                        y=sorted_sharpe["Sharpe Ratio"],
                                        marker_color=["green" if x >= 0 else "red" for x in sorted_sharpe["Sharpe Ratio"]],
                                        name="Sharpe Ratio"
                                    ),
                                    row=1, col=2
                                )
                                
                                fig3.update_layout(height=400, showlegend=False)
                                st.plotly_chart(fig3, use_container_width=True)
                                
                                try:
                                    buffer = io.BytesIO()
                                    fig.write_image(buffer, format="png")
                                    buffer.seek(0)
                                    st.download_button("Download Risk/Return Chart", buffer.getvalue(), "portfolio_risk_return.png", "image/png")
                                    csv = metrics_df.to_csv()
                                    st.download_button("Download Statistics", csv, "portfolio_stats.csv", "text/csv")
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
                                
                                # Display correlation matrix with gradient coloring
                                st.markdown("### Correlation Matrix")
                                st.dataframe(correlation.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format("{:.2f}"))
                                
                                # Display correlation heatmap
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
                                
                                # Add explanations
                                st.markdown("""
                                **What this chart shows:** The correlation heatmap shows how the returns of different stocks move in relation to each other.
                                
                                **Key terms:**
                                - **Correlation:** Ranges from -1 to +1
                                - **+1 (dark blue):** Perfect positive correlation (stocks move exactly together)
                                - **0 (white):** No correlation (stocks move independently)
                                - **-1 (dark red):** Perfect negative correlation (stocks move exactly opposite)
                                
                                **How to interpret:** 
                                - **Positive correlation (blue):** Stocks tend to move in the same direction
                                - **Negative correlation (red):** Stocks tend to move in opposite directions
                                - **Portfolio diversification:** Combining stocks with low or negative correlations can reduce overall portfolio risk
                                - **Sector relationships:** Stocks in the same sector often show higher correlation
                                - **Diversification benefit:** The more red/white areas in your heatmap, the better diversified your portfolio
                                """)
                                
                                # Create hierarchical clustering of correlations
                                import scipy.cluster.hierarchy as sch
                                from scipy.spatial.distance import squareform
                                
                                # Convert correlation to distance matrix
                                dist = 1 - abs(correlation)
                                
                                # Apply hierarchical clustering
                                linkage = sch.linkage(squareform(dist), method='ward')
                                
                                # Get clustered index order
                                dendro = sch.dendrogram(linkage, no_plot=True)
                                clustered_index = correlation.index[dendro['leaves']]
                                
                                # Create clustered correlation heatmap
                                clustered_corr = correlation.loc[clustered_index, clustered_index]
                                
                                fig2 = go.Figure(data=go.Heatmap(
                                    z=clustered_corr.values,
                                    x=clustered_corr.columns,
                                    y=clustered_corr.index,
                                    colorscale='RdBu',
                                    zmin=-1,
                                    zmax=1,
                                    text=[[f"{val:.2f}" for val in row] for row in clustered_corr.values],
                                    texttemplate="%{text}"
                                ))
                                fig2.update_layout(title="Clustered Correlation Heatmap", width=600, height=600)
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                st.markdown("""
                                **What this chart shows:** Same correlation data as above, but reorganized to group similar stocks together.
                                
                                **How to interpret:**
                                - Clusters (blocks) of blue indicate groups of stocks that tend to move together
                                - These clusters often represent stocks in the same sector or affected by similar factors
                                - An ideal diversified portfolio would include stocks from different clusters
                                """)
                                
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
                
                with tab4:
                    if show_optimization and len(portfolio_data.columns) > 1:
                        st.subheader("Portfolio Optimization")
                        
                        # Risk-free rate input
                        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 3.0, 0.1) / 100
                        
                        try:
                            # Run portfolio optimization
                            optimization_results = optimize_portfolio(portfolio_data, risk_free_rate)
                            
                            if optimization_results:
                                # Display optimal portfolio allocation
                                st.markdown("### Optimal Portfolio (Maximum Sharpe Ratio)")
                                
                                # Create pie chart of optimal weights
                                optimal_weights = optimization_results["optimal_weights"]
                                
                                # Sort weights for better visualization
                                sorted_weights = {k: v for k, v in sorted(optimal_weights.items(), key=lambda item: item[1], reverse=True)}
                                
                                fig = go.Figure(data=[go.Pie(
                                    labels=list(sorted_weights.keys()),
                                    values=list(sorted_weights.values()),
                                    hole=0.4,
                                    textinfo='label+percent'
                                )])
                                
                                fig.update_layout(title="Optimal Portfolio Allocation")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display optimal portfolio metrics
                                optimal_metrics = optimization_results["optimal_metrics"]
                                st.markdown(f"""
                                **Expected Annual Return:** {optimal_metrics['return'] * 100:.2f}%  
                                **Expected Volatility:** {optimal_metrics['risk'] * 100:.2f}%  
                                **Sharpe Ratio:** {optimal_metrics['sharpe']:.2f}
                                """)
                                
                                # Display minimum volatility portfolio
                                st.markdown("### Minimum Volatility Portfolio")
                                
                                # Create pie chart of min vol weights
                                min_vol_weights = optimization_results["min_vol_weights"]
                                
                                # Sort weights for better visualization
                                sorted_min_vol_weights = {k: v for k, v in sorted(min_vol_weights.items(), key=lambda item: item[1], reverse=True)}
                                
                                fig2 = go.Figure(data=[go.Pie(
                                    labels=list(sorted_min_vol_weights.keys()),
                                    values=list(sorted_min_vol_weights.values()),
                                    hole=0.4,
                                    textinfo='label+percent'
                                )])
                                
                                fig2.update_layout(title="Minimum Volatility Portfolio Allocation")
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                # Display min vol portfolio metrics
                                min_vol_metrics = optimization_results["min_vol_metrics"]
                                st.markdown(f"""
                                **Expected Annual Return:** {min_vol_metrics['return'] * 100:.2f}%  
                                **Expected Volatility:** {min_vol_metrics['risk'] * 100:.2f}%  
                                **Sharpe Ratio:** {min_vol_metrics['sharpe']:.2f}
                                """)
                                
                                # Plot efficient frontier
                                st.markdown("### Efficient Frontier")
                                
                                fig3 = go.Figure()
                                
                                # Add efficient frontier
                                results = optimization_results["all_results"]
                                
                                # Add scatter plot for all portfolios
                                fig3.add_trace(go.Scatter(
                                    x=results[:, 1] * 100,  # Convert to percentage
                                    y=results[:, 0] * 100,  # Convert to percentage
                                    mode='markers',
                                    marker=dict(
                                        size=5,
                                        color=results[:, 2],  # Sharpe ratio
                                        colorscale='Viridis',
                                        colorbar=dict(title="Sharpe Ratio"),
                                        line=dict(width=1)
                                    ),
                                    text=[f"Sharpe: {sharpe:.2f}" for sharpe in results[:, 2]],
                                    name='Simulated Portfolios'
                                ))
                                
                                # Add optimal portfolio point
                                fig3.add_trace(go.Scatter(
                                    x=[optimal_metrics['risk'] * 100],
                                    y=[optimal_metrics['return'] * 100],
                                    mode='markers',
                                    marker=dict(
                                        size=15,
                                        color='red',
                                        symbol='star'
                                    ),
                                    name='Maximum Sharpe Portfolio'
                                ))
                                
                                # Add minimum volatility portfolio point
                                fig3.add_trace(go.Scatter(
                                    x=[min_vol_metrics['risk'] * 100],
                                    y=[min_vol_metrics['return'] * 100],
                                    mode='markers',
                                    marker=dict(
                                        size=15,
                                        color='green',
                                        symbol='star'
                                    ),
                                    name='Minimum Volatility Portfolio'
                                ))
                                
                                # Add individual assets
                                for i, asset in enumerate(optimization_results["assets"]):
                                    # We need to calculate individual asset metrics
                                    if asset in metrics:
                                        asset_vol = metrics[asset]["Volatility (%)"]
                                        asset_ret = metrics[asset]["Annual Return (%)"]
                                        
                                        fig3.add_trace(go.Scatter(
                                            x=[asset_vol],
                                            y=[asset_ret],
                                            mode='markers+text',
                                            marker=dict(
                                                size=10,
                                                color='blue'
                                            ),
                                            text=[asset],
                                            textposition="bottom center",
                                                                            st.plotly_chart(fig, use_container_width=True)
                                
                                # Add explanations
                                st.markdown("""
                                **What this chart shows:** This chart compares stockholders' equity (what the company is worth on paper) and market capitalization (what investors think the company is worth).
                                
                                **Key terms:**
                                - **Stockholders' Equity:** The company's assets minus its liabilities (book value)
                                - **Market Cap:** The total value of all outstanding shares (stock price × number of shares)
                                
                                **How to interpret:** 
                                - When market cap grows faster than equity, investors are becoming more optimistic about future growth
                                - When equity grows faster than market cap, the stock might be becoming undervalued
                                - Large gaps between the two might indicate the market's expectation of future performance differs from current financial statements
                                """)
                                
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
                                    
                                    # Add explanations
                                    st.markdown("""
                                    **What this chart shows:** The Price-to-Book (P/B) ratio, which compares the company's market value to its book value.
                                    
                                    **Key terms:**
                                    - **P/B Ratio:** Market Cap divided by Stockholders' Equity
                                    
                                    **How to interpret:** 
                                    - P/B ratio > 1: Investors value the company more than its accounting value (common for growth companies)
                                    - P/B ratio < 1: The stock might be undervalued, or the company could have underlying problems
                                    - Industry average: Different industries have different typical P/B ratios (tech companies often have higher P/B than manufacturing)
                                    - Changes over time: An increasing P/B suggests growing investor confidence; decreasing suggests the opposite
                                    """)
                                    
                                    try:
                                        buffer = io.BytesIO()
                                        fig.write_image(buffer, format="png")
                                        buffer.seek(0)
                                        st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_pb_ratio.png", "image/png")
                                    except Exception as e:
                                        st.warning(f"Could not generate downloadable chart: {str(e)}")
                            else:
                                st.warning("No equity/market cap data available")
                        
                        with fin_tab4:
                            if "quarterly_financials" in additional_data and not additional_data["quarterly_financials"].empty:
                                st.subheader("Quarterly Financial Data")
                                st.dataframe(additional_data["quarterly_financials"])
                                
                                # Create quarterly revenue growth chart
                                if "Total Revenue" in additional_data["quarterly_financials"].index:
                                    quarterly_revenue = additional_data["quarterly_financials"].loc["Total Revenue"]
                                    
                                    # Calculate QoQ growth
                                    revenue_growth = quarterly_revenue.pct_change() * 100
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(
                                        x=quarterly_revenue.index,
                                        y=quarterly_revenue,
                                        name="Quarterly Revenue",
                                        marker_color="blue"
                                    ))
                                    
                                    # Add trend line
                                    fig.add_trace(go.Scatter(
                                        x=quarterly_revenue.index,
                                        y=quarterly_revenue,
                                        mode='lines',
                                        name='Trend',
                                        line=dict(color='red', width=1, dash='dot')
                                    ))
                                    
                                    fig.update_layout(
                                        title="Quarterly Revenue",
                                        xaxis_title="Quarter",
                                        yaxis_title="Revenue"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Growth rate chart
                                    fig2 = go.Figure()
                                    fig2.add_trace(go.Bar(
                                        x=revenue_growth.index,
                                        y=revenue_growth,
                                        name="QoQ Growth Rate",
                                        marker_color=['green' if x >= 0 else 'red' for x in revenue_growth]
                                    ))
                                    
                                    fig2.update_layout(
                                        title="Quarter-over-Quarter Revenue Growth Rate (%)",
                                        xaxis_title="Quarter",
                                        yaxis_title="Growth Rate (%)"
                                    )
                                    
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                try:
                                    csv_data = additional_data["quarterly_financials"].to_csv()
                                    st.download_button("Download Quarterly Financial Data", csv_data, f"{ticker}_quarterly_financials.csv", "text/csv")
                                except Exception as e:
                                    st.warning(f"Could not generate downloadable content: {str(e)}")
                            else:
                                st.warning("No quarterly financial data available")
                    else:
                        st.warning("No financial data available to analyze")
                
                with tab3:
                    if show_technical and (tech_data is not None or advanced_tech_data is not None) and ((tech_data is not None and not tech_data.empty) or (advanced_tech_data is not None and not advanced_tech_data.empty)):
                        st.subheader("Technical Indicators")
                        
                        # Use available data
                        display_data = tech_data if tech_data is not None and not tech_data.empty else advanced_tech_data
                        
                        if "Moving Averages" in tech_indicators:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=display_data.index, y=display_data['Close'], name="Price"))
                            if 'MA50' in display_data.columns:
                                fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA50'], name="MA50"))
                            if 'MA200' in display_data.columns:
                                fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA200'], name="MA200"))
                            fig.update_layout(title="Price and Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanations
                            st.markdown("""
                            **What this chart shows:** This chart displays the stock price with moving averages over different time periods.
                            
                            **Key terms:**
                            - **Moving Average (MA):** The average price over a specified number of days
                            - **MA50:** Average price over the last 50 trading days (about 2.5 months)
                            - **MA200:** Average price over the last 200 trading days (about 10 months)
                            
                            **How to interpret:** 
                            - Moving averages smooth out day-to-day price fluctuations to show the overall trend
                            - When price crosses above MA50 or MA200, it may signal a bullish trend (prices rising)
                            - When price crosses below MA50 or MA200, it may signal a bearish trend (prices falling)
                            - When MA50 crosses above MA200 (Golden Cross), it's often considered a strong bullish signal
                            - When MA50 crosses below MA200 (Death Cross), it's often considered a strong bearish signal
                            """)
                            
                            try:
                                buffer = io.BytesIO()
                                fig.write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_ma.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                        
                        if "RSI" in tech_indicators and 'RSI' in display_data.columns:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=display_data.index, y=display_data['RSI'], name="RSI", line=dict(color="purple")))
                            fig.add_hline(y=70, line_dash="dash", line_color="red")
                            fig.add_hline(y=30, line_dash="dash", line_color="green")
                            fig.update_layout(title="Relative Strength Index (RSI)", xaxis_title="Date", yaxis_title="RSI", yaxis_range=[0, 100])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanations
                            st.markdown("""
                            **What this chart shows:** The Relative Strength Index (RSI), a momentum indicator that measures the speed and change of price movements.
                            
                            **Key terms:**
                            - **RSI:** Oscillates between 0 and 100, showing overbought or oversold conditions
                            - **Overbought (above 70):** The stock may be overvalued and due for a price correction
                            - **Oversold (below 30):** The stock may be undervalued and due for a price recovery
                            
                            **How to interpret:** 
                            - RSI above 70 (above red line): Stock might be overbought, suggesting a potential sell opportunity
                            - RSI below 30 (below green line): Stock might be oversold, suggesting a potential buy opportunity
                            - RSI trending upward or downward: Can indicate momentum even before price movements are obvious
                            - Divergence between RSI and price: If price makes new highs but RSI doesn't, might signal a weakening trend
                            """)
                            
                            try:
                                buffer = io.BytesIO()
                                fig.write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_rsi.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                        
                        if "MACD" in tech_indicators and all(col in display_data.columns for col in ['MACD', 'Signal', 'MACD_Histogram']):
                            fig = make_subplots(rows=2, cols=1, subplot_titles=("MACD", "Histogram"), vertical_spacing=0.1)
                            fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MACD'], name="MACD", line=dict(color="blue")), row=1, col=1)
                            fig.add_trace(go.Scatter(x=display_data.index, y=display_data['Signal'], name="Signal", line=dict(color="red")), row=1, col=1)
                            
                            # Create colors for histogram
                            colors = ["green" if x >= 0 else "red" for x in display_data['MACD_Histogram']]
                            fig.add_trace(go.Bar(x=display_data.index, y=display_data['MACD_Histogram'], name="Histogram", marker_color=colors), row=2, col=1)
                            fig.update_layout(height=600, title_text="MACD Analysis")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanations
                            st.markdown("""
                            **What this chart shows:** The Moving Average Convergence Divergence (MACD), a trend-following momentum indicator.
                            
                            **Key terms:**
                            - **MACD Line:** The difference between the 12-day and 26-day exponential moving averages (blue line)
                            - **Signal Line:** 9-day exponential moving average of the MACD line (red line)
                            - **Histogram:** Difference between MACD line and signal line (green/red bars)
                            
                            **How to interpret:** 
                            - MACD crossing above signal line: Bullish signal suggesting potential price increase
                            - MACD crossing below signal line: Bearish signal suggesting potential price decrease
                            - Histogram bars changing from red to green: Momentum shifting from negative to positive
                            - Histogram bars changing from green to red: Momentum shifting from positive to negative
                            - Divergence between MACD and price: If price makes new highs/lows but MACD doesn't, might signal a trend weakening
                            """)
                            
                            try:
                                buffer = io.BytesIO()
                                fig.write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_macd.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                        
                        if "Bollinger Bands" in tech_indicators and all(col in display_data.columns for col in ['Close', 'MA20', 'Upper_Band', 'Lower_Band']):
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=display_data.index, y=display_data['Close'], name="Price", line=dict(color="blue")))
                            fig.add_trace(go.Scatter(x=display_data.index, y=display_data['MA20'], name="MA20", line=dict(color="orange")))
                            fig.add_trace(go.Scatter(x=display_data.index, y=display_data['Upper_Band'], name="Upper Band", line=dict(color="green", dash="dash")))
                            fig.add_trace(go.Scatter(x=display_data.index, y=display_data['Lower_Band'], name="Lower Band", line=dict(color="red", dash="dash")))
                            fig.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price (USD)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanations
                            st.markdown("""
                            **What this chart shows:** Bollinger Bands, which measure price volatility and potential overbought or oversold conditions.
                            
                            **Key terms:**
                            - **Middle Band (MA20):** 20-day moving average of closing prices (orange line)
                            - **Upper Band:** Middle band plus two standard deviations (green dashed line)
                            - **Lower Band:** Middle band minus two standard deviations (red dashed line)
                            
                            **How to interpret:** 
                            - Price touching upper band: Potentially overbought condition (may reverse downward)
                            - Price touching lower band: Potentially oversold condition (may reverse upward)
                            - Bands narrowing: Low volatility, often occurs before major price movements
                            - Bands widening: Increasing volatility, often during strong trends or market uncertainty
                            - Price moving between bands: Normal trading conditions
                            - "Band squeeze" followed by breakout: Often signals the beginning of a new trend
                            """)
                            
                            try:
                                buffer = io.BytesIO()
                                fig.write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_bollinger.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                        
                        # Add advanced technical indicators
                        if advanced_tech_data is not None and not advanced_tech_data.empty:
                            # Stochastic Oscillator
                            if "Stochastic Oscillator" in tech_indicators and all(col in advanced_tech_data.columns for col in ['%K', '%D']):
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=advanced_tech_data.index, y=advanced_tech_data['%K'], name="%K", line=dict(color="blue")))
                                fig.add_trace(go.Scatter(x=advanced_tech_data.index, y=advanced_tech_data['%D'], name="%D", line=dict(color="red")))
                                fig.add_hline(y=80, line_dash="dash", line_color="red")
                                fig.add_hline(y=20, line_dash="dash", line_color="green")
                                fig.update_layout(title="Stochastic Oscillator", xaxis_title="Date", yaxis_title="Value", yaxis_range=[0, 100])
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **What this chart shows:** The Stochastic Oscillator is a momentum indicator comparing a security's closing price to its price range over a specific period.
                                
                                **Key terms:**
                                - **%K:** The main line (blue) showing current price relative to high-low range
                                - **%D:** 3-day moving average of %K (red line)
                                - **Overbought (>80):** May indicate a potential reversal downward
                                - **Oversold (<20):** May indicate a potential reversal upward
                                
                                **How to interpret:**
                                - %K crossing above %D: Bullish signal
                                - %K crossing below %D: Bearish signal
                                - Readings above 80: Potentially overbought conditions
                                - Readings below 20: Potentially oversold conditions
                                - Divergence from price action can signal potential reversals
                                """)
                            
                            # ADX
                            if "ADX" in tech_indicators and 'ADX' in advanced_tech_data.columns:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=advanced_tech_data.index, y=advanced_tech_data['ADX'], name="ADX", line=dict(color="purple")))
                                fig.add_hline(y=25, line_dash="dash", line_color="green")
                                fig.update_layout(title="Average Directional Index (ADX)", xaxis_title="Date", yaxis_title="ADX")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **What this chart shows:** The Average Directional Index (ADX) measures the strength of a trend, regardless of its direction.
                                
                                **Key terms:**
                                - **ADX:** Measures trend strength on a scale from 0 to 100
                                - **Strong trend (>25):** Indicates a strong trend is present
                                
                                **How to interpret:**
                                - ADX above 25: Strong trend (either up or down)
                                - ADX below 25: Weak or absent trend (might be consolidating)
                                - Rising ADX: Increasing trend strength
                                - Falling ADX: Decreasing trend strength
                                - ADX doesn't indicate direction, only strength of the trend
                                """)
                            
                            # On-Balance Volume
                            if "On-Balance Volume" in tech_indicators and 'OBV' in advanced_tech_data.columns:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=advanced_tech_data.index, y=advanced_tech_data['OBV'], name="OBV", line=dict(color="blue")))
                                fig.update_layout(title="On-Balance Volume (OBV)", xaxis_title="Date", yaxis_title="OBV")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **What this chart shows:** On-Balance Volume (OBV) is a momentum indicator that uses volume flow to predict changes in stock price.
                                
                                **Key terms:**
                                - **OBV:** Cumulative total of volume, adding volume on up days and subtracting on down days
                                
                                **How to interpret:**
                                - Rising OBV: Volume increasing on up days, suggesting positive pressure
                                - Falling OBV: Volume increasing on down days, suggesting negative pressure
                                - OBV trends often precede price movements
                                - Divergence (OBV rising while price falling or vice versa) can signal potential reversals
                                """)
                            
                            # Fibonacci Retracement
                            if 'Fib_50%' in advanced_tech_data.columns:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=advanced_tech_data.index, y=advanced_tech_data['Close'], name="Price", line=dict(color="black")))
                                fig.add_trace(go.Scatter(x=advanced_tech_data.index, y=advanced_tech_data['Fib_23.6%'], name="23.6%", line=dict(color="blue", dash="dash")))
                                fig.add_trace(go.Scatter(x=advanced_tech_data.index, y=advanced_tech_data['Fib_38.2%'], name="38.2%", line=dict(color="green", dash="dash")))
                                fig.add_trace(go.Scatter(x=advanced_tech_data.index, y=advanced_tech_data['Fib_50%'], name="50%", line=dict(color="orange", dash="dash")))
                                fig.add_trace(go.Scatter(x=advanced_tech_data.index, y=advanced_tech_data['Fib_61.8%'], name="61.8%", line=dict(color="red", dash="dash")))
                                fig.update_layout(title="Fibonacci Retracement Levels", xaxis_title="Date", yaxis_title="Price (USD)")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("""
                                **What this chart shows:** Fibonacci Retracement levels are horizontal lines that indicate potential support and resistance levels.
                                
                                **Key terms:**
                                - **23.6%, 38.2%, 50%, 61.8%:** Key Fibonacci levels that often act as support or resistance
                                
                                **How to interpret:**
                                - These levels often act as psychological support/resistance in market trends
                                - In downtrends, they can indicate potential reversal points for upward corrections
                                - In uptrends, they can indicate potential reversal points for downward corrections
                                - The 61.8% level is considered particularly significant
                                """)
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
                            
                            # Add explanations
                            st.markdown("""
                            **What this chart shows:** Key valuation ratios that help investors determine if a stock is fairly priced.
                            
                            **Key terms:**
                            - **P/E Ratio (Price-to-Earnings):** Stock price divided by earnings per share
                            - **P/B Ratio (Price-to-Book):** Market value divided by book value
                            
                            **How to interpret:** 
                            - **P/E Ratio:**
                              - Higher P/E: Investors expect higher growth or are willing to pay premium (often growth stocks)
                              - Lower P/E: Could indicate undervaluation or problems with the company (often value stocks)
                              - Industry comparison: Different industries have different typical P/E ranges
                            
                            - **P/B Ratio:**
                              - P/B > 1: Market values company more than its assets minus liabilities
                              - P/B < 1: Could be undervalued or have fundamental problems
                              - Financial/asset-heavy companies often evaluated heavily on P/B ratio
                            
                            - Changes over time often more meaningful than absolute values
                            """)
                            
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
                            
                            # Add explanations
                            st.markdown("""
                            **What this chart shows:** Performance and solvency ratios that measure profitability and financial health.
                            
                            **Key terms:**
                            - **ROE (Return on Equity):** Net income divided by shareholders' equity, expressed as a percentage
                            - **Debt to Equity:** Total debt divided by shareholders' equity
                            
                            **How to interpret:** 
                            - **ROE (Return on Equity):**
                              - Higher ROE: Company generates more profit from shareholder investments
                              - 15-20% or higher: Generally considered good
                              - Changes over time: Increasing ROE suggests improving efficiency
                              - Industry comparison: Some industries naturally have higher/lower ROE
                            
                            - **Debt to Equity:**
                              - Higher ratio: More debt relative to equity (higher risk but potentially higher returns)
                              - Lower ratio: Less debt relative to equity (lower risk but potentially lower returns)
                              - Ratio > 2.0: Often considered high debt load for many industries
                              - Some industries (utilities, financial) typically operate with higher debt levels
                            """)
                            
                            try:
                                buffer = io.BytesIO()
                                fig.write_image(buffer, format="png")
                                buffer.seek(0)
                                st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_performance_ratios.png", "image/png")
                            except Exception as e:
                                st.warning(f"Could not generate downloadable chart: {str(e)}")
                        
                        # Advanced Metrics from additional_data
                        if "key_metrics" in additional_data:
                            st.subheader("Additional Key Metrics")
                            metrics_to_display = {k: v for k, v in additional_data["key_metrics"].items() if v != 'N/A'}
                            
                            if metrics_to_display:
                                # Convert to DataFrame for better display
                                metrics_df = pd.DataFrame.from_dict(metrics_to_display, orient='index', columns=["Value"])
                                metrics_df.index.name = "Metric"
                                
                                # Format percentage values
                                for idx, val in metrics_df.iterrows():
                                    if idx in ["Profit Margin", "Operating Margin", "Return on Assets", "Return on Equity", "Dividend Yield", "Payout Ratio"]:
                                        if isinstance(val["Value"], (int, float)):
                                            metrics_df.at[idx, "Value"] = f"{val['Value'] * 100:.2f}%" if val["Value"] < 1 else f"{val['Value']:.2f}%"
                                
                                st.dataframe(metrics_df)
                                
                                # Create visualizations for key metrics
                                metrics_to_chart = ["PEG Ratio", "Forward P/E", "EV/EBITDA", "Profit Margin", "Operating Margin", "Return on Equity"]
                                chart_data = {k: v for k, v in metrics_to_display.items() if k in metrics_to_chart and isinstance(v, (int, float))}
                                
                                if chart_data:
                                    fig = go.Figure(data=[
                                        go.Bar(
                                            x=list(chart_data.keys()),
                                            y=list(chart_data.values()),
                                            marker_color='blue'
                                        )
                                    ])
                                    fig.update_layout(title="Key Financial Metrics", xaxis_title="Metric", yaxis_title="Value")
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No additional metrics available for this stock.")
                        
                        try:
                            csv = ratios_df.to_csv()
                            st.download_button("Download Ratio Data", csv, f"{ticker}_ratios.csv", "text/csv")
                        except Exception as e:
                            st.warning(f"Could not generate downloadable data: {str(e)}")
                    else:
                        st.warning("No financial ratio data available")
                
                with tab5:
                    if show_peer_comparison and peer_data is not None and not peer_data.empty:
                        st.subheader("Industry Peer Comparison")
                        
                        # Display the peer comparison data
                        st.def single_stock_analysis():
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
        with st.spinner(f"Analyzing {ticker}..."):
            # Get basic financial data
            financials, scale_ticker, stock_price_history, company_name, ratios_df = get_financials(ticker)
            
            if financials is None:
                st.error(company_name)
            else:
                # Get advanced data
                additional_data = get_additional_fundamentals(ticker)
                tech_data = generate_technical_indicators(ticker, period) if show_technical else None
                advanced_tech_data = generate_advanced_technical(ticker, period) if show_technical else None
                news_data = get_stock_news(ticker, 5) if show_news else []
                peer_data = industry_comparison(ticker) if show_peer_comparison else None
                
                # Compile data for AI insights
                insights_data = {
                    "financials": scale_ticker,
                    "technical": tech_data,
                    "advanced_technical": advanced_tech_data
                }
                
                # Generate AI insights
                ai_insights = generate_ai_insights(ticker, insights_data) if show_ai_insights else []
                
                # Display analysis header
                st.header(f"{company_name} ({ticker}) Analysis")
                
                # Current price and quick stats
                try:
                    current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
                    prev_close = yf.Ticker(ticker).history(period="2d")["Close"].iloc[-2]
                    price_change = current_price - prev_close
                    price_change_pct = (price_change / prev_close) * 100
                    
                    # Display current price stats in columns
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"${current_price:.2f}")
                    col2.metric("Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
                    
                    # Add more metrics from additional_data
                    if "key_metrics" in additional_data and "Forward P/E" in additional_data["key_metrics"]:
                        col3.metric("Forward P/E", f"{additional_data['key_metrics']['Forward P/E']}")
                    if "key_metrics" in additional_data and "Beta" in additional_data["key_metrics"]:
                        col4.metric("Beta", f"{additional_data['key_metrics']['Beta']}")
                except Exception as e:
                    st.warning(f"Could not fetch current price information: {str(e)}")
                
                # Display AI Insights
                if show_ai_insights and ai_insights:
                    st.subheader("AI-Powered Insights")
                    insights_col1, insights_col2 = st.columns([3, 1])
                    with insights_col1:
                        for insight in ai_insights:
                            st.markdown(f"- {insight}")
                    with insights_col2:
                        # Display sentiment gauge based on insights
                        sentiment = sum([1 if "✅" in insight else (-1 if "⚠️" in insight else 0) for insight in ai_insights])
                        sentiment_normalized = max(min((sentiment / max(len(ai_insights), 1) + 1) / 2, 1), 0)
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = sentiment_normalized * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Sentiment Score"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 33], 'color': "red"},
                                    {'range': [33, 66], 'color': "yellow"},
                                    {'range': [66, 100], 'color': "green"}
                                ]
                            }
                        ))
                        fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                
                # News & Sentiment
                if show_news and news_data:
                    st.subheader("Recent News & Sentiment")
                    for i, news in enumerate(news_data):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{news['title']}**")
                            st.markdown(f"*{news['publisher']} - {news['published'].strftime('%Y-%m-%d')}*")
                            st.markdown(f"[Read more]({news['link']})")
                        with col2:
                            sentiment = news['sentiment']
                            color = "green" if sentiment > 0.2 else "red" if sentiment < -0.2 else "orange"
                            st.markdown(
                                f"""
                                <div style="background-color:{color}; padding:10px; border-radius:5px;">
                                    <h4 style="margin:0; color:white; text-align:center;">{sentiment:.2f}</h4>
                                    <p style="margin:0; color:white; text-align:center;">Sentiment</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        st.markdown("---")
                
                # Main Analysis Tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stock Performance", "Financial Analysis", "Technical Indicators", "Financial Ratios", "Peer Comparison"])
                
                with tab1:
                    st.subheader("Stock Price History")
                    if not stock_price_history.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=stock_price_history.index, y=stock_price_history, name="Price", line=dict(color="blue")))
                        fig.update_layout(title=f"{time_period} Stock Price History", xaxis_title="Date", yaxis_title="Price (USD)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanations
                        st.markdown("""
                        **What this chart shows:** This chart displays the stock's price history over time. The line represents the closing price of the stock for each trading day.
                        
                        **How to interpret:** 
                        - Upward trends suggest increasing investor confidence or positive market sentiment
                        - Downward trends may indicate declining performance or negative market outlook
                        - Look for patterns like consistent growth, cyclical behavior, or sudden changes that might correlate with company news or market events
                        """)
                        
                        # Generate and provide download for chart
                        try:
                            buffer = io.BytesIO()
                            fig.write_image(buffer, format="png")
                            buffer.seek(0)
                            st.download_button("Download Chart", buffer.getvalue(), f"{ticker}_price.png", "image/png")
                        except Exception as e:
                            st.warning(f"Could not generate downloadable chart: {str(e)}")
                            
                        # Add volume chart
                        volume_data = yf.Ticker(ticker).history(period=period)["Volume"]
                        if not volume_data.empty:
                            fig2 = go.Figure()
                            fig2.add_trace(go.Bar(x=volume_data.index, y=volume_data, name="Volume", marker_color="rgba(0, 0, 255, 0.5)"))
                            fig2.update_layout(title="Trading Volume", xaxis_title="Date", yaxis_title="Volume")
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Add explanation for volume
                            st.markdown("""
                            **What this chart shows:** This chart displays the trading volume (number of shares traded) for the stock over time.
                            
                            **How to interpret:** 
                            - High volume often accompanies significant price movements
                            - Volume spikes may indicate important events or news
                            - Higher than average volume suggests stronger conviction behind a price move
                            - Lower volume during price increases might indicate weaker momentum
                            """)
                    else:
                        st.warning("No stock price history available")
                
                with tab2:
                    st.subheader("Financial Analysis")
                    if not scale_ticker.empty:
                        fin_tab1, fin_tab2, fin_tab3, fin_tab4 = st.tabs(["Revenue & Profit", "EBIT & EBITDA", "Equity & Market Cap", "Quarterly Data"])
                        
                        with fin_tab1:
                            available_metrics = [m for m in ["Total Revenue", "Gross Profit", "Net Income"] if m in scale_ticker.columns]
                            if available_metrics:
                                fig = go.Figure()
                                for metric in available_metrics:
                                    fig.add_trace(go.Scatter(x=scale_ticker.index, y=scale_ticker[metric], name=metric, mode='lines+markers'))
                                fig.update_layout(title="Revenue and Profit Trends", xaxis_title="Date", yaxis_title="Percentage Change")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add explanations
                                st.markdown("""
                                **What this chart shows:** This chart tracks the company's revenue, gross profit, and net income over time, showing their relative growth compared to their minimum values.
                                
                                **Key terms:**
                                - **Total Revenue:** The total money the company earns from selling its products or services
                                - **Gross Profit:** Revenue minus the direct costs of making products or providing services
                                - **Net Income:** The final profit after all expenses, taxes, and costs are subtracted from revenue
                                
                                **How to interpret:** 
                                - Growing revenue suggests the company is selling more products/services
                                - Widening gap between revenue and net income might indicate increasing costs
                                - Consistent growth across all metrics typically signals a healthy business
                                """)
                                
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
                                
                                # Add explanations
                                st.markdown("""
                                **What this chart shows:** This chart displays EBIT (Earnings Before Interest and Taxes) and EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization) over time.
                                
                                **Key terms:**
                                - **EBIT:** Profit before paying interest on debt and income taxes
                                - **EBITDA:** Profit before interest, taxes, and accounting for depreciation and amortization of assets
                                
                                **How to interpret:** 
                                - These metrics show a company's operational performance without the effects of capital structure, tax rates, or non-cash expenses
                                - Growing EBIT/EBITDA indicates improving operational efficiency
                                - EBITDA is often higher than EBIT since it adds back depreciation and amortization expenses
                                - Investors use these to compare companies in the same industry regardless of their debt levels or tax situations
                                """)
                                
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
    
    return insightsimport streamlit as st
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
from textblob import TextBlob
import requests
from io import StringIO

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

# --- Enhanced Functions ---
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
