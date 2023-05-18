"""
This is the file for experimenting weird ideas
@created 05/16/2023 - 12:45 PM
@author Kaiwen Zhou
"""
from datetime import timedelta

import impyute as impy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from Strategy_Base_Single_Period import Strategy_Base_Single_Period
if __name__ == '__main__':
    # TODO: NOT consistent with the one i have on notebook
    # Load data
    df_data = pd.read_csv('weekly_closing_prices_2012_2022.csv', index_col='Date')
    df_data.index = pd.to_datetime(df_data.index)

    ## Investing based on our allocation framework
    date_end_estimation=pd.to_datetime('2017-01-02')
    date_end_investing=pd.to_datetime('2022-12-26')
    number_of_allocation_needed = int((date_end_investing-date_end_estimation)/timedelta(weeks=1))


    ####################################################
    # List of tickers
    tickers = df_data.columns
    start_date = pd.to_datetime('2012-01-09')
    end_date = pd.to_datetime('2022-12-26')

    #####################################
    # Load necessary data for APT model #
    #####################################

    #### Get Market Capitalization data for each stock ####
    raw_data = pd.read_csv('CRSP_rawdata.csv')
    df_market_cap = pd.DataFrame()

    for ticker in tickers:
        ticker_data = raw_data[raw_data['Ticker']==ticker].copy()
        ticker_data.index = pd.to_datetime(ticker_data['DlyCalDt'])
        ticker_weekly_df = ticker_data.resample('W-MON').last()
        ticker_weekly_df = ticker_weekly_df[ticker_weekly_df.index <= end_date ]
        # print(ticker_weekly_df)
        df_market_cap.index = ticker_weekly_df.index
        df_market_cap[ticker] = ticker_weekly_df['DlyCap']

    ##### Get risk-free rate of return ####
    df_risk_free_rate = pd.read_csv('DTB3.csv')
    df_risk_free_rate.index = pd.to_datetime(df_risk_free_rate['DATE'])
    df_risk_free_rate = df_risk_free_rate.resample('W-MON').last()
    df_risk_free_rate = df_risk_free_rate[df_risk_free_rate.index <= end_date ]
    df_risk_free_rate = df_risk_free_rate.drop(columns=['DATE'])
    df_risk_free_rate = df_risk_free_rate.replace('.', np.nan)
    df_risk_free_rate = df_risk_free_rate.astype('float')
    # Apply imputation technique in dealing with missing values
    """
    For each set of missing indices, use the value of one row before(same column). 
    In the case that the missing value is the first row, look one row ahead instead. 
    If this next row is also NaN, look to the next row. Repeat until you find a row in this column that’s not NaN. 
    All the rows before will be filled with this value.
    """
    np.float = float
    df_risk_free_rate['DTB3'] = impy.imputation.ts.locf(df_risk_free_rate['DTB3'].values.reshape(1,-1), axis=0)
    df_risk_free_rate['risk free weekly'] = (1+df_risk_free_rate['DTB3']/100)**(1/12)-1
    df_risk_free_rate = df_risk_free_rate.drop(columns=['DTB3'])

    #### Get weekly rate of return for S&P500 index ####
    ticker = '^GSPC'
    # Download historical market data
    sp500 = yf.Ticker(ticker)
    hist = sp500.history(start=start_date, end=end_date)
    # Calculate daily returns
    hist['Return'] = hist['Close'].pct_change()
    # Convert daily returns to weekly returns
    hist.index = pd.to_datetime(hist.index)
    SP500_weekly_returns = hist['Return'].resample('W-MON').apply(lambda x: (1 + x).prod() - 1)
    SP500_weekly_returns = pd.DataFrame(SP500_weekly_returns)
    SP500_weekly_returns.index = [pd.to_datetime(datetime.date()) for datetime in SP500_weekly_returns.index]

    ###################
    # Start Investing #
    ###################
    # Create a DataFrame to store investment related data
    df_investing3 = pd.DataFrame(index=df_data.index, columns=['BL_APT'])
    df_investing3 = df_investing3[(df_investing3.index >= date_end_estimation)]
    df_investing3['BL_APT'][0] = 1000  # initial wealth
    tau_horizon = 3

    for i in range(0, number_of_allocation_needed, tau_horizon):
        initial_wealth = df_investing3['BL_APT'][i]
        print('initial wealth is: ', initial_wealth)
        # update the end date for estimation
        end_estimation_date = date_end_estimation + i*timedelta(weeks=1)
        v_sup = 100+i*2
        optimal_allocation = Strategy_Base_Single_Period().BL_APT( df_data=df_data,
                                                                   df_data_risk_free=df_risk_free_rate,
                                                                   df_data_market_cap=df_market_cap,
                                                                   df_data_SP500=SP500_weekly_returns,
                                                                   initial_wealth=initial_wealth,
                                                                   tau_estimate=1,
                                                                   tau_horizon=tau_horizon,
                                                                   date_start_estimation=pd.to_datetime('2012-01-09'),
                                                                   date_end_estimation=end_estimation_date,
                                                                   minimum_interval=timedelta(weeks=1),
                                                                   trailing_window=30,
                                                                   estimator_type='sample',
                                                                   obj_type='absolute wealth',
                                                                   asset_class='equity',
                                                                   CE_type='power',
                                                                   utility_coef=-9,
                                                                   N_MC=100,
                                                                   v_inf=1,
                                                                   v_sup=v_sup,
                                                                   v_step=1
                                                                   )

        ################# Calculate the PnL generated by holding the optimal portfolio
        # current (buy-in) price
        price_current = df_data[df_data.index==end_estimation_date].values.squeeze()
        # investment horizon (sell-off) price
        price_investment_horizon = df_data[df_data.index == end_estimation_date + tau_horizon*timedelta(weeks=1)].values.squeeze()
        # PnL generated by holding the optimal portfolio
        PnL = optimal_allocation@(price_investment_horizon-price_current)
        print(PnL)
        # update your absolute wealth
        updated_wealth = initial_wealth + PnL
        print('updated_wealth is: ', updated_wealth)
        df_investing3['BL_APT'][i+tau_horizon] = updated_wealth

    ####################
    ####################
    df_result3=df_investing3.dropna().copy()

    df_result3.plot()
    plt.show()
