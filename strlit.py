from datetime import date, datetime
from matplotlib.pyplot import figtext, figure
import streamlit as st
import pandas as pd
import mplfinance as mpf
#from pandas_datareader import data as pdr
import numpy as np


def annotated(df, act):

    signal = []

    for i in range(len(df)):
        # print(f'J={df["lottery"][i]}, Close = {df["close"][i]}-Len:{len(df)}')
        if df["lottery"][i] == "T":
            if act == True:
                if df["close"][i] - df["open"][i] > 0:
                    signal.append(df["low"][i] * 0.99)
                else:
                    signal.append(np.nan)
            else:
                if df["close"][i] - df["open"][i] <= 0:
                    signal.append(df["high"][i] * 1.01)
                else:
                    signal.append(np.nan)
        else:
            signal.append(np.nan)
        # print(f'Signal = {i}:{signal[i]}')
    return signal

#st.experimental_memo(persist='disk')
def get_historical_data(symbol, start_date = None):
    #df = pdr.get_data_yahoo(symbol, start=start_date, end=datetime.now())
    """
    df = df.rename(columns = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj close', 'Volume': 'volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    """
    df = pd.read_csv(
    "CSV/"+symbol+"_BACKTEST.csv",
    index_col=0,
    parse_dates=True,
    )
    data_frame = pd.DataFrame(df)
    data_frame = data_frame.rename(columns = {'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_pric': 'close',
        'ddate': 'date', 'pdelivery': 'volume'})
    #print(data_frame.columns)
    data_frame["date"] = pd.to_datetime(data_frame["date"])
    data_frame = data_frame.set_index("date").sort_values("date")
    
    if start_date:
        data_frame = data_frame[data_frame.index >= start_date]
    return data_frame

st.title('Daily Jackpot Backtest Demo')

file = pd.read_csv("CSV/symbol.csv")
#print(file, file["symbol"])
symbols = file["symbol"].tolist()
#print(symbols)
c1, c2, c3 = st.columns([1,1,1])
with c1:
    #symbol = st.selectbox('Choose stock symbol', options=[ 'CIPLA', 'INFY', 'TATACHEM', 'ITC', 'RELIANCE', 'MARICO', 'SUNPHARMA', 'NAUKRI','TATASTEEL'], index=0)
    symbol = st.selectbox('Choose stock symbol', options=symbols, index=0)
with c2:
    date_from = st.date_input('Show data from', date(2021, 10, 1))
with c3:
    st.markdown('&nbsp;')
    #show_data = st.checkbox('Show data table', False)

st.markdown('---')

st.sidebar.subheader('Settings')
st.sidebar.caption('Adjust charts settings and then press apply')

with st.sidebar.form('settings_form'):
    show_nontrading_days = st.checkbox('Show non-trading days', False)
    # https://github.com/matplotlib/mplfinance/blob/master/examples/styles.ipynb
    chart_styles = [
        'default', 'binance', 'blueskies', 'brasil', 
        'charles', 'checkers', 'classic', 'yahoo',
        'mike', 'nightclouds', 'sas', 'starsandstripes'
    ]
    chart_style = st.selectbox('Chart style', options=chart_styles, index=chart_styles.index('yahoo'))
    chart_types = [
        'candle', 'ohlc', 'line'
    ]
    chart_type = st.selectbox('Chart type', options=chart_types, index=chart_types.index('candle'))

    mav1 = st.number_input('Moving Average 1', min_value=3, max_value=30, value=5, step=1)
    mav2 = st.number_input('Moving Average 2', min_value=3, max_value=30, value=20, step=1)
    mav3 = st.number_input('Moving Average 3', min_value=3, max_value=50, value=50, step=1)

    st.form_submit_button('Apply')

data = get_historical_data(symbol, str(date_from))
buysig = annotated(data, True)
sellsig = annotated(data, False)
apds = [
     mpf.make_addplot(buysig, type="scatter", markersize=50, marker="^", panel=0),
     mpf.make_addplot(sellsig, type="scatter", markersize=50, marker="v", panel=0)
]
fig, ax = mpf.plot(
    data,
    title=f'{symbol}, {date_from}',
    type=chart_type,
    addplot=apds,
    show_nontrading=show_nontrading_days,
    mav=(int(mav1),int(mav2),int(mav3)),
    volume=True,
    style=chart_style,
    figsize=(24,12),
    
    # Need this setting for Streamlit, see source code (line 778) here:
    # https://github.com/matplotlib/mplfinance/blob/master/src/mplfinance/plotting.py
    returnfig=True
)

st.pyplot(fig)

"""if show_data:
    st.markdown('---')
    st.dataframe(data)
"""
