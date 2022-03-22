import pandas as pd
import streamlit as st

import plotly.express as px


header_container = st.container()
stats_container = st.container()	
#######################################



# You can place things (titles, images, text, plots, dataframes, columns etc.) inside a container
with header_container:

	# for example a logo or a image that looks like a website header
	st.image('./streamlit template/logo.png')

	# different levels of text you can include in your app
	st.title("A cool new Streamlit app")
	st.header("Welcome!")
	st.subheader("This is a great app")
	st.write("cHello World1")