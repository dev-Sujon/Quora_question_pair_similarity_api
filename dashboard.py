from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd
import streamlit as st
 
# Adjust the width of the Streamlit page
st.set_page_config(
    page_title="Use Pygwalker In Streamlit",
    layout="wide"
)
# Import your data
df = pd.read_csv("data/advfextract.csv")

st.write(df)
st.write(df.describe())
st.write("Shape of Dataset:",df.shape)
pyg_app = StreamlitRenderer(df)
 
pyg_app.explorer()
 