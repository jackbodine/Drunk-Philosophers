import pandas as pd
import streamlit as st

st.title("Drunk Philosophers")

data_raw = pd.read_csv("data/philosophy_data.csv")
philosophers = list(dict.fromkeys(data_raw['author'].tolist()))

option = st.selectbox('Select a Philosopher', philosophers)

selected_data = data_raw.loc[data_raw['author'] == option]

st.dataframe(selected_data)