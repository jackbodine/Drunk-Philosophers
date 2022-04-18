import pandas as pd
import streamlit as st

st.title("Drunk Philosophers")

data_raw = pd.read_csv("data/philosophy_data.csv")
data_raw = data_raw.drop(['sentence_spacy', 'original_publication_date', 'corpus_edition_date', 'sentence_length', 'sentence_lowered', 'lemmatized_str'], axis=1)
philosophers = list(dict.fromkeys(data_raw['author'].tolist()))

option = st.selectbox('Select a Philosopher', philosophers)

selected_data = data_raw.loc[data_raw['author'] == option]
text = ' '.join(selected_data['sentence_str'].tolist())

st.dataframe(selected_data)

st.write(text[:2000])