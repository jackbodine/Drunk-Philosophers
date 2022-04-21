from urllib import response
import pandas as pd
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras

st.title("Drunk Philosophers")

data_raw = pd.read_csv("data/philosophy_data.csv")
data_raw = data_raw.drop(['sentence_spacy', 'original_publication_date', 'corpus_edition_date', 'sentence_length', 'sentence_lowered', 'lemmatized_str'], axis=1)
philosophers = list(dict.fromkeys(data_raw['author'].tolist()))

option = st.selectbox('Select a Philosopher', philosophers)

selected_data = data_raw.loc[data_raw['author'] == option]
text = ' '.join(selected_data['sentence_str'].tolist())

st.dataframe(selected_data)

st.write(text[:1000])

st.header("Generated Text")

model_loc = "./models/" + option
st.write("Loading ",model_loc, "...")
model = keras.models.load_model(model_loc)

seed = st.text_input("Seed", "Magic mirror on the wall, who's the fairest of them all?")
#seed = st.text_input("Seed", "Magic mirror on t")

# GARBAGE
seqlen = 40
seed = seed[:seqlen]

response_length = 100
diversity = 0.5
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def sample(preds, temperature=1.0):
    """Helper function to sample an index from a probability array."""
    preds = np.asarray(preds).astype('float64')
    preds = np.exp(np.log(preds) / temperature)  # softmax
    preds = preds / np.sum(preds)                #
    probas = np.random.multinomial(1, preds, 1)  # sample index
    return np.argmax(probas)    

# GARBAGE

bar = st.progress(0)

response_text = option + ": "
for i in range(response_length):
        x_pred = np.zeros((1, seqlen, len(chars)))
        for t, char in enumerate(seed):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)
        next_index = sample(preds[0, -1], diversity)
        next_char = indices_char[next_index]

        seed = seed[1:] + next_char

        response_text = response_text + next_char
        bar.progress(i/response_length)

st.write(response_text)
