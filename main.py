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

# model_loc = "./models/" + option
# st.write("Loading ",model_loc, "...")
# model = keras.models.load_model(model_loc)

# seed = st.text_input("Seed", "Magic mirror on the wall, who's the fairest of them all?")

# # GARBAGE
# while len(seed) < 40:
#     seed = seed + " "

# seqlen = 40
# seed = seed[:seqlen]

# response_length = 100
# diversity = 0.5
# chars = sorted(list(set(text)))
# char_indices = dict((c, i) for i, c in enumerate(chars))
# indices_char = dict((i, c) for i, c in enumerate(chars))

# def sample(preds, temperature=1.0):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.exp(np.log(preds) / temperature)  # softmax
#     preds = preds / np.sum(preds)                #
#     probas = np.random.multinomial(1, preds, 1)  # sample index
#     return np.argmax(probas)    

# # GARBAGE
# response_text = option + ": "
# #for i in range(response_length):
# next_char = ""
# i = 0
# with st.spinner('Wait for it...'):
#     while (next_char != '.') and (i < 4):
#             x_pred = np.zeros((1, seqlen, len(chars)))
#             for t, char in enumerate(seed):
#                 x_pred[0, t, char_indices[char]] = 1.

#             preds = model.predict(x_pred, verbose=0)
#             next_index = sample(preds[0, -1], diversity)
#             next_char = indices_char[next_index]

#             seed = seed[1:] + next_char

#             response_text = response_text + next_char
#             i = i + 1
#             print("i:", i)


# st.write(response_text)


def generate_text(philosopher_name, seed):

    selected_data = data_raw.loc[data_raw['author'] == philosopher_name]
    text = ' '.join(selected_data['sentence_str'].tolist())

    model_loc = "./models/" + philosopher_name
    model = keras.models.load_model(model_loc)

    while len(seed) < 40:
        seed = seed + " "

    seqlen = 40
    seed = seed[:seqlen]

    diversity = 0.5
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.exp(np.log(preds) / temperature)  # softmax
        preds = preds / np.sum(preds)                #
        probas = np.random.multinomial(1, preds, 1)  # sample index
        return np.argmax(probas)    

    response_text = philosopher_name + ": "
    next_char = ""
    i = 0
    with st.spinner('Wait for it...'):
        while (next_char != '.') and (i < 250):
                x_pred = np.zeros((1, seqlen, len(chars)))
                for t, char in enumerate(seed):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)
                next_index = sample(preds[0, -1], diversity)
                next_char = indices_char[next_index]

                seed = seed[1:] + next_char

                response_text = response_text + next_char
                i = i + 1

    st.write(response_text)
    return response_text

st.header("Conversation")

option_2 = st.slider('How many dialogs should we generate?', 1, 10, 1)
st.write('Values:', option_2)

option_3 = st.selectbox('Select Philosopher 1', philosophers)
option_4 = st.selectbox('Select Philosopher 2', philosophers)

conversation_seed = st.text_input("Prompt", "Magic mirror on the wall, who's the fairest of them all?")

if st.button('Generate!'):
    for i in range(option_2):
        conversation_seed = generate_text(option_3, conversation_seed)
        conversation_seed = generate_text(option_4, conversation_seed)
