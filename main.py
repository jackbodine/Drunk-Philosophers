from urllib import response
import pandas as pd
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import openai

st.title("Drunk Philosophers")

st.image("./SchoolOfAthens.jpeg", caption="Sober Philosophers Gathered in Athens")
data_raw = pd.read_csv("data/philosophy_data.csv")
data_raw = data_raw.drop(['sentence_spacy', 'original_publication_date', 'corpus_edition_date', 'sentence_length', 'sentence_lowered', 'lemmatized_str'], axis=1)
philosophers = list(dict.fromkeys(data_raw['author'].tolist()))

option = st.selectbox('Select a Philosopher', philosophers)

selected_data = data_raw.loc[data_raw['author'] == option]
text = ' '.join(selected_data['sentence_str'].tolist())

st.dataframe(selected_data)

def generate_text(philosopher_name, seed, debug):

    selected_data = data_raw.loc[data_raw['author'] == philosopher_name]
    text = ' '.join(selected_data['sentence_str'].tolist())

    model_loc = "./models/" + philosopher_name
    model = keras.models.load_model(model_loc)

    if debug:
        st.header(philosopher_name)
        st.write("Prompt: ", seed)

    while len(seed) < 40:
        seed = seed + " "

    seqlen = 40
    #seed = seed[:seqlen]
    seed = seed[len(seed)-seqlen:]
    print("Seed is: ", seed)

    if debug:
        st.write("Seed: ", seed)

    diversity = 0.2
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.exp(np.log(preds) / temperature)  # softmax
        preds = preds / np.sum(preds)                #
        probas = np.random.multinomial(1, preds, 1)  # sample index
        return np.argmax(probas)    

    response_text = ""
    next_char = ""
    i = 0
    with st.spinner('Reviving the dead...'):
        while (next_char != '.'):
                x_pred = np.zeros((1, seqlen, len(chars)))
                for t, char in enumerate(seed):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)
                next_index = sample(preds[0, -1], diversity)
                next_char = indices_char[next_index]

                if (i > 250) and (next_char == ' '):
                    next_char = '.'

                seed = seed[1:] + next_char

                response_text = response_text + next_char
                i = i + 1

    if debug:
        st.write("LSTM: ", response_text)

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=response_text + "\n\nTl;dr",
        temperature=0.3,
        max_tokens=60,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

    response_text = response["choices"][0]["text"]
    firstLetter = findFirstLetter(response_text)
    response_text = response_text[firstLetter:]
    if debug:
        st.write("GPT3: ", response_text)
    else:
        st.write(philosopher_name, ": ", response_text)

    return response_text

def findFirstLetter(string):
    i = 0
    letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    for char in string:
        if char in letters:
            return i

        i = i + 1

    return 0

def summonSatan():
    st.write("")

st.header("Conversation")

option_2 = st.slider('How many dialogs should we generate?', 1, 10, 1)

option_3 = st.selectbox('Select Philosopher 1', philosophers)
option_4 = st.selectbox('Select Philosopher 2', philosophers)

conversation_seed = st.text_input("Prompt", "The soul is")

debug = st.checkbox('Debug Mode')

if st.button('Generate!'):
    for i in range(option_2):
        conversation_seed = generate_text(option_3, conversation_seed, debug)
        conversation_seed = generate_text(option_4, conversation_seed, debug)

    st.balloons()