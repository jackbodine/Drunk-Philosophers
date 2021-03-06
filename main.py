import pandas as pd
import streamlit as st
import tensorflow
import numpy as np
from tensorflow import keras
import os
import openai
import random

# Streamlit Page Config
st.set_page_config(
     page_title="Drunk Philosophers",
     page_icon="🍺",
 )

st.title("Drunk Philosophers")
st.image("./SchoolOfAthens.jpeg", caption="Sober Philosophers Gathered in Athens")

developer = st.checkbox('Developer Mode')   # dev mode toggle

# Data prep
data_raw = pd.read_csv("data/philosophy_data.csv")
data_raw = data_raw.drop(
    ['sentence_spacy', 'original_publication_date', 'corpus_edition_date', 'sentence_length', 'sentence_lowered', 'lemmatized_str'], 
    axis=1
)
philosophers = list(dict.fromkeys(data_raw['author'].tolist()))

# Just some fun
loading_messages = ["Reviving the dead...", "Cloning Evan Trabitz...", "Disappointing Ulrich...", "Sequencing Genomes...", 
    "Gaslighting our way into the group...", "Travelling to Denmark...", "Fighting the queen...", "Crashing the bus...", 
    "Not bringing the wand to London...", "Making 9 BILLION QUID...", "Seeing the whites of their eyes...", "Visiting Silicon Roundabout...", 
    "Writing the blog post...", "Running away from tibetan monks...", "Squeening Bjorn...", "Turning on supercomputers...", 
    "Sending Grace to Switzerland...", "Using the bathroom at Tivoli...", "Being mad bro...", "Rickrolling Evan...", 
    "Pretending Ulrich has a wife...", "Getting out of here...", "Jeg bor i Lyngby...", "Bringing Tuborg to class...", 
    "5 Tuborgs in...", "Approaching boombox guy...", "Not buying overpriced coffee at Brickaccino...", 
    "Bjorn breaking his leg in Lego House...", "Evan undergoing meiosis...", "Crimping quid...", "Searching for mini-keg...", 
    "Trying to understand spoken British...", "Red Bull Red Bull Red Bull...", "MARCH 24TH WATER BOTTLE INCIDENT", 
    "Writing fanfictions about Evan...", "Not scheduling field studies...", "Getting drunk before 8am class...", 
    "Getting ourselves to Roskilde...", "Studying WiNd EnErGy...", "Volunteer firefighting...", "Bricking up...", 
    "Falling down stairs at an FCK game...", "Breaking the laws of robotics...", "Falling in love with Pepper..."]

# Constants
if developer:
    seqlen = st.number_input('seqlen', value=100)
    lstm_diversity = st.number_input('lstm_diversity', value=0.2)
    lstm_max_length = st.number_input('lstm_max_length', value=250)
    gpt3_temperature = st.number_input('gpt3_temperature', value=0.1)
else:
    seqlen = 100
    lstm_diversity = 0.2
    lstm_max_length = 250
    gpt3_temperature = 0.1

# Generate Text Function
def generate_text(philosopher_name, seed, debug):

    selected_data = data_raw.loc[data_raw['author'] == philosopher_name]
    text = ' '.join(selected_data['sentence_str'].tolist())

    model_loc = "./models6/" + philosopher_name + "v5"
    model = keras.models.load_model(model_loc)

    if debug:
        st.header(philosopher_name)
        st.write("Prompt: ", seed)

    while len(seed) < seqlen:
        pretext = ""
        for i in range(seqlen-len(seed)):
            pretext = pretext + " "
        seed = pretext + seed

    seed = seed[len(seed)-seqlen:]
    print("Seed is: ", seed)

    if debug:
        st.write("Seed: ", seed)

    diversity = lstm_diversity
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

    message = random.choice(loading_messages)
    with st.spinner(message):
        while (i < lstm_max_length):
                x_pred = np.zeros((1, seqlen, len(chars)))
                for t, char in enumerate(seed):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)
                next_index = sample(preds[0, -1], diversity)
                next_char = indices_char[next_index]

                seed = seed[1:] + next_char

                response_text = response_text + next_char
                i = i + 1

        if debug:
            st.write("LSTM: ", response_text)

        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Calls GPT3 Completion model on LSTM output.
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=response_text + "\n\nTl;dr",
            temperature=gpt3_temperature,
            max_tokens=80,
            top_p=1,
            frequency_penalty=0.5,
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

# Function to find the first alphebetical character in a string and return its location.
def findFirstLetter(string):
    i = 0
    letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"] # Evan did this
    for char in string:
        if char in letters:
            return i

        i = i + 1

    return 0

option_2 = st.slider('How many dialogs should we generate?', 1, 5, 1)
option_3 = st.selectbox('Select Philosopher 1', philosophers)
option_4 = st.selectbox('Select Philosopher 2', philosophers)

conversation_seed = st.text_input("Prompt", "The soul is")

if st.button('Generate!'):

    for i in range(option_2):
        conversation_seed = generate_text(option_3, conversation_seed, developer)
        conversation_seed = generate_text(option_4, conversation_seed, developer)

    st.balloons()
    st.header("Tusinde Tak")