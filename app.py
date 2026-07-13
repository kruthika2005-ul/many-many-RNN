import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATASET = "words_pos.csv"      # Rename your uploaded CSV to this
MODEL = "pos_rnn.keras"
TOKENIZER = "tokenizer.pkl"
LABEL_ENCODER = "label_encoder.pkl"

MAX_WORDS = 10000
MAX_LEN = 20

def train_model():

    df = pd.read_csv(DATASET)

    # Keep required columns
    df = df[["word", "pos_tag"]]

    df.dropna(inplace=True)

    words = df["word"].astype(str).str.lower()

    tags = df["pos_tag"].astype(str)

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        char_level=True,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(words)

    X = tokenizer.texts_to_sequences(words)

    X = pad_sequences(
        X,
        maxlen=MAX_LEN,
        padding="post"
    )

    encoder = LabelEncoder()

    y = encoder.fit_transform(tags)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=64,
            input_length=MAX_LEN
        )
    )

    model.add(
        SimpleRNN(64)
    )

    model.add(
        Dense(
            64,
            activation="relu"
        )
    )

    model.add(
        Dense(
            len(encoder.classes_),
            activation="softmax"
        )
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("Accuracy :", accuracy)

    model.save(MODEL)

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    with open(LABEL_ENCODER, "wb") as f:
        pickle.dump(encoder, f)

# Predict
def predict_word(word):

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    with open(LABEL_ENCODER, "rb") as f:
        encoder = pickle.load(f)

    seq = tokenizer.texts_to_sequences([word.lower()])

    seq = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post"
    )

    prediction = model.predict(seq, verbose=0)

    index = np.argmax(prediction)

    tag = encoder.inverse_transform([index])[0]

    confidence = prediction[0][index]

    return tag, confidence
# Train if model not available
if (
    not os.path.exists(MODEL)
    or not os.path.exists(TOKENIZER)
    or not os.path.exists(LABEL_ENCODER)
):
    train_model()


# Streamlit UI
st.set_page_config(page_title="POS Tag Prediction")

st.title("Part-of-Speech Tag Prediction using Simple RNN")

word = st.text_input(
    "Enter a Word",
    "running"
)

if st.button("Predict"):

    if word.strip() == "":
        st.warning("Please enter a word.")

    else:

        tag, confidence = predict_word(word)

        st.success(f"Predicted POS Tag : {tag}")

        st.write(
            f"Confidence : {confidence*100:.2f}%"
        )