import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

MAX_SEQ_LEN = 305

def get_masks(tokens):
    """Masks: 1 for real tokens and 0 for paddings"""
    return [1] * len(tokens) + [0] * (MAX_SEQ_LEN - len(tokens))


def get_segments(tokens):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (MAX_SEQ_LEN - len(tokens))

def get_ids(ids):
    """Token ids from Tokenizer vocab"""
    token_ids = ids
    input_ids = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))
    return input_ids

def create_single_input(sentence, tokenizer):
    """Create an input from a sentence"""
    encoded = tokenizer.encode(sentence)

    ids = get_ids(encoded.ids)
    masks = get_masks(encoded.tokens)
    segments = get_segments(encoded.tokens)

    return ids, masks, segments

def convert_sentences_to_features(sentences, tokenizer):
    """Convert sentences to features: input_ids, input_masks and input_segments"""
    input_ids, input_masks, input_segments = [], [], []

    for sentence in tqdm(sentences, position=0, leave=True):
        ids, masks, segments = create_single_input(sentence, tokenizer)
        assert len(ids) == MAX_SEQ_LEN
        assert len(masks) == MAX_SEQ_LEN
        assert len(segments) == MAX_SEQ_LEN
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]

def nlp_model(callable_object):
    # Load the pre-trained BERT base model
    bert_layer = hub.KerasLayer(handle=callable_object, trainable=True)

    # BERT layer three inputs: ids, masks and segments
    input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")
    input_masks = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_masks")
    input_segments = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids")

    inputs = [input_ids, input_masks, input_segments]  # BERT inputs
    pooled_output, sequence_output = bert_layer(inputs)  # BERT outputs

    # Add a hidden layer
    x = Dense(units=768, activation="relu")(pooled_output)
    x = Dropout(0.1)(x)

    # Add output layer
    outputs = Dense(2, activation="softmax")(x)

    # Construct a new model
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = nlp_model("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2")
model.summary()


if __name__ == '__main__':
    file_path = r"D:\ruin\data\IMDB Dataset2.csv"

    imdb_df = pd.read_csv(file_path)
    df_imdb = imdb_df.drop(['Unnamed: 0'], axis=1)

    train_df, test_df = train_test_split(df_imdb, test_size=0.2, random_state=0)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

    TAG_RE = re.compile(r"<[^>]+>")

    def remove_tags(text):
        return TAG_RE.sub("", text)

    def preprocess_text(sen):
        # Removing html tags
        sentence = remove_tags(sen)
        # Remove punctuations and numbers
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)
        # Removing multiple spaces
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence

    def preprocessing_data(dataframe):
        reviews = []
        sentences = list(dataframe['text'])
        for sen in sentences:
            reviews.append(preprocess_text(sen))

        reviews = convert_sentences_to_features(reviews, tokenizer)

        label = dataframe['label']
        label = np.array(list(label))
        label = to_categorical(label)

        return reviews, label


    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    save_path = "../test_code/bert_base_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

    tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
    tokenizer.enable_truncation(MAX_SEQ_LEN-2)

    x_train, y_train = preprocessing_data(train_df)
    x_test, y_test = preprocessing_data(test_df)
    x_val, y_val = preprocessing_data(val_df)

    BATCH_SIZE = 8
    EPOCHS = 3

    opt = Adam(learning_rate=2e-5)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=1)

    # Save the trained model
    # model.save('nlp_model.h5')

    pred_test = np.argmax(model.predict(x_test), axis=1)
    print(pred_test[:10])