import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import datetime
import csv

def nlp_model(callable_object):
    # Load the pre-trained BERT base model
    bert_layer = hub.KerasLayer(handle=callable_object, trainable=True)

    # BERT layer three inputs: ids, masks and segments
    input_ids = Input(shape=(maxlen,), dtype=tf.int32, name="input_ids")
    input_masks = Input(shape=(maxlen,), dtype=tf.int32, name="input_masks")
    input_segments = Input(shape=(maxlen,), dtype=tf.int32, name="segment_ids")

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


def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)

class ModelHelper:
    def __init__(self, class_num, maxlen, epochs, batch_size):
        self.class_num = class_num
        self.maxlen = maxlen
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []
        print('Build Model...')
        self.create_model()

    def create_model(self):
        model = nlp_model("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2")

        model.compile(optimizer=Adam(learning_rate=2e-5),
                      loss="categorical_crossentropy",
                      metrics=['acc',
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.Precision(name='precision'),
                     tfa.metrics.F1Score(name='F1_micro',
                                         num_classes=self.class_num,
                                         average='micro'),
                     tfa.metrics.F1Score(name='F1_macro',
                                         num_classes=self.class_num,
                                         average='macro')])

        self.model = model

    def get_callback(self, use_early_stop=True,
                     tensorboard_log_dir='logs\\FastText-epoch-5',
                     checkpoint_path="save_model_dir\\cp-moel.ckpt"):
        callback_list = []
        if use_early_stop:
            # EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
            callback_list.append(early_stopping)
        if checkpoint_path is not None:
            # save model
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          monitor='val_loss',
                                          mode='min',
                                          save_best_only=True,
                                          save_weights_only=True,
                                          verbose=1,
                                          period=2,
                                          )
            callback_list.append(cp_callback)
        if tensorboard_log_dir is not None:
            # tensorboard --logdir logs/FastText-epoch-5
            checkout_dir(tensorboard_log_dir, do_delete=True)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir,
                                               histogram_freq=1)
            callback_list.append(tensorboard_callback)
        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val):
        print('Train...')
        self.model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=1,
                       callbacks=self.callback_list)

    def load_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname((checkpoint_path))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('restore model name is : ', latest)
        self.model.load_weights(latest)


if __name__ == '__main__':
    file_path = r"D:\ruin\data\imdb_summarization\t5_large_with_huggingface_sentiment.csv"

    df_imdb = pd.read_csv(file_path)
    df_imdb = df_imdb.drop(['Unnamed: 0'], axis=1)

    start = 0
    end = 500

    original_data = df_imdb[start:end]
    # original_data = df_imdb

    train_df, test_df = train_test_split(original_data, test_size=0.2, random_state=0)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    def A2D_train(data_df):
        text_list = []
        label_list = []
        for i in range(len(data_df)):
            original_label = int(data_df['original_label'][i])
            huggingface_label = int(data_df['huggingface_sentiment'][i])
            if original_label == huggingface_label:
                text_list.append(data_df['summarized_text'][i])
                label_list.append(huggingface_label)

        return text_list, label_list

    sumtext_list, sumlabel_list = A2D_train(train_df)

    def concat_df(data_df, text_list, label_list):
        df = pd.DataFrame([x for x in zip(text_list, label_list)])
        df.columns = ['original_text', 'original_label']
        concating = pd.concat([data_df, df])
        concating = concating.reset_index(drop=True)

        return concating

    train_df = concat_df(train_df, sumtext_list, sumlabel_list)

    maxlen = 280
    class_num = 2
    epochs = 8
    batch_size = 8

    def get_masks(tokens):
        """Masks: 1 for real tokens and 0 for paddings"""
        return [1] * len(tokens) + [0] * (maxlen - len(tokens))


    def get_segments(tokens):
        """Segments: 0 for the first sequence, 1 for the second"""
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (maxlen - len(tokens))


    def get_ids(ids):
        """Token ids from Tokenizer vocab"""
        token_ids = ids
        input_ids = token_ids + [0] * (maxlen - len(token_ids))
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
            assert len(ids) == maxlen
            assert len(masks) == maxlen
            assert len(segments) == maxlen
            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)

        return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32),
                np.asarray(input_segments, dtype=np.int32)]

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

    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    save_path = "bert_base_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

    tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
    tokenizer.enable_truncation(maxlen-2)


    def preprocessing_data(dataframe):
        reviews = []
        sentences = list(dataframe['original_text'])
        for sen in sentences:
            reviews.append(preprocess_text(sen))

        reviews = convert_sentences_to_features(reviews, tokenizer)

        label = dataframe['original_label']
        label = np.array(list(label))
        label = to_categorical(label)

        return reviews, label

    x_train, y_train = preprocessing_data(train_df)
    x_test, y_test = preprocessing_data(test_df)
    x_val, y_val = preprocessing_data(val_df)

    print('X_train size:',len(y_train))
    print('X_test size:',len(y_test))
    print('X_val size:',len(y_val))

    use_early_stop = True
    MODEL_NAME = 'BERT-IMDB-Aug'

    tensorboard_log_dir = 'logs\\{}'.format(MODEL_NAME)
    checkpoint_path = 'save_model_dir\\' + MODEL_NAME + '\\cp-{epoch:04d}.ckpt'

    model_helper = ModelHelper(class_num=2,
                               maxlen=maxlen,
                               epochs=epochs,
                               batch_size=batch_size)

    model_helper.get_callback(use_early_stop=use_early_stop,
                              tensorboard_log_dir=tensorboard_log_dir,
                              checkpoint_path=checkpoint_path)

    model_helper.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    print('Restored Model...')
    model_helper = ModelHelper(class_num=class_num,
                               maxlen=maxlen,
                               epochs=epochs,
                               batch_size=batch_size)
    model_helper.load_model(checkpoint_path=checkpoint_path)

    loss, acc, recall, precision, F1_micro, F1_macro = model_helper.model.evaluate(x_test, y_test, verbose=1)


    def result_preprocessing(result):
        result = "{:5.2f}%".format(100 * result)
        return result


    loss = result_preprocessing(loss)
    acc = result_preprocessing(acc)
    recall = result_preprocessing(recall)
    precision = result_preprocessing(precision)
    F1_macro = result_preprocessing(F1_macro)
    F1_micro = result_preprocessing(F1_micro)

    print("Restored model, accuracy:", acc)
    print("Restored model, recall:", recall)
    print("Restored model, precision:", precision)
    print("Restored model, f1_micro:", F1_micro)
    print("Restored model, f1_macro:", F1_macro)

    now = datetime.datetime.now()
    csv_filename = r"D:\ruin\data\result\Aug_BERT_IMDB_t5_large.csv"
    result_list = [now, len(original_data), len(train_df), start, end, acc, loss,
                   recall, precision, F1_micro, F1_macro]

    if os.path.isfile(csv_filename):
        print("already csv file exist...")

    else:
        print("make new csv file...")
        column_list = ['date', 'number of full data',
                       'number of train data',
                       'start', 'end',
                       'acc', 'loss', 'recall',
                       'precision', 'F1_micro',
                       'F1_macro']
        df_making = pd.DataFrame(columns=column_list)
        df_making.to_csv(csv_filename, index=False)

    try:
        f = open(csv_filename, 'a', newline='')
        wr = csv.writer(f)
        wr.writerow(result_list)
        f.close()

    except PermissionError:
        print("지금 보고 있는 엑셀창을 닫아주세요.")