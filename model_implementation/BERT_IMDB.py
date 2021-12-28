import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

# 코드는 문제가 없는데 문장의 길이가 길면 OOM이 걸림.
# 그렇다고 배치사이즈를 너무 줄이면 학습하는데 시간도 오래걸리고 성능도 쓰레기됨
# 배치사이즈 8, 시퀀스길이 308 쯤에는 문제 없이 돌아감. 그런데 시간이 너무 오래걸림


tf.random.set_seed(1234)
np.random.seed(1234)

file_path = r"D:\ruin\data\IMDB Dataset2.csv"

imdb_csv = file_path
df_imdb = pd.read_csv(imdb_csv)
df_imdb = df_imdb.drop(['Unnamed: 0'], axis=1)

text_encoding = df_imdb['text']
t = Tokenizer()
t.fit_on_texts(text_encoding)
vocab_size = len(t.word_index) + 1
sequences = t.texts_to_sequences(text_encoding)

def max_text():
    for i in range(1, len(sequences)):
        max_length = len(sequences[0])
        if len(sequences[i]) > max_length:
            max_length = len(sequences[i])
    return max_length

text_num = max_text()
maxlen = text_num


BATCH_SIZE = 16
NUM_EPOCHS = 3
VALID_SPLIT = 0.2
MAX_LEN = 100 # EDA에서 추출된 Max Length
DATA_IN_PATH = 'data_in/KOR'
DATA_OUT_PATH = "../test_code/data_out/KOR"


tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir='../test_code/bert_ckpt', do_lower_case=False)

train_df, test_df = train_test_split(df_imdb, test_size=0.2, random_state=0)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

train_df = train_df[:1000]

def bert_tokenizer(sent, MAX_LEN):
    encoded_dict = tokenizer.encode_plus(
        text=sent,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=MAX_LEN,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True  # Construct attn. masks.

    )

    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict[
        'attention_mask']  # And its attention mask (simply differentiates padding from non-padding).
    token_type_id = encoded_dict['token_type_ids']  # differentiate two sentences

    return input_id, attention_mask, token_type_id


input_ids = []
attention_masks = []
token_type_ids = []
train_data_labels = []


for train_sent, train_label in tqdm(zip(train_df["text"], train_df["label"]), total=len(train_df)):
    try:
        input_id, attention_mask, token_type_id = bert_tokenizer(train_sent, MAX_LEN)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        train_data_labels.append(train_label)

    except Exception as e:
        print(e)
        print(train_sent)
        pass

train_movie_input_ids = np.array(input_ids, dtype=int)
train_movie_attention_masks = np.array(attention_masks, dtype=int)
train_movie_type_ids = np.array(token_type_ids, dtype=int)
train_movie_inputs = (train_movie_input_ids, train_movie_attention_masks, train_movie_type_ids)

train_data_labels = np.asarray(train_data_labels, dtype=np.int32)  # 레이블 토크나이징 리스트

print("# sents: {}, # labels: {}".format(len(train_movie_input_ids), len(train_data_labels)))

input_id = train_movie_input_ids[1]
attention_mask = train_movie_attention_masks[1]
token_type_id = train_movie_type_ids[1]

print(input_id)
print(attention_mask)
print(token_type_id)
print(tokenizer.decode(input_id))


class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(TFBertClassifier, self).__init__()

        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_class,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                    self.bert.config.initializer_range),
                                                name="classifier")

    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        # outputs 값: # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        return logits

cls_model = TFBertClassifier(model_name='bert-base-multilingual-cased',
                             dir_path='../test_code/bert_ckpt',
                             num_class=2)

optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
cls_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model_name = "tf2_bert_imdb"

# overfitting을 막기 위한 ealrystop 추가
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)
# min_delta: the threshold that triggers the termination (acc should at least improve 0.0001)
# patience: no improvment epochs (patience = 1, 1번 이상 상승이 없으면 종료)\

checkpoint_path = os.path.join(DATA_OUT_PATH, model_name, 'weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create path if exists
if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))

cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

# 학습과 eval 시작
history = cls_model.fit(train_movie_inputs, train_data_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])

# steps_for_epoch

print(history.history)


input_ids = []
attention_masks = []
token_type_ids = []
test_data_labels = []

for test_sent, test_label in tqdm(zip(test_df["text"], test_df["label"])):
    try:
        input_id, attention_mask, token_type_id = bert_tokenizer(test_sent, MAX_LEN)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        test_data_labels.append(test_label)
    except Exception as e:
        print(e)
        print(test_sent)
        pass

test_movie_input_ids = np.array(input_ids, dtype=int)
test_movie_attention_masks = np.array(attention_masks, dtype=int)
test_movie_type_ids = np.array(token_type_ids, dtype=int)
test_movie_inputs = (test_movie_input_ids, test_movie_attention_masks, test_movie_type_ids)

test_data_labels = np.asarray(test_data_labels, dtype=np.int32) #레이블 토크나이징 리스트

print("num sents, labels {}, {}".format(len(test_movie_input_ids), len(test_data_labels)))

results = cls_model.evaluate(test_movie_inputs, test_data_labels, batch_size=BATCH_SIZE)
print("test loss, test acc: ", results)