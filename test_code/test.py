import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as hub
import pandas as pd


def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

    preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1',
                                         name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)

    encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2', trainable=True,
                             name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(1, activation='softmax', name='classifier')(net)
    return tf.keras.Model(text_input, net)


def remove_and_split(s):
    s = s.replace('[', '')
    s = s.replace(']', '')
    return s.split(',')


def df_to_dataset(dataframe, shuffle=True, batch_size=2):
    dataframe = dataframe.copy()
    labels = tf.squeeze(tf.constant([dataframe.pop('labels')]), axis=0)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)).batch(
        batch_size)
    return ds


dummy_data = {'text': [
    "Improve the physical fitness of your goldfish by getting him a bicycle",
    "You are unsure whether or not to trust him but very thankful that you wore a turtle neck",
    "Not all people who wander are lost",
    "There is a reason that roses have thorns",
    "Charles ate the french fries knowing they would be his last meal",
    "He hated that he loved what she hated about hate",
], 'labels': ['1', '0', '1', '1', '0',
              '1']}

df = pd.DataFrame(dummy_data)
df["labels"] = df["labels"].apply(lambda x: [int(i) for i in remove_and_split(x)])
batch_size = 2

print(df)

train_ds = df_to_dataset(df, batch_size=batch_size)
val_ds = df_to_dataset(df, batch_size=batch_size)
test_ds = df_to_dataset(df, batch_size=batch_size)

loss = 'categorical_crossentropy'
metrics = ["accuracy"]

classifier_model = build_classifier_model()
classifier_model.compile(optimizer='adam',
                         loss=loss,
                         metrics=metrics)

history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=5)