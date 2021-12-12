import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional
from tensorflow.keras import Model
import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert (d_model > num_heads) and\
               (d_model % num_heads == 0), '{}, {}'.format(d_model, num_heads)

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.qw = tf.keras.layers.Dense(d_model)
        self.kw = tf.keras.layers.Dense(d_model)
        self.vw = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

        self.scaled_dot_product_attention = scaled_dot_product_attention

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def call(self, v, k, q, mask=None):
        # v = inputs
        batch_size = tf.shape(q)[0]

        q = self.qw(q)  # (batch_size, seq_len_q, d_model)
        k = self.kw(k)  # (batch_size, seq_len, d_model)
        v = self.vw(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len, depth_v)

        # scaled_attention, (batch_size, num_heads, seq_len_q, depth_v)
        # attention_weights, (batch_size, num_heads, seq_len_q, seq_len)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=(0, 2, 1, 3)) # (batch_size, seq_len_q, num_heads, depth_v)
        concat_attention = tf.reshape(scaled_attention, shape=(batch_size, -1, self.d_model)) # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output

def point_wise_feed_forward_network(dense_size):
    ffn = tf.keras.Sequential()
    for size in dense_size:
        ffn.add(Dense(size, activation='relu'))
    return ffn

class TextSelfAtt(Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 last_activation='softmax',
                 dense_size=None
                 ):
        super(TextSelfAtt, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.dense_size = dense_size

        self.embedding = Embedding(input_dim=self.max_features,
                                   output_dim=self.embedding_dims,
                                   input_length=self.maxlen)
        self.attention = MultiHeadAttention(d_model=embedding_dims,
                                            num_heads=4)
        self.bi_rnn = Bidirectional(layer=GRU(units=400,
                                              activation='tanh',
                                              return_sequences=True),
                                    merge_mode='ave' )

        if self.dense_size is not None:
            self.ffn = point_wise_feed_forward_network(dense_size)
        self.classifier = Dense(self.class_num,
                                activation=self.last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNNAtt must be 2, but now is {}'.format(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextBiRNNAtt must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        x = self.embedding(inputs)
        x = self.attention(x, x, x)
        x = self.bi_rnn(x)
        # x = tf.reduce_mean(x, axis=1)
        x = tf.reshape(x,  shape=(-1, 400*self.maxlen))
        if self.dense_size is not None:
            x = self.ffn(x)
        output = self.classifier(x)
        return output

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)

def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)

class ModelHelper:
    def __init__(self, class_num, maxlen, max_features, embedding_dims, epochs, batch_size):
        self.class_num = class_num
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []
        print('Bulid Model...')
        self.create_model()

    def create_model(self):
        model = TextSelfAtt(maxlen=self.maxlen,
                         max_features=self.max_features,
                         embedding_dims=self.embedding_dims,
                         class_num=self.class_num,
                         last_activation='softmax',
                          # dense_size=[128]
                          )
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        model.build_graph(input_shape=(None, self.maxlen))
        model.summary()
        self.model =  model

    def get_callback(self, use_early_stop=True,
                     tensorboard_log_dir='logs\\FastText-epoch-5',
                     checkpoint_path="save_model_dir\\cp-moel.ckpt"):
        callback_list = []
        if use_early_stop:
            # EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss',
                                           patience=2,
                                           mode='min')
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
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  callbacks=self.callback_list,
                  validation_data=(x_val, y_val))

    def load_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname((checkpoint_path))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('restore model name is : ', latest)
        self.model.load_weights(latest)

