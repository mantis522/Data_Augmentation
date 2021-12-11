import tensorflow as tf
import  numpy as np
import random as rn
import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

np.random.seed(0)
rn.seed(0)
tf.random.set_seed(0)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


def scaled_dot_product_attention(q, k, v, mask=None):

    # (batch_size, num_heads, seq_len_q, d ) dot (batch_size, num_heads, d, seq_ken_k) = (batch_size, num_heads,, seq_len_q, seq_len)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)

    if mask is not None:
        # (batch_size, num_heads,, seq_len_q, seq_len) + (batch_size, 1,, 1, seq_len)
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # (batch_size, num_heads, seq_len) dot (batch_size, num_heads, d_v) = (batch_size, num_heads, seq_len_q, d_v)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert (d_model > num_heads) and (d_model % num_heads == 0)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.qw = tf.keras.layers.Dense(d_model)
        self.kw = tf.keras.layers.Dense(d_model)
        self.vw = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

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
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=(0, 2, 1, 3)) # (batch_size, seq_len_q, num_heads, depth_v)
        concat_attention = tf.reshape(scaled_attention, shape=(batch_size, -1, self.d_model)) # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layer_norm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x+attn_output) # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1+ffn_output) # (batch_size, input_seq_len, d_model)
        return out2



class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.enc_layer = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # x.shape == (batch_size, seq_len)
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layer[i](x, training, mask)
        return  x #(batch_size, input_seq_len, d_model)

class TextTransformerEncoder(tf.keras.Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 num_layers,
                 num_heads,
                 dff,
                 pe_input,
                 last_activation = 'softmax',
                 rate=0.1):
        super(TextTransformerEncoder, self).__init__()
        self.maxlen = maxlen
        self.embedding_dims = embedding_dims
        self.encoder = Encoder(num_layers, embedding_dims, num_heads, dff, max_features, pe_input, rate)
        # self.flat = tf.keras.layers.Flatten()
        self.final_layer = tf.keras.layers.Dense(class_num, activation=last_activation)

    def call(self, inp, training=False, enc_padding_mask=None):
        # mask == (batch_size, 1, 1, seq_len) （batch_size, num_heads, seq_len, seq_len)
        # (batch_size, inp_seq_len, d_model)
        print('========= : ', training)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        # enc_output = tf.reshape(enc_output, (-1, self.maxlen*self.embedding_dims))
        # enc_output = self.flat(enc_output)
        enc_output = tf.reduce_mean(enc_output, axis=1)
        final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output

    def build_graph(self, input_shapes):
        input_shape, _ = input_shapes
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)



if __name__=='__main__':
    class_num = 2
    maxlen = 400
    embedding_dims = 400
    epochs = 10
    batch_size = 128
    max_features = 5000

    num_layers = 2
    num_heads = 8
    dff = 2048
    pe_input = 10000
    model = TextTransformerEncoder(
                        maxlen=maxlen,
                        max_features=max_features,
                        embedding_dims=embedding_dims,
                        class_num=class_num,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        dff=dff,
                        pe_input=pe_input
                        )
    # model.build_graph(input_shapes=(None, maxlen))
    # model.summary()
    temp_input = tf.random.uniform((64, maxlen))

    fn_out= model(temp_input, training=False, enc_padding_mask=None)
    # fn_out = model([temp_input, None], training=False)

    print(fn_out.shape)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)


class ModelHepler:
    def __init__(self, class_num, maxlen, max_features, embedding_dims, epochs, batch_size, num_layers,
                 num_heads,
                 dff,
                 pe_input,):
        self.class_num = class_num
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.pe_input = pe_input

    def create_model(self):
        input_seq = tf.keras.layers.Input((self.maxlen, ), dtype=tf.int64, name='inputs_seq')
        input_mask = tf.keras.layers.Input((1,1,self.maxlen), dtype=tf.float32, name='inputs_mask')
        internal_model = TextTransformerEncoder(maxlen=self.maxlen,
                         max_features=self.max_features,
                         embedding_dims=self.embedding_dims,
                         class_num=self.class_num,
                                       num_layers=self.num_layers,
                                       num_heads=self.num_heads,
                                       dff=self.dff,
                                       pe_input=self.pe_input,
                         last_activation='softmax',
                          # dense_size=[128]
                          )
        logits = internal_model(input_seq, enc_padding_mask=input_mask)
        logits = tf.keras.layers.Lambda(lambda x: x, name="logits",
                                        dtype=tf.float32)(logits)
        model = tf.keras.Model([input_seq, input_mask], logits)
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        # model.build_graph(input_shape=(None, self.maxlen))
        model.summary()
        self.model =  model

    def get_callback(self, use_early_stop=True, tensorboard_log_dir='logs\\FastText-epoch-5', checkpoint_path="save_model_dir\\cp-moel.ckpt"):
        callback_list = []
        if use_early_stop:
            # EarlyStopping
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
            callback_list.append(early_stopping)
        if checkpoint_path is not None:
            # save model
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                             monitor='val_accuracy',
                                             mode='max',
                                             save_best_only=True,
                                             save_weights_only=True,
                                             verbose=1,
                                             period=1,
                                             )
            callback_list.append(cp_callback)
        if tensorboard_log_dir is not None:
            # tensorboard --logdir logs/FastText-epoch-5
            checkout_dir(tensorboard_log_dir, do_delete=True)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
            callback_list.append(tensorboard_callback)
        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val, x_train_mask, x_val_mask):
        print('Bulid Model...')
        self.create_model()
        print('Train...')
        self.model.fit([x_train, x_train_mask], y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  callbacks=self.callback_list,
                  validation_data=((x_val, x_val_mask), y_val))

    def load_model(self, checkpoint_path):
        self.create_model()
        checkpoint_dir = os.path.dirname((checkpoint_path))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('restore model name is : ', latest)

        self.model.load_weights(latest)

# ================  params =========================
class_num = 2
maxlen = 400
embedding_dims = 200
epochs = 20
batch_size = 128
max_features = 5000

num_layers = 2
num_heads = 4
dff = 128
pe_input = max_features

MODEL_NAME = 'TextTransformerEncoder-epoch-20-emb-200'

use_early_stop=True
tensorboard_log_dir = 'logs\\{}'.format(MODEL_NAME)
# checkpoint_path = "save_model_dir\\{}\\cp-{epoch:04d}.ckpt".format(MODEL_NAME, '')
checkpoint_path = 'save_model_dir\\'+MODEL_NAME+'\\cp-{epoch:04d}.ckpt'
#  ====================================================================

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print('Pad sequences (samples x time)...')
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

x_train_mask = create_padding_mask(x_train)
x_test_mask = create_padding_mask(x_test)

print('x_train_mask shape:', x_train_mask.shape)

#
model_hepler = ModelHepler(class_num=class_num,
                           maxlen=maxlen,
                           max_features=max_features,
                           embedding_dims=embedding_dims,
                           epochs=epochs,
                           batch_size=batch_size,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        dff=dff,
                        pe_input=pe_input
                           )
model_hepler.get_callback(use_early_stop=use_early_stop, tensorboard_log_dir=tensorboard_log_dir, checkpoint_path=checkpoint_path)
model_hepler.fit(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test, x_train_mask=x_train_mask, x_val_mask=x_test_mask)


print('Test predict...')
result = model_hepler.model.predict((x_test, x_test_mask))
print('Test evaluate...')
test_score = model_hepler.model.evaluate((x_test, x_test_mask), y_test,
                            batch_size=batch_size)
print("test loss:", test_score[0], "test accuracy", test_score[1])


print('Restored Model...')
model_hepler = ModelHepler(class_num=class_num,
                           maxlen=maxlen,
                           max_features=max_features,
                           embedding_dims=embedding_dims,
                           epochs=epochs,
                           batch_size=batch_size,
                           num_layers=num_layers,
                           num_heads=num_heads,
                           dff=dff,
                           pe_input=pe_input
                           )
model_hepler.load_model(checkpoint_path=checkpoint_path)
# 0.8790
loss, acc = model_hepler.model.evaluate((x_test, x_test_mask), y_test, verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))