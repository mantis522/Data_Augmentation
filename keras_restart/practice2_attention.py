from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional
from tensorflow.keras import Model
import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Attention(Layer):
    def __init__(self, bias=True):
        super(Attention, self).__init__()
        self.bias = bias
        self.init = initializers.get('glorot_uniform')

    def build(self, input_shape):
        self.ouput_dim = input_shape[-1]
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[2], 1),
                                 initializer=self.init,
                                 trainable=True)
        if self.bias:
            self.b = self.add_weight(
                name='{}_b'.format(self.name),
                shape=(input_shape[1], 1),
                initializer='zero',
                trainable=True)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        score = tf.matmul(inputs, self.W)
        if self.bias:
            score += self.b

        score = tf.tanh(score)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = inputs * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def get_config(self):
        return {'units': self.output_dim}

class TextBiRNNAttention(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 last_activation='softmax',
                 dense_size=None):

        super(TextBiRNNAttention, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.dense_size = dense_size
        self.embedding = Embedding(input_dim=self.max_features,
                                   output_dim=self.embedding_dims,
                                   input_length=self.maxlen)
        self.bi_rnn = Bidirectional(layer=GRU(
            units=128, activation='tanh', return_sequences=True
        ), merge_mode='concat')
        self.attention = Attention()
        self.classifier = Dense(self.class_num,
                                activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNNAtt must be 2, but now is {}'.format(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError(
                'The maxlen of inputs of TextBiRNNAtt must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        x = self.bi_rnn(emb)
        x = self.attention(x)
        if self.dense_size is not None:
            x = self.ffn(x)
        output = self.classifier(x)
        return output

def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)

class ModelHepler:
    def __init__(self, class_num, maxlen, max_features,
                 embedding_dims, epochs, batch_size):
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
        model = TextBiRNNAttention(maxlen=self.maxlen,
                         max_features=self.max_features,
                         embedding_dims=self.embedding_dims,
                         class_num=self.class_num,
                         last_activation='softmax')
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )

        # model.summary()
        self.model = model

    def get_callback(self, use_early_stop=True,
                     tensorboard_log_dir='logs\\FastText-epoch-5',
                     checkpoint_path="save_model_dir\\cp-moel.ckpt"):
        callback_list = []
        if use_early_stop:
            # EarlyStopping
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
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
