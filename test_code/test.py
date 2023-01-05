import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

loaded_model = load_model("CNN.h5")
file_path = r"D:\ruin\data\imdb_summarization\t5_large_with_huggingface_sentiment.csv"
imdb_csv = file_path
df_imdb = pd.read_csv(imdb_csv)

original_data = df_imdb
text_encoding = original_data['original_text']
t = Tokenizer()
t.fit_on_texts(text_encoding)
sequences = t.texts_to_sequences(text_encoding)
word_to_index = t.word_index

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key

print('빈도수 상위 1등 단어 : {}'.format(index_to_word[4]))

# def sentiment_predict(new_sentence):
#   # 알파벳과 숫자를 제외하고 모두 제거 및 알파벳 소문자화
#   new_sentence = re.sub('[^0-9a-zA-Z ]', '', new_sentence).lower()
#   encoded = []
#
#   # 띄어쓰기 단위 토큰화 후 정수 인코딩
#   for word in new_sentence.split():
#     try :
#       # 단어 집합의 크기를 10,000으로 제한.
#       if word_to_index[word] <= 10000:
#         encoded.append(word_to_index[word]+3)
#       else:
#       # 10,000 이상의 숫자는 <unk> 토큰으로 변환.
#         encoded.append(2)
#     # 단어 집합에 없는 단어는 <unk> 토큰으로 변환.
#     except KeyError:
#       encoded.append(2)
#
#   pad_sequence = pad_sequences([encoded], maxlen=max_len)
#   score = float(loaded_model.predict(pad_sequence)) # 예측