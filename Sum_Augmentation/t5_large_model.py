from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import re
import torch
#
# torch.cuda.is_available()
import time

start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-large")
model = model.to(device)
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-large")

imdb_data = r"D:\ruin\data\IMDB Dataset2.csv"
df_imdb = pd.read_csv(imdb_data)

original_text = df_imdb['text'][28]

# encode the text into tensor of integers using the appropriate tokenizer
inputs = tokenizer.encode(original_text, return_tensors="pt", max_length=1024, truncation=True)
inputs = inputs.to(device)


# generate the summarization output
outputs = model.generate(
    inputs,
    max_length=300,
    min_length=40,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)
# just for debugging

def cleanText(readData):
    text = re.sub("<[^>]*>", '', readData)

    return text


print("original_text : ", original_text)

summarized_text = tokenizer.decode(outputs[0])

print("output_text : ", cleanText(summarized_text))

print("time :", time.time() - start)