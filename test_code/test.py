from transformers import pipeline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summarizer = pipeline("summarization")
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")

text = 'that rara avis : the intelligent romantic comedy with actual ideas on its mind . '

summary_text = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text)