from transformers import pipeline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summarizer = pipeline("summarization")
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")

text = """Rather than scratches and insect droppings, this one has random pixelations combined with muddy light and vague image resolution. Probably the cue should have been the packaging is straight out of your street corner bootleg dealer.If you've ever seen a reasonably good condition film copy, you know the defining visuals of his film are the crystal clear lighting contrasts in black and white. The surrounding countryside and 'old home' scenes are set with early morning ground mists or the haze of memory while the events on the bridge and in the water are bright, clear, and immediate.Here everything is dull, dark, and clouded. Or, if you remember the timbre and enunciation of Captain's commands, so are the visuals.After that, it is hard to believe this award winning, critically acclaimed film's best presentation is on YOUTUBE. Somewhere "out there" is a DVD that comes up to the standard of a 16mm public library reel.Just none of them appear to be on Amazon."""

summary_text = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text)
