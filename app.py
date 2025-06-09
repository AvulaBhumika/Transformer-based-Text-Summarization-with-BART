
import re
import nltk
import gradio as gr
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_chunks(text, max_tokens=1024):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

class Summarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize_text(self, text, max_length=150, min_length=40):
        text = clean_text(text)
        chunks = split_into_chunks(text)
        summaries = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors='pt', max_length=1024, truncation=True)
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        return " ".join(summaries)

def evaluate_summary(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)

summarizer = Summarizer()

def summarize_interface(text):
    return summarizer.summarize_text(text)

gr.Interface(
    fn=summarize_interface,
    inputs=gr.Textbox(lines=20, placeholder="Paste or upload long text here...", label="Input Text"),
    outputs=gr.Textbox(label="Summary"),
    title="🚀 Transformer-based Text Summarizer (BART-Large-CNN)",
    description="Performs advanced abstractive summarization using Hugging Face's BART transformer."
).launch()
