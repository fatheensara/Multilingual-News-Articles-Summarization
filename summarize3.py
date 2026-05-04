import tkinter as tk
from tkinter import ttk, filedialog
import nltk
from transformers import BertTokenizer, BertModel, T5ForConditionalGeneration, T5Tokenizer
import torch
from nltk.tokenize import sent_tokenize
from PIL import Image, ImageTk
from docx import Document
import PyPDF2
import requests
from deep_translator import GoogleTranslator
from bs4 import BeautifulSoup
from langdetect import detect
from rouge_score import rouge_scorer
from bert_score import score 

# Download NLTK data
nltk.download('punkt')

# Initialize Tkinter window
root = tk.Tk()
root.title("News Article Summarizer")
root.geometry("800x600")

# Create the canvas
canvas = tk.Canvas(root, highlightthickness=0)
canvas.place(x=0, y=0, relwidth=1, relheight=1)

# Load the background image
bg_image = Image.open("background2.jpg")
bg_photo = ImageTk.PhotoImage(bg_image)

# Display the background image on the canvas
bg_image_item = canvas.create_image(0, 0, anchor=tk.NW, image=bg_photo)

# Function to resize the background image dynamically
def resize_bg(event):
    new_width, new_height = event.width, event.height
    resized_image = bg_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    bg_photo_resized = ImageTk.PhotoImage(resized_image)
    canvas.itemconfig(bg_image_item, image=bg_photo_resized)
    canvas.image = bg_photo_resized

root.bind("<Configure>", resize_bg)

# Add the title
app_title = ttk.Label(canvas, text="News Article Summarizer", font=("Arial", 20, "bold"))
app_title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

# Create a frame
frame = ttk.Frame(canvas, padding="10")
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Add labels and textboxes
label = ttk.Label(frame, text="Enter News Article:", font=("Arial", 14))
label.grid(row=0, column=0, pady=10, sticky=tk.W)

article_text = tk.Text(frame, width=60, height=8)
article_text.grid(row=1, column=0, pady=10)

# Preprocess text
def preprocess_text(text):
    # Remove special characters, excessive spaces, and truncate
    clean_text = text.replace("\n", " ").strip()
    max_length = 512  # Truncate input for T5
    return clean_text[:max_length]

# Function to load a file
def load_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Text Files", ".txt"), ("PDF Files", ".pdf"), ("Word Documents", "*.docx")]
    )
    if not file_path:
        return

    content = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            content = "".join(page.extract_text() for page in reader.pages)
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        content = "\n".join(paragraph.text for paragraph in doc.paragraphs)

    article_text.delete(1.0, tk.END)
    article_text.insert(tk.END, content)

# Function to fetch article from URL
def fetch_from_url():
    url = url_entry.get().strip()
    if not url:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Please enter a valid URL.")
        result_text.config(state=tk.DISABLED)
        return

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join(paragraph.get_text() for paragraph in paragraphs)

        article_text.delete(1.0, tk.END)
        article_text.insert(tk.END, content)
    except Exception as e:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Error fetching URL: {str(e)}")
        result_text.config(state=tk.DISABLED)

# Custom summarization pipeline using Transformer for Encoding and BERT for Decoding
class TransformerBertSummarizer:
    def __init__(self):
        # Initialize T5 for encoding
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        
        # Initialize BERT for decoding
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

    def encode_transformer(self, text):
        # Encode input text using T5 to generate embeddings
        input_ids = self.t5_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            encoder_outputs = self.t5_model.encoder(input_ids)
        return encoder_outputs.last_hidden_state

    def decode_bert(self, transformer_embeddings, sentences):
        # Use BERT to rank sentences based on transformer embeddings
        inputs = self.bert_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            sentence_embeddings = self.bert_model(**inputs).last_hidden_state.mean(dim=1)
            transformer_summary_vector = transformer_embeddings.mean(dim=1)
            
            # Calculate similarity scores
            relevance_scores = torch.cosine_similarity(sentence_embeddings, transformer_summary_vector, dim=1)
        
        # Rank sentences based on scores
        ranked_sentences = [sentence for _, sentence in sorted(zip(relevance_scores, sentences), reverse=True)]
        return " ".join(ranked_sentences[:3])  # Return the top 3 sentences

    def summarize_abstractive(self, text, category):
        # Encode with T5
        input_text = f"{category}: {text}"
        transformer_embeddings = self.encode_transformer(input_text)
        
        # Decode with BERT
        sentences = sent_tokenize(text)
        summary = self.decode_bert(transformer_embeddings, sentences)
        return summary

    def summarize_extractive(self, text, category):
        # Use only BERT for extractive summarization
        sentences = sent_tokenize(text)
        category_tokens = self.bert_tokenizer(category, return_tensors="pt", truncation=True, max_length=512)
        inputs = self.bert_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            category_embedding = self.bert_model(**category_tokens).last_hidden_state.mean(dim=1)
            sentence_embeddings = self.bert_model(**inputs).last_hidden_state.mean(dim=1)
            relevance_scores = torch.cosine_similarity(sentence_embeddings, category_embedding, dim=1)
        
        ranked_sentences = [sentence for _, sentence in sorted(zip(relevance_scores, sentences), reverse=True)]
        return " ".join(ranked_sentences[:3])

    def summarize(self, text, category, method="abstractive"):
        if method == "abstractive":
            return self.summarize_abstractive(text, category)
        elif method == "extractive":
            return self.summarize_extractive(text, category)




# Evaluation functions
def calculate_rouge(reference, summary):
    if not reference or not summary:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    avg_rouge_score = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
    return avg_rouge_score

def calculate_bert_score(reference, summary):
    P, R, F1 = score([summary], [reference], lang="en", verbose=False)
    return F1.mean().item()

def generate_summary():
    article = article_text.get("1.0", tk.END).strip()
    if not article:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Please enter an article text.")
        result_text.config(state=tk.DISABLED)
        return

    try:
        input_language = detect(article)
    except Exception as e:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Error detecting language: {str(e)}")
        result_text.config(state=tk.DISABLED)
        return

    if input_language != "en":
        try:
            translator = GoogleTranslator(source=input_language, target="en")
            article = translator.translate(article)
        except Exception as e:
            result_text.config(state=tk.NORMAL)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Error translating article to English: {str(e)}")
            result_text.config(state=tk.DISABLED)
            return

    article = preprocess_text(article)
    method = summary_method.get()
    category = category_combobox.get()
    summarizer = TransformerBertSummarizer()

    try:
        if method == "abstractive":
            summary = summarizer.summarize_abstractive(article, category)
        else:
            summary = summarizer.summarize_extractive(article, category)

        # Calculate metrics
        rouge_score = calculate_rouge(article, summary)
        bert_score = calculate_bert_score(article, summary)

        # Translate summary if output language is not English
        target_language = lang_combobox.get()
        if target_language != "English":
            try:
                lang_code = {"English": "en", "Spanish": "es", "French": "fr", "German": "de",
                             "Italian": "it", "Portuguese": "pt", "Russian": "ru", "Chinese": "zh",
                             "Arabic": "ar", "Hindi": "hi"}[target_language]
                translator = GoogleTranslator(source="en", target=lang_code)
                summary = translator.translate(summary)
            except Exception as e:
                summary = f"Translation failed: {str(e)}"

        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Summary:\n{summary}\n\n")
        result_text.insert(tk.END, f"Rouge Score (average): {rouge_score:.4f}\n")
        result_text.insert(tk.END, f"BERTScore: {bert_score:.4f}\n")
        result_text.config(state=tk.DISABLED)
    except Exception as e:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Error generating summary: {str(e)}")
        result_text.config(state=tk.DISABLED)


# Save summary, clear text, and GUI components remain unchanged.
# Function to save summary
def save_summary():
    summary = result_text.get("1.0", tk.END).strip()
    if not summary:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "No summary to save.")
        result_text.config(state=tk.DISABLED)
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if not file_path:
        return

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(summary)

# Function to clear text boxes
def clear_text():
    article_text.delete(1.0, tk.END)
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.config(state=tk.DISABLED)

# Input buttons
input_frame = ttk.Frame(frame)
input_frame.grid(row=2, column=0, pady=10, sticky=tk.W)

file_button = ttk.Button(input_frame, text="Upload File", command=load_file)
file_button.pack(side=tk.LEFT, padx=5)

url_label = ttk.Label(input_frame, text="OR Enter URL:")
url_label.pack(side=tk.LEFT, padx=5)

url_entry = ttk.Entry(input_frame, width=30)
url_entry.pack(side=tk.LEFT, padx=5)

fetch_button = ttk.Button(input_frame, text="Fetch", command=fetch_from_url)
fetch_button.pack(side=tk.LEFT, padx=5)

# Dropdowns
summary_method_label = ttk.Label(frame, text="Select Summarization Method:", font=("Arial", 12))
summary_method_label.grid(row=3, column=0, pady=10, sticky=tk.W)

summary_method = ttk.Combobox(frame, values=["abstractive", "extractive"], state="readonly")
summary_method.grid(row=4, column=0, pady=10)
summary_method.set("abstractive")

lang_label = ttk.Label(frame, text="Select Output Language:", font=("Arial", 12))
lang_label.grid(row=5, column=0, pady=10, sticky=tk.W)

lang_combobox = ttk.Combobox(frame, values=["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Chinese", "Arabic", "Hindi"], state="readonly")
lang_combobox.grid(row=6, column=0, pady=10)
lang_combobox.set("English")

# Category Selection
category_label = ttk.Label(frame, text="Select Category:", font=("Arial", 12))
category_label.grid(row=7, column=0, pady=10, sticky=tk.W)

category_combobox = ttk.Combobox(
    frame,
    values=["Sports", "Crime", "Politics", "Entertainment", "Technology", "Health", "Other"],
    state="readonly"
)
category_combobox.grid(row=8, column=0, pady=10)
category_combobox.set("Other")

# Buttons
button_frame = ttk.Frame(frame)
button_frame.grid(row=9, column=0, pady=20, padx=120, sticky=tk.E)

summarize_button = ttk.Button(button_frame, text="Summarize", command=generate_summary)
summarize_button.pack(side=tk.LEFT, padx=5)

clear_button = ttk.Button(button_frame, text="Clear", command=clear_text)
clear_button.pack(side=tk.LEFT, padx=5)

save_button = ttk.Button(button_frame, text="Save Summary", command=save_summary)
save_button.pack(side=tk.LEFT, padx=5)

# Output Textbox
result_label = ttk.Label(frame, text="Summary Output:", font=("Arial", 14))
result_label.grid(row=10, column=0, pady=10, sticky=tk.W)

result_text = tk.Text(frame, width=60, height=8, state=tk.DISABLED)
result_text.grid(row=11, column=0, pady=10)

# Start the Tkinter main loop
root.mainloop()