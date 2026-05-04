# Multilingual-News-Articles-Summarization

### 1.0 INTRODUCTION
This project addresses information overload by providing an automated tool to distill critical insights from news articles across multiple languages. By integrating advanced **Natural Language Processing (NLP)** models, the system offers a hybrid approach to summarization, combining the contextual understanding of **BERT** for extractive tasks with the generative fluency of the **T5 Transformer** for abstractive tasks. The tool supports diverse article categories such as Politics, Sports, and Technology, ensuring high-quality, coherent summaries regardless of the input language.

**Key features of this project:**
*   **Hybrid Summarization Architecture**: Leverages BERT for identifying key sentences and T5 for rewriting content into fluent summaries.
*   **Multilingual Support**: Automatically detects input languages (Spanish, French, Chinese, German, etc.) and provides translations to and from English.
*   **Multi-Source Input**: Supports direct text input, URL fetching via web scraping, and file uploads (PDF, DOCX, TXT).
*   **Real-time Evaluation**: Provides quantitative feedback on summary quality using ROUGE and BERTScore metrics.

### 2.0 OBJECTIVES
***
*   Automate news article summarization by implementing both extractive and abstractive NLP techniques.
*   Enhance global accessibility by offering seamless multilingual translation and summarization.
*   Validate the effectiveness of different model configurations (e.g., BERT vs. Transformer encoding/decoding) through standardized evaluation metrics.

### 3.0 DATA SOURCES
***
The system is tested against a heterogeneous corpus designed to evaluate linguistic versatility:
*   **Multilingual Articles**: Includes content in Spanish, French, Chinese, and German.
*   **Category-Specific Data**: Articles covering Politics, Sports, Technology, Health, and Crime.
*   **Preprocessing Pipeline**: Cleans text by removing special characters, truncating content to a maximum of **512 tokens**, and utilizing the **Google Translator API** for language uniformity.

### 4.0 RUNNING THE SCRIPT
***
To use the tool, you can execute any of the three model variants depending on the desired hybrid configuration:
*   **Model 1**: `python summarize1.py` (BERT Encoding + Transformer Decoding).
*   **Model 2**: `python summarize2.py` (Transformer Encoding/Decoding with BERT Extractive logic).
*   **Model 3**: `python summarize3.py` (Transformer Encoding + BERT Decoding).

**Workflow:**
1.  **Input**: Enter text manually, provide a URL, or upload a supported file.
2.  **Configuration**: Select the summarization method (Abstractive/Extractive), Article Category, and Target Language.
3.  **Process**: Click "Summarize" to generate the output and view performance metrics.
4.  **Save**: Use the "Save Summary" button to export the result as a text file.

### 5.0 EVALUATION
***
Summary quality is assessed using two primary quantitative measures:
*   **ROUGE Scores (Average)**: Calculates the precision, recall, and F1-score between the generated and reference summaries.
*   **BERTScore**: Measures semantic similarity using contextual embeddings to ensure the core meaning is preserved even when rewritten.
*   **Qualitative Analysis**: The system is evaluated for narrative coherence and factual accuracy across different domains.

### 6.0 REQUIREMENTS
***
*   **Core Libraries**: `transformers` (T5, BERT), `torch`, `nltk`.
*   **GUI & Processing**: `tkinter`, `PIL`, `docx`, `PyPDF2`.
*   **Translation & Scraping**: `deep-translator`, `langdetect`, `requests`, `beautifulsoup4`.
*   **Metrics**: `rouge-score`, `bert-score`.

### 7.0 TEAM
***
*   Faheyra Ezzah binti Musa
*   Azwa binti Mat Yasin
*   Anis Nazira Binti Abd Ghani
*   Fatheen Sara Sofiah binti Romy Norfidzy

### ***📝 AUTHOR***
***
***Fatheen Sara Sofiah binti Romy Norfidzy***

***This project is for educational purposes.***
