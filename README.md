# 📘 PDFGPT – PDF Summarizer with Audio

PDFGPT is a **Streamlit-based web app** that allows users to upload a PDF, extract its text, generate an **AI-powered summary**, and convert that summary into **audio (MP3)** using Text-to-Speech.

---

## 🚀 Features

- 📤 Upload any PDF file  
- 📄 Extract readable text from PDFs  
- 🧠 AI-powered summarization using **HuggingFace BART**  
- ✂️ Automatic token-safe text chunking  
- 🎧 Convert summary into **audio narration (MP3)**  
- 🌐 Simple and interactive **Streamlit UI**

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **PyPDF2**
- **HuggingFace Transformers**
- **Facebook BART (bart-large-cnn)**
- **gTTS (Google Text-to-Speech)**

---

## ▶️ Run the Application 
streamlit run app.py

---

## 📋 requirements.txt
- streamlit
- PyPDF2
- transformers
- torch
- gtts

---
## 🧠 How It Works
- User uploads a PDF
- Text is extracted using PyPDF2
= Long text is split into token-safe chunks
- Each chunk is summarized using BART
- All summaries are merged into a final summary
- Summary is converted into MP3 audio

---
## 👩‍💻 Author

Disha

