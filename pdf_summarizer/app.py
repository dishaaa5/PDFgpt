import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer
from gtts import gTTS
import tempfile

# Streamlit app setup
st.set_page_config(page_title="PDFGPT", page_icon="📘")
st.title("📘 PDF Summarizer")

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

def split_text_by_tokens(text, tokenizer, max_tokens=900):
    """Split text into smaller, token-safe chunks."""
    words = text.split()
    chunks, current_chunk = [], ""
    for word in words:
        if len(tokenizer.tokenize(current_chunk + " " + word)) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = word
        else:
            current_chunk += " " + word
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

if uploaded_file is not None:
    st.info("📄 Extracting text from PDF...")
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()

    if not text.strip():
        st.warning("⚠️ No readable text found in the PDF.")
    else:
        with st.expander("Show Extracted Text"):
            st.write(text[:2000] + "...")

        if st.button("✨ Summarize"):
            with st.spinner("Summarizing... Please wait ⏳"):
                model_name = "facebook/bart-large-cnn"  # stable and local
                summarizer = pipeline("summarization", model=model_name, framework="pt")
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                chunks = split_text_by_tokens(text, tokenizer)
                summaries = []

                for i, chunk in enumerate(chunks):
                    st.write(f"🔹 Summarizing part {i+1}/{len(chunks)}...")
                    summary = summarizer(
                        chunk, max_length=180, min_length=40, do_sample=False
                    )[0]['summary_text']
                    summaries.append(summary)

                final_summary = " ".join(summaries)

            st.subheader("🧠 Final Summary:")
            st.write(final_summary)

            # Convert to audio
            with st.spinner("🎧 Generating audio..."):
                tts = gTTS(final_summary)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(temp_file.name)
                st.audio(temp_file.name, format="audio/mp3")

            st.success("✅ Summary and audio generated successfully!")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 14px;
        text-align: center;
    }
    </style>
    <div class="footer">
        Made with ❤️ by Disha
    </div>
    """,
    unsafe_allow_html=True
)
