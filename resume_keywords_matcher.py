# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:01:01 2025

@author: sansk
"""
import spacy
import fitz  # PyMuPDF for PDF processing
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sentence_transformers import SentenceTransformer

# Load NLP model
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_keywords(text):
    """Extracts keywords from text using spaCy."""
    doc = nlp(text.lower())
    keywords = {token.lemma_ for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB"] and not token.is_stop}
    return list(keywords)

def calculate_similarity(resume_text, job_desc_text):
    """Calculates similarity between resume and job description using BERT embeddings."""
    resume_embedding = bert_model.encode([resume_text], convert_to_tensor=True)
    job_desc_embedding = bert_model.encode([job_desc_text], convert_to_tensor=True)
    similarity_score = cosine_similarity(resume_embedding.cpu().numpy(), job_desc_embedding.cpu().numpy())[0][0]
    return similarity_score

def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF resume."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

def main():
    st.title("📄 Resume Keyword Matcher")
    st.write("Upload your resume or paste the text, and provide a job description to check keyword match and similarity score.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
    with col2:
        resume_text = st.text_area("Or, Paste Your Resume Text")
    
    job_desc_text = st.text_area("Paste Job Description")
    
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ PDF uploaded successfully!")
    
    if st.button("🔍 Match Keywords"):
        if resume_text and job_desc_text:
            with st.spinner("Processing..."):
                resume_keywords = extract_keywords(resume_text)
                job_keywords = extract_keywords(job_desc_text)
                missing_keywords = set(job_keywords) - set(resume_keywords)
                similarity_score = calculate_similarity(resume_text, job_desc_text)
            
            st.subheader(f"✅ Match Score: {similarity_score * 100:.2f}%")
            
            st.markdown("### **Keywords Found in Your Resume:**")
            st.success(", ".join(resume_keywords) if resume_keywords else "No keywords found.")
            
            with st.expander("🔻 Important Keywords Missing"):
                st.warning(", ".join(missing_keywords) if missing_keywords else "None ✅")
        else:
            st.error("Please provide both a resume and job description.")

if __name__ == "__main__":
    main()
