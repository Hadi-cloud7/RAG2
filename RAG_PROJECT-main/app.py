import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to generate quiz using Gemini
def generate_quiz(text, num_quizzes):
    model = genai.GenerativeModel("gemini-2.5-flash")
    quizzes = []
    for i in range(num_quizzes):
        # Use the full text as context (simplified without RAG)
        context = text[:2000]  # Limit to first 2000 characters for demo

        prompt = f"""
        Based on the following context, generate a multiple-choice quiz question with 4 options. Provide the question, options A, B, C, D, and the correct answer.

        Context: {context}

        Format:
        Question: [Question]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        Correct Answer: [Correct Option Letter]
        """

        response = model.generate_content(prompt)
        quizzes.append(response.text)
    return quizzes

# Streamlit UI
st.title("RAG-Based Quiz Generator from PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
num_quizzes = st.number_input("Number of Quizzes (1-20)", min_value=1, max_value=20, value=5)

if st.button("Generate Quizzes"):
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        with st.spinner("Generating Quizzes..."):
            quizzes = generate_quiz(text, num_quizzes)
        
        for i, quiz in enumerate(quizzes, 1):
            st.subheader(f"Quiz {i}")
            st.write(quiz)
    else:
        st.error("Please upload a PDF file.")
