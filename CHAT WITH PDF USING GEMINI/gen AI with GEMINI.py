import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text using PyPDF2
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
            st.info(f"Text extracted from {pdf.name} using PyPDF2.")
        except Exception as e:
            st.warning(f"PyPDF2 failed for {pdf.name}. Trying OCR... Error: {e}")
            text += extract_text_with_ocr(pdf)
    return text

# OCR-based text extraction
def extract_text_with_ocr(uploaded_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        images = convert_from_path(tmp_file_path)
        for i, image in enumerate(images):
            ocr_text = pytesseract.image_to_string(image)
            text += ocr_text
            st.info(f"OCR extracted text from page {i+1}.")
    except Exception as e:
        st.error(f"OCR failed: {e}")
    finally:
        os.remove(tmp_file_path)

    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain setup
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available, say "Answer not available in the context."
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main app
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF using Gemini - RAG 💁")

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        response = user_input(user_question)
        st.session_state.qa_history.append(("Q: " + user_question, "A: " + response))

    if st.session_state.qa_history:
        for qa in st.session_state.qa_history:
            st.write(qa[0])  # Question
            st.write(qa[1])  # Answer

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text could be extracted from the uploaded files.")
                else:
                    st.success("Text Extraction Complete!")
                    st.text_area("Extracted Text Preview", raw_text, height=300)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Vector Store Creation Complete")

if __name__ == "__main__":
    main()
