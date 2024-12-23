import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")



st.title("RAG Doc Assistant Using Llama3")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template(
    """
    Please provide the most accurate response based on the context and inputs
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Function to process the uploaded PDF
def vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        # Save the uploaded file to a temporary location
        with open("temp_uploaded_file.pdf", "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        
        # Load and process the temporary file
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFLoader("temp_uploaded_file.pdf")  # Load saved file
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
        
        # Remove the temporary file after processing
        os.remove("temp_uploaded_file.pdf")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Embedding button
if st.button("Documents Embedding") and uploaded_file:
    vector_embedding(uploaded_file)
    st.write("Vector Store DB is ready")

# Input for the question
prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        # With a Streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.write("Please embed the document first.")
