import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3:8b",
    temperature=0
)

st.set_page_config(page_title="Article Analyzer", page_icon="📈", layout="wide")
st.title("News Research Tool 📈")
st.sidebar.title("News Articles URLs")

n = st.sidebar.text_input("Enter no. of URLs and press 'Enter'")
if n.isdigit():  # Check if input is a digit
    n = int(n)  # Convert input to integer
    urls = []
    for i in range(n):
        url = st.sidebar.text_input(f"URL {i + 1}")
        urls.append(url)
else:
    st.sidebar.error("Please enter a valid number.")



process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_pkl"

main_placefolder = st.empty()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placefolder.text("Splitting Text")
    docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding Vector")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.success(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.write("Sources: ")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)