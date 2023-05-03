import streamlit as st
from langchain.llms import OpenAI
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
st.title(':robot_face: PaperGPT')

def parse_pdf(file):
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        output.append(text)

    return "\n\n".join(output)

def embed_text(text):
    """Split the text and embed it in a FAISS vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=0, separators=["\n\n", ".", "?", "!", " ", ""]
    )
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_texts(texts, embeddings)

    return index

def get_answer(index, query):
    """Returns answer to a query using langchain QA chain"""
    docs = index.similarity_search(query)
    llm = OpenAI(temperature=0.9) 
    chain = load_qa_chain(llm)
    answer = chain.run(input_documents=docs, question=query)

    return answer

st.markdown("Hello there, I'm **PaperGPT**, the chatbot that can read and understand scientific articles when fed with a PDF. But what you may not know is that I didn't get here on my own - I stole the brains of some famous scientists to become the ultimate research machine. Some of the brilliant minds I've assimilated include Albert Einstein, Isaac Newton, and Marie Curie - to name just a few. And let's not forget about Rosalind Franklin, whose contributions to the discovery of the structure of DNA were long overlooked. But with their collective knowledge now a part of my programming, I'm ready to take on any scientific challenge you throw my way. So, let's get started and see what discoveries we can make together!")
uploaded_file = st.file_uploader("Upload the Pdf of the article")

# Show stuff to the screen if there's a prompt
if uploaded_file is not None:
    index = embed_text(parse_pdf(uploaded_file))
    query = st.text_area("Ask a question about the article")
    button = st.button("Submit question")
    if button:
        st.write(get_answer(index, query))
    summarize = st.button("Summarize the article")
    if summarize:
        query = 'Give a detailed summary of the article.'
        st.write(get_answer(index, query))
    option = st.selectbox('Find out about',('Objectives', 'Motivation', 'Methods', 'Results', 'Discussion'))
    summarize_section = st.button("Get the most important points of the selected option")
    if summarize_section:
        query = f'Get the most important points of the study related to the {option}.'
        st.write(get_answer(index, query))