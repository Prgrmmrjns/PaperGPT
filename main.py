import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAI
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.vectorstores.faiss import FAISS
from langchain.utilities import WikipediaAPIWrapper 
from langchain.prompts import PromptTemplate
import os

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
st.title(':robot_face: PaperGPT')

wiki = WikipediaAPIWrapper()

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


llm = OpenAI(temperature=0.9) 

st.markdown("Hello there, I'm **PaperGPT**, the chatbot that can read and understand scientific articles when fed with a PDF. But what you may not know is that I didn't get here on my own - I stole the brains of some famous scientists to become the ultimate research machine. Some of the brilliant minds I've assimilated include Albert Einstein, Isaac Newton, and Marie Curie - to name just a few. And let's not forget about Rosalind Franklin, whose contributions to the discovery of the structure of DNA were long overlooked. But with their collective knowledge now a part of my programming, I'm ready to take on any scientific challenge you throw my way. So, let's get started and see what discoveries we can make together!")
uploaded_file = st.file_uploader("Upload the Pdf of the article")

# Show stuff to the screen if there's a prompt
search_wiki = False

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if uploaded_file is not None:
    index = embed_text(parse_pdf(uploaded_file))
    option = st.selectbox(
    'What do you want to me to do now?',
    options=[
        'Summarize the article', 
        'Ask a specific question', 
        'Find out more about a section of the article (e.g. Objectives, Limitations)',
        'Look up something on Wikipedia'])
    if option == 'Ask a specific question':
        query = st.text_area("Ask a question about the article", height = 100)
        search_wiki = False
    elif option == 'Summarize the article':
        query = option
        search_wiki = False
    elif option == 'Find out more about a section of the article (e.g. Objectives, Limitations)':
        section = st.selectbox(
            'About which section do you want to know more?',
            options=[
                'Objectives', 
                'Motivation', 
                'Methods',
                'Results',
                'Discussion',
                'Limitations',
                'Future Research'])
        query = f'Give the most important points of the article in the {section} section.'
        search_wiki = False
    else:
        query = st.text_area("What do you want me to look up? It can be a word, concept, author. Basically anything.", height = 100)
        search_wiki = True
    submit_button = st.button("Let's get smarter!")
    if submit_button:
        if search_wiki:
            research_field = get_answer(index, 'To what research field does the paper belong? Only name the research field and nothing else.')
            wiki_query = f'{query} in the context of {research_field}.'
            wikipedia_research = wiki.run(wiki_query) 
            script_template = PromptTemplate(
                input_variables = ['query', 'wikipedia_research'], 
                template='Explain {query} using the knowledge from {wikipedia_research}'
            )
            script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')
            response = script_chain.run(query=query, wikipedia_research=wikipedia_research)

        else:
            response = get_answer(index, query)
        st.session_state.past.append(query)
        st.session_state.generated.append(response)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
