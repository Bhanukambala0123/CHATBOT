import os
#from langchain_community.llms import Ollama
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from groq import Groq
import random
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma 
#from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from getpass import getpass
from langchain_ai21 import AI21Embeddings

AI21_API_KEY = os.environ["AI21_API_KEY"]

#model = Ollama(model="llama3")

os.environ["GROQ_API_KEY"] = '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'
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

def main():
    groq_api_key = os.environ["GROQ_API_KEY"]
    
    spacer, col = st.columns([5,1])
    with col:
        st.image('a.jpg')
    

    st.title('RERATE CHATBOT')
    st.write("Hello!, I am your friendly chatbot. Please ask any questions.")

    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
    'choose a model',
    ['llama3-70b-8192', 'llama3-8b-8192']
     )

    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value =5)
    memory = ConversationBufferWindowMemory(k=conversational_memory_length)
    user_question = st.text_input("Ask a Question..")

            
    loader = TextLoader('./constitution.txt')
    
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #\n\n, \n, ' '

    chunks = text_splitter.split_documents(documents)    

    embeddings = AI21Embeddings()

    vector_store = Chroma.from_documents(chunks, embeddings)
    
    
    groq_chat= ChatGroq(
    groq_api_key = groq_api_key,
    model_name=model 
    )
    
    retriever=vector_store.as_retriever()
    
    #chain = RetrievalQA.from_chain_type(groq_chat, retriever=retriever )
    crc = ConversationalRetrievalChain.from_llm(groq_chat,retriever)
    st.session_state.crc = crc

    #conversation = ConversationChain(
    #llm=groq_chat,
    #memory=memory
    #)

    if user_question:
        if 'crc' in st.session_state:
            crc = st.session_state.crc
            if 'history' not in st.session_state:
                st.session_state['history'] = []

            response = crc.run({'question':user_question,'chat_history':st.session_state['history']})

            st.session_state['history'].append((user_question,response))
            st.write(response)

        #st.write(st.session_state['history'])
            for prompts in st.session_state['history']:
                st.write("Question: " + prompts[0])
                st.write("Answer: " + prompts[1])  

    


if __name__ == "__main__":
    main()
    


    
    
