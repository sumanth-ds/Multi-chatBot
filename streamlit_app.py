import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)

import tempfile

import warnings
warnings.filterwarnings('ignore')

from langchain_core.runnables import RunnablePassthrough, RunnableLambda


from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_core.messages import SystemMessage
from langchain_together import ChatTogether

st.sidebar.subheader("To crear the chat refresh the page")
system_prompt = st.sidebar.text_area(label="System Prompt")
creativity_level = st.sidebar.slider(label="Choose Model Creativity [1:low , 10:high]" , min_value = 1, max_value = 10)
chat_type = st.sidebar.selectbox(label="Select ChatBot", options=["Normal Chatbot", "Chat with CSV (Rag)", "Chat with PDF (Rag)", "Chat with URL (Rag)"])


google = st.secrets['google']
together = st.secrets['together']


# google_api = open("gemini-demo.txt").read()
# together_api = open("Q_A/togetherai.txt").read()

# model = ChatGroq(api_key=open('Groq.txt').read() , model= "llama-3.2-90b-vision-preview")

model = ChatGoogleGenerativeAI(api_key= google, model = "gemini-1.5-flash", temperature=creativity_level/10) #'llama-3.2-90b-vision-preview') #'llama-3.2-11b-vision-preview' )#'llama-3.2-90b-vision-preview')
memory = ConversationBufferMemory( memory_key="chat_history", return_messages=True)
string = StrOutputParser()


chat = ChatTogether(
    model="meta-llama/Llama-Vision-Free",
    api_key=together
)

st.title("Welcome to Multi-ChatBot")


chatBots= ["Normal Chatbot", "Chat with CSV (Rag)", "Chat with PDF (Rag)", "Chat with URL (Rag)"]

for chat in chatBots:
    if chat == chat_type:
        st.subheader(f"{chat}")


if chat_type == "Chat with CSV (Rag)":
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    

if chat_type == "Chat with PDF (Rag)":
    uploaded_file = st.file_uploader("Choose a CSV file", type=["pdf"])
    

if chat_type == "Chat with URL (Rag)":
    html = st.text_input("Provide URL")
    



if 'chat_list' not in st.session_state:
    st.session_state.chat_list = []


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for messages in st.session_state.chat_list:
    st.chat_message(messages['role']).write(messages['content'])




for messages in st.session_state.chat_history:
    human_input = messages['human']
    ai_output = messages.get('AI')
    if human_input and ai_output:
        memory.save_context({'input': messages['human']}, {'output': messages.get('AI')})



prompt = st.chat_input()
if prompt:
    st.chat_message('human').write(prompt)



if chat_type == "Normal Chatbot":
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=system_prompt
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),
        ]
    )

    conversation = LLMChain(
        llm=model,
        prompt=prompt_template,
        verbose=True,
        memory=memory,
    )
    
    if prompt:
        # st.chat_message("user").write(prompt)
        try:
            response = conversation.invoke(prompt)['text']

        except Exception as e:
            st.error(f"Error executing agent: {e}")
            response = "Sorry, there was an issue processing your request."

        
        st.chat_message('assistant').write(response)
        
        st.session_state.chat_list.append({'role':'user', 'content':prompt})
        st.session_state.chat_list.append({'role':'assistant', 'content':response})
        messages = {'human' : prompt , 'AI' : response}
        st.session_state.chat_history.append(messages)

if chat_type == "Chat with CSV (Rag)":

    if uploaded_file:
        var = uploaded_file.name
        if '.csv' in var:
            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                tmpfile_path = tmpfile.name

            agent =  create_csv_agent(
                llm = model,
                path=tmpfile_path,
                allow_dangerous_code=True,
                handle_parsing_errors=True ,
                verbose = True,
            )


            if prompt: 
                # st.chat_message('user').write(prompt)
        
                chat_history = memory.load_memory_variables({})["chat_history"]
                history_context = "\n".join(
                    f"User: {msg.content}" if msg.type == "human" else f"AI: {msg.content}"
                    for msg in chat_history)

                full_prompt = f"Previous conversation history:\n{history_context}\n\nNew query:\n{prompt}"


                try:
                
                    output = agent.invoke(full_prompt)
                    print(output)
                    response = output['output']

                except Exception as e:
                    st.error(f"Error executing agent: {e}")
                    response = "Sorry, there was an issue processing your request."

                st.chat_message('assistant').write(response)
                
                st.session_state.chat_list.append({'role':'user', 'content':prompt})
                st.session_state.chat_list.append({'role':'assistant', 'content':response})
                messages = {'human' : prompt , 'AI' : response}
                st.session_state.chat_history.append(messages)

if chat_type == "Chat with PDF (Rag)":
    if uploaded_file:
        var = uploaded_file.name
        if '.pdf' in var:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                tmpfile_path = tmpfile.name
                loader = PyMuPDFLoader(tmpfile_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                embedding  = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001', google_api_key=google )
                vector_store = FAISS.from_documents(texts, embedding)
                def retriever(query):
                    retrieval = vector_store.as_retriever()
                    context = retrieval.invoke(query) 
                    return '\n\n'.join([i.page_content for i in context])
                prompting = """Your are helpfull ai assistant. Your name is qa Bot.  Answer the question based on the context : {context}. Answer the question {question}.""" + system_prompt
                prompt_template = ChatPromptTemplate.from_template(prompting)
                custom_chain = {"context" : retriever, "question": RunnablePassthrough()} | prompt_template | model | string



                if prompt:
                    # st.chat_message('user').write(prompt)
                    try:
                    
                        response = custom_chain.invoke(prompt)

                    except Exception as e:
                        st.error(f"Error executing agent: {e}")
                        response = "Sorry, there was an issue processing your request."


                    st.chat_message('assistant').write(response)
                    
                    st.session_state.chat_list.append({'role':'user', 'content':prompt})
                    st.session_state.chat_list.append({'role':'assistant', 'content':response})
                    messages = {'human' : prompt , 'AI' : response}
                    st.session_state.chat_history.append(messages)



if chat_type == "Chat with URL (Rag)":
    if html:
        response = requests.get(html)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser').get_text()
            text = " ".join(soup.split())
            documents =  [Document(page_content=text, metadata={"source": html})]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=500)
            texts = text_splitter.split_documents(documents)
            embedding  = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001', google_api_key=google )
            vector_store = FAISS.from_documents(texts, embedding)
            def retriever(query):
                retrieval = vector_store.as_retriever()
                context = retrieval.invoke(query) 
                return '\n\n'.join([i.page_content for i in context])
            prompting = """You are a helpful AI assistant. Your purpose is to answer {question} based on the given context: {context} correctly.If incase questions are not relevent to context, then please ask user to provide next next question politly """ + system_prompt
            prompt_template = ChatPromptTemplate.from_template(prompting)
            custom_chain = {"context" : retriever, "question": RunnablePassthrough()} | prompt_template | model | StrOutputParser()

            
            if prompt:
                # st.chat_message('user').write(prompt)
                try:
                
                    response = custom_chain.invoke(prompt)

                except Exception as e:
                    st.error(f"Error executing agent: {e}")
                    response = "Sorry, there was an issue processing your request."


                st.chat_message('assistant').write(response)
                
                st.session_state.chat_list.append({'role':'user', 'content':prompt})
                st.session_state.chat_list.append({'role':'assistant', 'content':response})
                messages = {'human' : prompt , 'AI' : response}
                st.session_state.chat_history.append(messages)
                
        else:
            st.chat_message('ai').write(f"Failed to retrieve the webpage. Status code: {response.status_code}")
