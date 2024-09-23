import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""  # Initializing a variable to store all the text from PDFs
    for pdf in pdf_docs:  # Loop to traverse all the PDFs
        pdf_reader = PdfReader(pdf)  # Initialized PdfReader object for each PDF
        for page in pdf_reader.pages:  # Looping through all the pages of pdf_reader
            text += page.extract_text()
    return text   

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(  # Initializing the instance of CharacterTextSplitter
        separator="\n",  # Line break is used as separator
        chunk_size=1000,  # Size of chunk is 1000 characters
        chunk_overlap=200,  # To prevent only part of a sentence from getting selected when the end of the chunk is mid-sentence
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.5, "max_length":2048})

    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

# def handle_user_input(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.conversation):
#         if i % 2 == 0:
#             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):  # Changed from st.session_state.conversation
        # Print the type and content of message
        print(message)
        print(type(message))

        if i % 2 == 0:
            # Check if message is a tuple or another structure
            if isinstance(message, tuple):
                # Adjust based on what you find from print
                content = message[0]  # or another index if needed
            else:
                content = message.content  # if it's an object with 'content' attribute
            
            st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
        else:
            # Handle bot messages similarly
            if isinstance(message, tuple):
                content = message[0]  # or another index if needed
            else:
                content = message.content
            
            st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)


def main():
    load_dotenv() 
    st.set_page_config(page_title="Chat With Multiple PDFs", page_icon=":books:")

    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat With Multiple PDFs :books:")
    
    user_question = st.text_input("Ask a question about your documents:")


    if user_question:
        handle_user_input(user_question)

    # st.write(user_template.replace("{{MSG}}","Hello Chatbot"),unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}","Hello Brother"),unsafe_allow_html=True)
    


    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.write("Vector store created successfully!")

                # Create a conersation chain
                st.session_state.conversation=get_conversation_chain(vectorstore)

                
if __name__ == '__main__':
    main()
