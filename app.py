# https://github.com/seanreed1111/rag-v1.git
import json, os, sys, tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
# from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from htmlTemplates import css, bot_template, user_template
from loguru import logger

load_dotenv('deployment.env')
# model variables
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = 0

# embedding variables
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
EMBEDDING_API_TYPE = os.getenv('EMBEDDING_TYPE')
EMBEDDING_API_VERSION = os.getenv('EMBEDDING_VERSION')

def get_pages(pdf_doc):
    return PyPDFLoader(pdf_doc).load_and_split()


# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


def get_vectorstore(pages):
    embeddings = OpenAIEmbeddings(
                    openai_api_key=EMBEDDING_API_KEY,
                    openai_api_base=EMBEDDING_API_BASE,
                    openai_api_type=EMBEDDING_API_TYPE,
                    openai_api_version = EMBEDDING_API_VERSION,
                    deployment=EMBEDDING_DEPLOYMENT_NAME,
                    show_progress_bar=False
    )
    vectorstore = FAISS.from_documents(pages, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    chat = AzureChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    openai_api_base=OPENAI_API_BASE,
                    openai_api_type=OPENAI_API_TYPE,
                    openai_api_version=OPENAI_API_VERSION,
                    model_name=MODEL_NAME,
                    deployment=DEPLOYMENT_NAME,
                    temperature=TEMPERATURE,
                    request_timeout=20
    )

    memory = ConversationBufferWindowMemory(
        k=10,
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# uploader stores pdf document in memory, but langchain wants the file saved to disk
def get_file_path(pdf_doc):
    if pdf_doc is not None:
        file_details = {"FileName":pdf_doc.name,"FileType":pdf_doc.type}
        file_path = tempdir / pdf_doc.name
        st.write(file_details)
        st.write(file_path.resolve())
        logger.debug(pdf_doc.getbuffer())
        with open(file_path, "wb") as f:
            f.write(pdf_doc.getbuffer())
        return file_path.resolve()


def main(tempdir):
    load_dotenv()
    st.set_page_config(page_title="Retrieval Augmented Generation")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Retrieval Augmented Generation bot")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_doc = st.file_uploader("Upload your PDF here and click on 'Start'")
        file_path = get_file_path(pdf_doc, tempdir)
        if st.button("Start"):
            with st.spinner("Processing PDF"):
                # get pdf text
                pages = get_pages(pdf_doc)

                # # get the text chunks
                # text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(pages)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    logger.add(sys.stderr, format="{time} {level} {message}", level="DEBUG")
    logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>", level="DEBUG")
    tempdir = Path(tempfile.TemporaryDirectory())
    main(tempdir)
