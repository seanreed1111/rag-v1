import os, sys
from pathlib import Path
import json
import tiktoken
import openai
import numpy as np
import pickle
from dotenv import load_dotenv
from loguru import logger
from langchain.embeddings import OpenAIEmbeddings
# Load environment variables
load_dotenv("embedding.env")

# Configure Azure OpenAI Service API
EMBEDDING_API_KEY=os.getenv('EMBEDDING_API_KEY')
EMBEDDING_API_BASE=os.getenv('EMBEDDING_API_BASE')
EMBEDDING_API_TYPE = os.getenv('EMBEDDING_TYPE')
EMBEDDING_API_VERSION = os.getenv('EMBEDDING_VERSION')
EMBEDDING_DEPLOYMENT_NAME=os.getenv('EMBEDDING_DEPLOYMENT_NAME')
EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CHUNK_SIZE = 1000



embeddings = OpenAIEmbeddings(
                openai_api_key=EMBEDDING_API_KEY,
                openai_api_base=EMBEDDING_API_BASE,
                openai_api_type=EMBEDDING_API_TYPE,
                openai_api_version = EMBEDDING_API_VERSION,
                deployment=EMBEDDING_DEPLOYMENT_NAME,
                show_progress_bar=True
)
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
logger.debug(f"{openai.api_type}, {openai.api_version}, {openai.api_base}")

from llama_index import SimpleDirectoryReader
cwd = Path.cwd()
data_path = cwd / "data"
# list filenames in the directory
files = [f for f in data_path.iterdir() if f.is_file()]
logger.info(json.dumps(files))
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = PyPDFLoader(files[0])
pages = loader.load_and_split()
logger.info(json.dumps(pages))
