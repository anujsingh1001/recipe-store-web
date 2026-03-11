import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load documents from folder
loader = DirectoryLoader("./recipe_txt", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

print("Loaded Documents:")
print(documents)

# Split documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)

print("Text Chunks:")
print(texts)

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create vector database
docsearch = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory="db"
)

# Query
query = "suggest me some banana recipe"

# Create retriever
retriever = docsearch.as_retriever(search_kwargs={"k": 2})

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Run query
result = qa_chain(query)

print("\nAnswer:")
print(result["result"])