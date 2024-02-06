from dotenv import load_dotenv

# A loader can open file(s) from a source and create a document
#   documents  
#       page_content: <contents of file>
#       metadata: ("source": "facts.txt")
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    # This isn't very intuitive.  It will basically grab 200 characters (chunk_size) and then read backwards to the nearest separator and use
    # that as a chunk.
    chunk_size=200,
    # this one will cause an overlap that will start the next chunk by reading backwards.  The purpose of this is to avoid awkard chunks in
    # which context might get lost through chunking.
    chunk_overlap=0
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

db = Chroma.from_documents(
    docs,
    # mismatching convention on the part of LangChain
    embedding=embeddings,
    persist_directory="emb"
)

results = db.similarity_search(
    "What is an interesting fact about food",
    k=4)

for result in results:
    print("\n")
    print(result.page_content)