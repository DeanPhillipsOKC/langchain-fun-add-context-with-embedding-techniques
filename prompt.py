from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

# In LangChain a retriever is anything that has a method called "get_relevant_docuements" that takes a string
# and returns documents.
#
# In this case db has a similarity_search method.  When it is converted to a retriever it has a get_relevant_documents method that just calls
# similarity_search.
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    # Describes the fact that we are stuffing the context from the store into the prompt (not another word for "things")
    chain_type="stuff"
)

result = chain.run("What is an interesting fact about the English language?")

print(result)