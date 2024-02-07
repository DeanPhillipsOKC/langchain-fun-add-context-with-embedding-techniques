from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Tells langchain to print out all intermediate stuff so we can see what is going on with the prompts
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
retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    # Here's how the map_reduce chain type works...
    # 1. We get 4 documents from our vector store
    # 2. For each document, a prompt like this is sent to the LLM
    #   System: Use the following portion of a long document to see if any of the text is relevant to answer the question.  Return any relevant text verbatim.
    #           {document}
    #   User: Here is the user's question: {question}
    # 3. NOTE: Sometimes the document might not really be relevant to the question so Chat GPT might make up an answer based off of its training data
    # 4. All of the answers to the prompt are sent to another prompt
    #   System: Use the following context to answer the users question {summaries}
    #   User: Here is th euser's question: {question}
    chain_type="map_reduce"
)

result = chain.run("What is an interesting fact about the English language?")

print(result)