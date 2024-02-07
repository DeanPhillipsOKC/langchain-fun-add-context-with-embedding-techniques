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
    # Refine is a strange one.
    # 1. In will grab the most relevant documents like the other chain types
    # 2. It will feed the top result into a prompt
    #    System: Use the following context to answer the question: {context}
    #    Human: Here is the user's question: {question}
    # 3. It will feed the answer into another prompt template
    #    Human: Here is the user's question: {question}
    #    AI: {Answer from previous step}
    #    Human: We have a chance to refine the answer using this additional context: {context from next document}
    # 4. So on and so fourth until we have a final, refined answer.
    # THOUGHTS: This seems terrible for this kind of use as they are just different facts and not context.  This might be great
    #           for things that parse say a technical manual and take several chunked documents into account to refine a good answer to
    #           the human user's question.
    chain_type="refine"
)

result = chain.run("What is an interesting fact about the English language?")

print("====== PRINTING THE RESULTS ======")
print(result)