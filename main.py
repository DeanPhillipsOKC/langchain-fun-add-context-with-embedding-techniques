from dotenv import load_dotenv

# A loader can open file(s) from a source and create a document
#   documents  
#       page_content: <contents of file>
#       metadata: ("source": "facts.txt")
from langchain.document_loaders import TextLoader

load_dotenv()

loader = TextLoader("facts.txt")
docs = loader.load()

print(docs)
