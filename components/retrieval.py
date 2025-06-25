from components.vectorstore import faiss_vector_store
from dotenv import load_dotenv

load_dotenv()



class Retrieval:
    """This class helps in retrieving the vector database and feeding it to the LLM chain.
    """
    def __init__(self):
        self.vectorstore_instance = faiss_vector_store()
        self.db = self.vectorstore_instance.get_vectorstore()

    def get_db(self):
        return self.db