import os
import json
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()



class faiss_vector_store:
    """This class is to check if the JSON file was left untouched or not. 
    If the file might be replaced (e.g., copied over, git pull) without its timestamp changing,
    then the vector DB is updated. If not, then it just calls the old vector DB.
    """
    def __init__(self, json_path = "data/json test.json", index_dir = "faiss_index", hash_file = "index_hash.json"):
        self.json_path = json_path
        self.index_dir = index_dir
        self.hash_file = hash_file
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self.load_or_update_faiss()

    def file_hash(self):
        """Function to check if there were any changes made to the data file,i.e., new entries in dataset.
        """
        with open(self.json_path, "rb") as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()

    def load_stored_hash(self):
        if not os.path.exists(self.hash_file):
            return None
        with open(self.hash_file, "r") as f:
            return json.load(f).get("hash")
        
    def save_hash(self, hash_val):
        with open(self.hash_file, "w") as f:
            json.dump({"hash": hash_val}, f)

    def load_documents(self):
        with open(self.json_path, 'r') as f:
            raw = json.load(f)
        return [Document(page_content=f"Q: {item['query']}\nA: {item['response']}",
                metadata={"source": item.get("source", "")})
                for item in raw]
    
    def load_or_update_faiss(self):
        """Function to load the database if no changes were made to dataset.
        Updates and loads the database if there were changes in the dataset

        Returns:
            pickle file: Vector database
        """
        current_hash = self.file_hash()
        stored_hash = self.load_stored_hash()

        if current_hash != stored_hash or not os.path.exists(self.index_dir):
            print("Detected changes in data. Rebuilding vector DB...")
            docs = self.load_documents()
            db = FAISS.from_documents(docs, self.embeddings)
            db.save_local(self.index_dir)
            self.save_hash(current_hash)
        else:
            print("No changes detected. Loading cached vector DB...")
            db = FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)
        
        return db

    def get_vectorstore(self):
        return self.vectorstore

