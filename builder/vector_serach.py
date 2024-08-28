from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from utils_vector.envhandler import get_env

_DEFAULT_NUM_CANDIDATES = 50
_DEFAULT_LIMIT = 50

class VectorSearchManager:
    """
    A class for managing vector search operations using MongoDB.
    """
    def __init__(self,
        database_name: str,
        collection_name: str,         
        path: str,
        index: str, 
        num_candidates: int = _DEFAULT_NUM_CANDIDATES, 
        limit: int = _DEFAULT_LIMIT, 
        
        ):
        """
        Initializes the VectorSearchManager class.

        Args:
            database_name (str): The name of the MongoDB database.
            collection_name (str): The name of the MongoDB collection.
            path (str, optional): The path to the vector field in the collection.
            index (str, optional): The name of the vector index to use for the search.
            num_candidates (int, optional): The number of candidate documents to consider for the search. Defaults to 50.
            limit (int, optional): The maximum number of documents to return in the search results. Defaults to 50.
            
        """

        self.database_name = database_name
        self.collection_name = collection_name
        self.path = path
        self.index = index
        self.num_candidates = num_candidates
        self.limit = limit
        
        self.db_uri = get_env('MONGODB_URI')
        self.client = MongoClient(self.db_uri, server_api=ServerApi('1'))

    def close(self):
        """
        Closes the MongoDB client connection.
        """
        self.client.close()

    def request(self,  embedding, **fields):
        """
        Executes a vector search on the MongoDB collection.

        Args:
            embedding (list): The query vector for the search.
            **fields: Additional fields to project in the search results.

        Returns:
            list: The search results as a list of documents.
        """
        db = self.client[self.database_name]
        collection = db[self.collection_name]

        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": embedding,
                    "path": self.path,
                    "numCandidates": self.num_candidates,
                    "limit": self.limit,
                    "index": self.index
                }
            },
            {
                "$project": fields
            }
        ]

        return list(collection.aggregate(pipeline))
    

