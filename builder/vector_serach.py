import asyncio
from functools import partial, wraps
from typing import Optional, List, Dict

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from config.static import SearchBalancer
from utils_vector.envhandler import get_env

class VectorSearchManager:
    """
    A class for managing vector search operations using MongoDB.
    """
    def __init__(self,
        database_name: str,
        collection_name: str,         
        path: str,
        index: str, 
        num_candidates: int = SearchBalancer.DEFAULT_NUM_CANDIDATES, 
        limit: int = SearchBalancer.DEFAULT_LIMIT_PER_GROUP,
        connec_client: Optional[MongoClient] = None 
        
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
            connec_client (Optional[MongoClient], optional): The MongoDB client to use/reuse. Defaults to None.
            
        """

        self.database_name = database_name
        self.collection_name = collection_name
        self.path = path
        self.index = index
        self.num_candidates = num_candidates
        self.limit = limit


        if not connec_client:
            self.db_uri = get_env('MONGODB_URI')
            self.client = MongoClient(self.db_uri, server_api=ServerApi('1'))
        else:
            self.client = connec_client



    def close(self):
        """
        Closes the MongoDB client connection.
        """
        self.client.close()

    async def request(self, embedding, **fields) -> List[Dict]:
        async_func = self.async_wrap(self._request)
        return await async_func(embedding, **fields)

    def _request(self,  embedding, **fields) -> List[Dict]:
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
    

    def async_wrap(self, func):
        @wraps(func)
        async def run(*args, loop=None, executor=None, **kwargs):
            if loop is None:
                loop = asyncio.get_event_loop()
            pfunc = partial(func, *args, **kwargs)
            return await loop.run_in_executor(executor, pfunc)
        return run