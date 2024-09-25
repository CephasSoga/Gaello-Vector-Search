import time
import asyncio
import functools
from enum import Enum
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from vector_search.utils.envhandler import get_env
from vector_search.builder.executor import Executor
from vector_search.builder.embeddings import VectorEmbeddingManager
from vector_search.builder.context import Filter
from vector_search.config.static import SearchArgs, SearchBalancer
from vector_search.utils.logs import Logger, timer, async_timer

logger = Logger("Vector Search")

db_uri = get_env('MONGODB_URI')

@timer(logger=logger)
def create_client() -> MongoClient | None:
    # Create a MongoClient instance and configure the server API version to use.
    client = MongoClient(db_uri, server_api=ServerApi('1'))

    # Send a ping to confirm a successful connection
    try:
        s = time.perf_counter()
        logger.log("info", "Attempting to connect to MongoDB...")
        # Create a MongoClient instance and configure the server API version to use.
        client = MongoClient(db_uri, server_api=ServerApi('1'))
        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        e = time.perf_counter()
        logger.log("info", "Pinged your deployment. You successfully connected to MongoDB!")
        logger.log("info", f"Connected to MongoDB in {e-s:.4f} seconds!")
        return client
    except Exception as e:
        logger.log("error", f"Unable to connect to MongoDB: {e}")
        return None

class ConnectionErrors(Enum):
    CONNECTION_FAILURE = ConnectionFailure
    SERVER_SELECTION_TIMEOUT = ServerSelectionTimeoutError
    CONNECTION_ERROR = ConnectionError
    OTHER = Exception

def async_retry_on_connection_error(errors: List[ConnectionErrors], retries=3, delay=2, backoff=2):
    """
    A decorator that retries a function if a connection error occurs.
    
    Args:
        retries (int): The maximum number of retries before giving up.
        delay (int): Initial delay between retries in seconds.
        backoff (int): Multiplier applied to the delay between retries.
    """
    error_types = tuple(error.value for error in errors)

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            _retries, _delay = retries, delay
            while _retries > 0:
                try:
                    return await func(*args, **kwargs)
                except error_types as e:
                    logger.log("info", f"Connection error: {e}. Retrying in {_delay} seconds...")
                    await asyncio.sleep(_delay)
                    _retries -= 1
                    _delay *= backoff
            raise ConnectionError(f"Failed after {retries} retries")
        return wrapper
    return decorator


@dataclass
class ExecutorArg:
    """Serves as blueprint for how arguments should be structured for the `Executor`."""
    database_name: str
    collection_name: str
    path: str
    index: str
    num_candidates: Optional[int]
    limit: Optional[int]
    connec_client: Optional[MongoClient] = None

    def map_to_dict(self) -> dict[str, Any]:
        return {
            'database_name': self.database_name,
            'collection_name': self.collection_name,
            'path' : self.path,
            'index': self.index,
            'num_candidates': self.num_candidates,
            'limit': self.limit,
            'connec_client': self.connec_client
        }
    
    def  __call__(self) -> Dict[str, str | int] :
        return self.map_to_dict()
    
@timer(logger)   
def flatten_list(l: List[List[Any]]) -> List[Any]:
    if all(isinstance(item, list) for item in l):
        return [item for sublist in l for item in sublist]
    else:
        raise TypeError("Input must be a list of lists. All items in the list must be of the same type.")
    
@async_retry_on_connection_error([
    ConnectionErrors.CONNECTION_FAILURE, 
    ConnectionErrors.SERVER_SELECTION_TIMEOUT, 
    ConnectionErrors.CONNECTION_ERROR], 
    retries=3, delay=2, backoff=2)
async def _on_query(client: MongoClient, query: str) -> List[str]:
    """
    Handles a query by embedding the query and performing a vector search.

    Args:
        query (str): The query to be embedded and searched.
        num_candidates (int, optional): The number of candidates to return. Defaults to _DEFAULT_NUM_CANDIDATES.
        limit_per_group (int, optional): The maximum number of results to return per group. Defaults to _DEFAULT_LIMIT.

    Returns:
        List[str]: A list of strings representing the context of the query.
    """
    ctx = []

    @async_timer(logger)
    async def embed_query(query: str) -> Any:
        try:
            embedder = VectorEmbeddingManager()
            return await embedder.request(query)
        except Exception as e:
            logger.log("error", "Embedding error.", e)
            raise
        finally:
            await embedder.close()

    @async_timer(logger)
    async def embedding_callback(embedding: Any, *args) -> List[Any]:
        nonlocal ctx
        executor = Executor(*args)
        fields = {
            '_id': 1, 
            'content': 1, 
            'contentStr': 1, 
            'content_embedding': 1, 
            'name_embedding': 1, 
            'description_embedding': 1, 
            'price_embedding': 1
        }
        ctx = await executor.build_context(embedding, fields=fields)

        flatten_ctx = flatten_list(ctx)
        return flatten_ctx

    @timer(logger)
    def filter_search(query_embedding: Any, flatten_ctx: List[Any]) -> List[Any]:
        # instanciate the filter
        filter = Filter(
            query_embedding, 
            flatten_ctx, 
            threshold=SearchBalancer.THRESHOLD,
            batch_size=SearchBalancer.BATCH_SIZE    
        ) 
        # Filtering the context and flattening it
        final_ctx = flatten_list(list(filter()))
        # apply stop index according to context tokens limit 
        return final_ctx[:SearchBalancer.STOP_INDEX] 
    try:
        arg_1 = ExecutorArg(
            **SearchArgs.TICKERS.value,
            connec_client = client
        )()

        arg_2 = ExecutorArg(
            **SearchArgs.ARTICLES.value,
            connec_client = client
        )()
        # more...
        args = arg_1, arg_2, # more...
        # embed query here
        embedding = await embed_query(query)
        # on embedding callback
        ctx = await embedding_callback(embedding, *args)
        # on filter search
        final_ctx = filter_search(embedding, ctx)

        return final_ctx

    except Exception as e:
        logger.log("error", "Error while performing vector search", e)
        logger.log("warning", "Vector search aborted. Returning an empty list")
        raise

async def search(client: MongoClient, query: str) -> List[Any]:
    """
    Main access function of the vector search. Provides a high-level handling of the vector search.

    First, a list of `ExecutorArg` objects is created. Then, the vector search is performed by calling the
    `Executor` class with the list of ExecutorArg objects. 
    Finally, the search results are filtered to return the `n` most relevant results.

    Args:
        client (MongoClient): The MongoClient object.
        query (str): The query to be searched for. Since the search is vector based, the query is first embedded 
        so that the result of the search  can be similarity based.

    Returns:
        List[str]: A list of strings representing the result of the query.
    """
    try:
        return await _on_query(client, query)
    except Exception as e:
        logger.log("error", "Error while performing vector search", e)
        logger.log("warning", "Vector search aborted. Returning an empty list")
        raise

@async_timer(logger)
async def main(query: str) -> None: 
    # create client
    client = create_client()
    if not client:
        raise ValueError("No client found. Aborting...")

    ctx = await search(client, query)

    if isinstance (ctx, list):
        print("Context totat components: ", len(ctx))
        for c in ctx:
            print(f'{c["_id"]}: {c["score"]}')
    else:
        print("Warning! Returned value seems to be valid but is nor a list. [Did not return a list]")

if __name__ == "__main__":
    asyncio.run(main('is the apple stock going down? i heard it is!'))