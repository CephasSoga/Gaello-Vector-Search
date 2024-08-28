import time
import aiohttp
import asyncio
import functools
from enum import Enum
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from builder.executor import Executor
from builder.embeddings import VectorEmbeddingManager
from builder.vector_serach import _DEFAULT_LIMIT, _DEFAULT_NUM_CANDIDATES
from utils_vector.logs import Logger

logger = Logger("Vector Search")


class ConnectionErrors(Enum):
    CONNECTION_FAILURE = ConnectionFailure
    SERVER_SELECTION_TIMEOUT = ServerSelectionTimeoutError
    REQUESTS_CONNECTION_ERROR = aiohttp.ClientConnectionError
    TIMEOUT = aiohttp.ClientTimeout
    OTHER = Exception

def async_retry_on_connection_error(errors, retries=3, delay=2, backoff=2):
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
    num_candidates: Optional[int] = _DEFAULT_NUM_CANDIDATES
    limit: Optional[int] = _DEFAULT_LIMIT

    def map_to_dict(self) -> dict[str, Any]:
        return {
            'database_name': self.database_name,
            'collection_name': self.collection_name,
            'path' : self.path,
            'index': self.index,
            'num_candidates': self.num_candidates,
            'limit': self.limit
        }
    
    def  __call__(self) -> Dict[str, str | int] :
        return self.map_to_dict()
    
@async_retry_on_connection_error([ConnectionErrors.CONNECTION_FAILURE, ConnectionErrors.SERVER_SELECTION_TIMEOUT, ConnectionErrors.REQUESTS_CONNECTION_ERROR, ConnectionErrors.TIMEOUT], retries=3, delay=2, backoff=2)
async def _on_query(query: str) -> List[str]:
    """
    Handles a query by embedding the query and performing a vector search.

    Args:
        query (str): The query to be embedded and searched.

    Returns:
        List[str]: A list of strings representing the context of the query.
    """
    arg_1 = ExecutorArg(
        database_name = 'market',
        collection_name = 'articles',
        path  = 'content_embedding',
        index = 'article_index',
        num_candidates = _DEFAULT_NUM_CANDIDATES,
        limit = _DEFAULT_LIMIT
    )()

    arg_2 = ExecutorArg(
        database_name = 'market',
        collection_name = 'ticker',
        path  = 'name_embedding',
        index = 'ticker_index',
        num_candidates = _DEFAULT_NUM_CANDIDATES,
        limit = _DEFAULT_LIMIT
    )()

    # more...

    args = arg_1, arg_2, # more...

    ctx = []

    _s0 = time.perf_counter()
    
    embedder = VectorEmbeddingManager()
    embedding = await embedder.request(query)

    _e0 = time.perf_counter()
    
    logger.log("info", f"Finished embedding query in {_e0 - _s0}s")

    _s1 = time.perf_counter()

    executor = Executor(*args)
    fields = {'_id': 0, 'content': 1, 'contentStr': 1}
    ctx = executor.build_context(embedding, fields=fields)
    
    _e1 = time.perf_counter()
    
    logger.log("info", f"Finished vector search in {_e1 - _s1}s")
    
    await embedder.close()

    return ctx
    


async def call(query: str) -> List[str]:
    """
    Main access function of the vector search. Handles a query by performing a vector search.

    Args:
        query (str): The query to be searched.

    Returns:
        List[str]: A list of strings representing the result of the query.
    """
    try:
        return await _on_query(query)
    except Exception as e:
        logger.log("error", "Error while performing vector search", e)
        logger.log("warning", "Vector search aborted. Returning an empty list")
        return []

async def main(): 
    ctx = await call('hello')

    if ctx:
        for c in ctx:
            print(c)

if __name__ == "__main__":
    asyncio.run(main())