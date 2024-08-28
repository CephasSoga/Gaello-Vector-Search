import os
from typing import Tuple, Dict, Any
import concurrent.futures as futures
from builder.vector_serach import VectorSearchManager


class ContextBuilder:
    """
    A class for building context by executing vector search on multiple targets concurrently.
    """
    def __init__(self, *args: Dict[str, Any], **kwargs):
        """
        Initializes the ContextBuilder class.

        Args:
            *args: Variable length argument list containing target parameters for vector search.
            **kwargs: Additional keyword arguments for the ThreadPoolExecutor.
        """
        self.max_workers = min((os.cpu_count() or 4) * 2, len(args))
        self.executor = futures.ThreadPoolExecutor(max_workers=self.max_workers, **kwargs)
        self.futures = []
        self.results = []
        self.targets = args

    def build(self, embedding, fields: Dict[str, Any] = {}):
        """
    	Builds the context by executing vector search on multiple targets concurrently.

    	Args:
    		embedding: The input embedding to be used for vector search.
    		fields (Dict[str, Any], optional): The fields to be projected in the search results. Defaults to an empty Dict.

    	Returns:
    		List[Any]: The combined search results from all targets.
    	"""
        with self.executor as executor:
            for target in self.targets:
                future = executor.submit(self.vector_search_on_target, embedding, fields, target)
                self.futures.append(future)
            for future in futures.as_completed(self.futures):
                self.results.extend(future.result())

        return self.results

    def vector_search_on_target(self, embedding, fields: Dict[str, Any] = {}, target: Dict[str, Any] = {}):
        """
    	Executes a vector search on a target.

    	Args:
    		embedding: The input embedding to be used for vector search.
    		fields (Dict[str, Any], optional): The fields to be projected in the search results. Defaults to an empty Dict.
    		target (Dict[str, Any], optional): The target parameters for vector search. Defaults to an empty Dict.

    	Returns:
    		The search results from the target.
    	"""
        v = VectorSearchManager(**target)
        return v.request(embedding, **fields)

