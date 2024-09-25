from typing import Dict, List, Any

from vector_search.builder.context import ContextBuilder

class Executor:
    """
    A class for executing vector search on multiple targets concurrently.
    """

    def __init__(self, *args: Dict[str, Any], **kwargs):
        """
        Initializes the Executor class.

        Args:
            *args: Variable length argument list containing target parameters for vector search.
            **kwargs: Additional keyword arguments for the ThreadPoolExecutor.
        """
        self.builder = ContextBuilder(*args, **kwargs)


    async def build_context(self, embedding, fields: Dict[str, Any] = {}) -> List[List[Dict]]:
        """
        Builds the context by executing vector search on multiple targets concurrently.

        Args:
            embedding: The input embedding to be used for vector search.
            fields (Dict[str, Any] | None, optional): The fields to be projected in the search results. Defaults to None.

        Returns:
            List[Any]: The combined search results from all targets.
        """
        return await self.builder.build(embedding, fields)