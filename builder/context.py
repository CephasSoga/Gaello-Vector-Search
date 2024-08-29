import asyncio
from typing import Dict, Any, List

from builder.vector_serach import VectorSearchManager


class ContextBuilder:
    """
    A class for building context by executing vector search on multiple targets concurrently.
    """
    def __init__(self, *args: Dict[str, Any]):
        self.targets = args

    async def build(self, embedding, fields: Dict[str, Any] = {}) -> List[List[Dict]]:
        tasks = []
        for target in self.targets:
            task = asyncio.create_task(self.vector_search_on_target(embedding, fields, target))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

    
    async def vector_search_on_target(self, embedding, fields: Dict[str, Any] = {}, target: Dict[str, Any] = {}) -> List[Dict]:    
        v = VectorSearchManager(**target)
        return  await v.request(embedding, **fields)

