import asyncio
import numpy as np
import concurrent.futures as cf
from dataclasses import dataclass
from typing import Dict, Any, List, Union

from sklearn.metrics.pairwise import cosine_similarity

from vector_search.builder.serach import VectorSearchManager


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


@dataclass
class Filter:
    query_embedding: Union[List, np.ndarray]
    ctx_items: List[Dict]
    threshold: float
    batch_size: int

    def process_batch(self, query_embedding: Union[List, np.ndarray], ctx_items: List[Dict], threshold: float) -> List[Dict]:
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
                
            ctx_embeddings = [
                np.array(
                    item.get("content_embedding") 
                    or item.get("name_embedding") 
                    or item.get("description_embedding") 
                    or item.get("price_embedding")
                ) for item in ctx_items
            ]
            
            similarities = cosine_similarity(ctx_embeddings, query_embedding.reshape(1, -1)).flatten()

            # Filter out the context items with similarity below the threshold
            return [
                    {
                        "_id": item.get("_id"),
                        "description": item.get("description"),
                        "name": item.get("name"),
                        "price": item.get("price"),
                        "content": item.get("content") or item.get("contentStr"),
                        "score": similarity
                    } for item, similarity in zip(ctx_items, similarities) 
                    if similarity >= threshold
                ]
    
    def __call__(self):
        with cf.ThreadPoolExecutor() as executor:
            futures = []
            # Iterate over the context items in batches
            for batch in range(0, len(self.ctx_items), self.batch_size):
                batch_items = self.ctx_items[batch:batch + self.batch_size]
                # Submit each batch to the executor
                future = executor.submit(self.process_batch, self.query_embedding, batch_items, self.threshold)
                futures.append(future)

            # Yield the results once futures complete
            for future in cf.as_completed(futures):
                yield future.result()