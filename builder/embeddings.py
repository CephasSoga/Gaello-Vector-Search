import aiohttp

from utils_vector.envhandler import get_env

class VectorEmbeddingManager:
    """
    A class for managing vector embedding requests using the OpenAI API.
    """
    def __init__(self):
        """
        Initializes the VectorEmbeddingManager class.
        """ 
        self.api_url = 'https://api.openai.com/v1/embeddings'
        self.apikey = get_env('OPENAI_API_KEY')
        self.session = None

    def create_session(self):
        """
        Creates an aiohttp ClientSession for making API requests.
        """
        self.session = aiohttp.ClientSession()


    async def request(self, query: str):
        """
        Makes an asynchronous request to the OpenAI API to get the vector embedding for the given query.

        Args:
            query (str): The input query for which the embedding needs to be obtained.

        Returns:
            list: The vector embedding for the input query.

        Raises:
            Exception: If the API request fails.
        """
        # Call OpenAI API to get the embeddings.
        headers = {
            'Authorization': f'Bearer {self.apikey}',
            'Content-Type': 'application/json'
        }
        payload = {
            'input': query,
            'model': 'text-embedding-ada-002'
        }

        if not self.session:
            self.create_session()

        async with self.session.post(self.api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data['data'][0]['embedding']
            else:
                raise Exception(f"Failed to get embedding. Status code: {response.status}")
            

    async def close(self):
        """
        Closes the aiohttp ClientSession when done with the API requests.
        """
        if self.session:
            await self.session.close()  # Properly close the session when done


