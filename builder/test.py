import requests

def request(query='hello'):
        # Define the OpenAI API url and key.
    url = 'https://api.openai.com/v1/embeddings'
    openai_key = 'sk-proj-QyWSLHFDxXFRXNiUuTFtT3BlbkFJJiA8SyBqwjGX4TsZGKXj'  # Replace with your OpenAI key.
    
    # Call OpenAI API to get the embeddings.
    headers = {
        'Authorization': f'Bearer {openai_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'input': query,
        'model': 'text-embedding-ada-002'
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    else:
        raise Exception(f"Failed to get embedding. Status code: {response.status_code}")
    

if __name__ == "__main__":
    import time
    s = time.perf_counter()
    r=request()
    e = time.perf_counter()
    print("Time taken:", e-s)