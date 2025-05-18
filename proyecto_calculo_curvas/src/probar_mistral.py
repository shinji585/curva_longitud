import requests

prompt = "generame una lista de 10 números aleatorios entre 1 y 100"

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False  # IMPORTANTE para respuesta completa en un solo JSON
    }
)

data = response.json()
print(data["response"])
