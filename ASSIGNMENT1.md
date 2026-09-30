# Assignment 1: Word Embedding API

This project extends the Module 3 classroom FastAPI application
with a spaCy word embedding endpoint.

Based on the course repository:
https://github.com/mmerler/applied_genai_notebooks

## Requirements

Install and start Docker Desktop.

## Run with Docker

Clone the repository and enter its directory:

```bash
git clone https://github.com/SeVen77-cyq/APAN5560-applied-genai-projects.git
cd APAN5560-applied-genai-projects
```

Build and run the API:

```bash
docker build -t genai-assignment1 .
docker run --rm -p 8000:8000 genai-assignment1
```

The Docker image installs the dependencies and the spaCy
en_core_web_md model automatically.

## Query the embedding endpoint

Open this URL in a browser:

http://127.0.0.1:8000/embedding?word=apple

Or use curl:

```bash
curl "http://127.0.0.1:8000/embedding?word=apple"
```

The JSON response includes:
- word: the queried word
- dimension: 300
- embedding: the list of 300 vector values

The endpoint accepts one alphabetic word.
Invalid input returns HTTP 400.
Words without an available vector return HTTP 404.

## Interactive API documentation

Visit http://127.0.0.1:8000/docs while the server is running.

The original POST /generate endpoint is also available.

## Stop the server

Press Ctrl+C in the terminal running the container.
