import spacy
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from app.bigram_model import BigramModel

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned "
    "and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

# Load the spaCy model once when the server starts
nlp = spacy.load("en_core_web_md")


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word, request.length
    )
    return {"generated_text": generated_text}


@app.get("/embedding")
def get_embedding(
    word: str = Query(..., min_length=1, max_length=100)
):
    word = word.strip()
    doc = nlp.make_doc(word)

    # Require a single word rather than a sentence or punctuation
    if len(doc) != 1 or not doc[0].is_alpha:
        raise HTTPException(
            status_code=400,
            detail="Please enter a single word."
        )

    token = doc[0]

    if not token.has_vector:
        raise HTTPException(
            status_code=404,
            detail="No embedding is available for this word."
        )

    return {
        "word": token.text,
        "dimension": int(token.vector.size),
        "embedding": token.vector.tolist()
    }

from app.image_classifier import router as classifier_router
app.include_router(classifier_router)
