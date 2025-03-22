import requests
from pyquery import PyQuery as pq
import numpy as np
import spacy
import re


url = "https://en.wikipedia.org/wiki/"
# For reformulate the question
nlp = spacy.load("en_core_web_sm")

def search_wiki(item) -> list[str]:
    """Search for Wikipedia article and return the article"""
    response = requests.get(url + item)
    doc = pq(response.text)
    items = doc("#mw-content-text > div > p")
    documents = []
    for item in items.items():
        documents.append(item.text())
    return documents

def data_cleaning(documents: list[str]) -> list[str]:
    """Clean the data by removing special characters"""
    cleaned_docs = []
    for doc in documents:
        doc = nlp(doc.lower())
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        cleaned_docs.append(" ".join(filtered_tokens))
    return cleaned_docs

def prep_question(question: str) -> str:
    types = ["what", "where", "when", "who"]

    for t in types:
        pattern = fr"(?i){t}\s+(is|was|are|were)\s+(.*)"
        match = re.search(pattern, question)
        if match:
            doc = nlp(match.group(2))
            if doc[-1].dep_ == "ROOT":
                return doc[:-1].text + " " + match.group(1) + " " + doc[-1].text
            else:
                return match.group(2) + " " + match.group(1)

def tfidf(documents: list[str], query: str) -> list[str]:
    clean_docs = data_cleaning(documents)
    unique_words = {word for doc in clean_docs for word in doc.split()}
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    tf = np.zeros((len(clean_docs), len(unique_words)))
    for idx, doc in enumerate(clean_docs):
        for word in doc.split():
            tf[idx, word_to_idx[word]] += 1
    df = (tf != 0).sum(axis=0)
    idf = np.log(len(clean_docs) / df) + 1
    tfidf = tf * idf
    pass

def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.
    Cosine similarity measures how similar two vectors are by calculating the cosine of the angle between them.

    Args:
        vector_a: First vector (numpy array)
        vector_b: Second vector (numpy array)

    Returns:
        float: Similarity score between -1 and 1
               1: Vectors are identical
               0: Vectors are perpendicular
              -1: Vectors are opposite
    """

    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    similarity = dot_product / (magnitude_a * magnitude_b)

    return similarity

def main():
    print("This is a QA system by YourName. It will try to answer questions that start with Who, What, When or Where. Enter 'exit' to leave the program.")
    while True:
        question = input("Please enter a question: ").lower().replace("?", "")

        if question == "exit":
            print("Goodbye!")
            break
        query = prep_question(question)
        
    


if __name__ == "__main__":
    main()