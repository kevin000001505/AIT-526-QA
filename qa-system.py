import requests
from pyquery import PyQuery as pq
from typing import Tuple
import logging
import numpy as np
import spacy
import nltk
from nltk.util import ngrams
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

url = "https://en.wikipedia.org/wiki/"
# For reformulate the question
nlp = spacy.load("en_core_web_sm")

def search_wiki(item: str) -> list[str]:
    """
    Search for Wikipedia article and return the article.

    Args:
        item: The item to search for

    Returns:
        list[str]: The article
    """
    response = requests.get(url + item)
    doc = pq(response.text)
    items = doc("#mw-content-text > div > p")
    documents = []
    for item in items.items():
        documents.append(item.text())
    return documents

def data_cleaning(documents: list[str]) -> list[str]:
    """
    Clean the data by removing punctuation, stop words and converting to lowercase.

    Args:
        documents: List of documents to clean

    Returns:
        list[str]: List of cleaned documents
    """
    cleaned_docs = []
    for doc in documents:
        doc = nlp(doc.lower())
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        cleaned_docs.append(" ".join(filtered_tokens))
    return cleaned_docs

def prep_question(question: str) -> Tuple[str, str]:
    """
    Preprocess the question by removing the question type and transform it to the query for answer and objects to search.

    Args:
        question: The question to preprocess

    Returns:
        Tuple[str, str]: The query and the object to search
    """
    types = ["what", "where", "when", "who"]

    for t in types:
        pattern = fr"(?i){t}\s+(is|was|are|were)\s+(.*)"
        match = re.search(pattern, question)
        if match:
            doc = nlp(question)
            if doc[-1].dep_ == "ROOT" and doc[-1].pos_ == "VERB":
                pronoun = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
                verb = [token.text for token in doc if token.pos_ == "VERB"]
                return f"{pronoun[0]} {match.group(1)} {verb[0]}", match.group(2)
            else:
                return match.group(2) + " " + match.group(1), match.group(2)

def tfidf(documents: list[str], query: str) -> np.ndarray:
    """
    Calculate the tfidf for the documents and the query.

    Args:
        documents: List of documents
        query: The query

    Returns:
        np.ndarray: The tfidf matrix for the documents
        np.ndarray: The tfidf vector for the query
    """
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
    query_vector = np.zeros((1, len(unique_words)))
    for word in data_cleaning([query])[0].split():
        if word in word_to_idx:
            query_vector[0, word_to_idx[word]] += 1
    return tfidf, query_vector

def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
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
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    similarity = dot_product / (magnitude_a * magnitude_b)

    return similarity

def top_k_scores(similarity: list[float], k: int) -> list[int]:
    """
    Return the top k scores.

    Args:
        similarity: List of similarity scores
        k: The number of top scores to return

    Returns:
        list[int]: The indices of the top k scores
    """
    flat_scores = np.array([s[0] if s != 0 else 0 for s in similarity])
    sorted_indices = np.argsort(flat_scores)[::-1]
    top_k_indices = sorted_indices[:k]
    return top_k_indices


def generate_ngrams(text, n) -> list[str]:
    clean_text = re.sub(r'\([^)]*\)', '', text).strip()
    n_grams = ngrams(nltk.word_tokenize(clean_text.lower()), n)
    return [ ' '.join(grams) for grams in n_grams]

def n_grams_filter(documents: list[str], query: str) -> list[str]:
    n = len(query.split())
    ans = []
    while n > 0:
        for document in documents:
            n_grams = generate_ngrams(document, n)
            if query.lower() in n_grams:
                for idx, token in enumerate(n_grams):
                    if token == query.lower():
                        ans.append(n_grams[idx+n])
                        ans.append(n_grams[idx+2*n])
                        # ans.append(n_grams[idx+3*n])
                        return ans
        n -= 1
        query = " ".join(query.split()[-n:])
    return ans


def main():
    print("This is a QA system by YourName. It will try to answer questions that start with Who, What, When or Where. Enter 'exit' to leave the program.")
    while True:
        question = input("Please enter a question: ").replace("?", "")

        if question == "exit":
            print("Goodbye!")
            break
        query, search_object = prep_question(question)
        logging.info(f"Answer format: {query}, Search Object: {search_object}")
        documents = search_wiki(search_object)
        logging.info(f"Successfully Scrape the Number of Articles: {len(documents)}")
        documents_vector, query_vector = tfidf(documents, query)
        similarity = [cosine_similarity(query_vector, doc) for doc in documents_vector]
        top_k_indices = top_k_scores(similarity, k = 5)
        logging.info(f"Select the Top 5 Articles: {top_k_indices}")
        select_docs = [documents[idx] for idx in top_k_indices]
        answer = n_grams_filter(select_docs, query)
        result = []
        for sen in answer:
            for word in sen.split():
                if word not in result:
                    result.append(word)
        print(query + " " + " ".join(result))


if __name__ == "__main__":
    main()