import wikipedia
from typing import Tuple
import logging
import numpy as np
import spacy
import nltk
from nltk.util import ngrams
import os
import re

# Silence the GuessedAtParserWarning
import warnings
from bs4 import GuessedAtParserWarning
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
wikipedia.set_lang("en")

# For reformulate the question
nlp = spacy.load("en_core_web_sm")

def search_wiki(search_object: str) -> list[str]:
    """
    Search for Wikipedia article and return the article.

    Args:
        item: The item to search for

    Returns:
        list[str]: The article
    """
    documents = []
    try:
        results = wikipedia.search(search_object, results = 5)
    except Exception:
        # logging.error(f"Error: {e}")
        return documents
    logging.info(f"Successfully Search the Number of item: {results}")
    for result in results:
        try:
            documents.append(wikipedia.summary(result))
        except Exception:
            # logging.error(f"It wikipedia not our fault: \nError: {e}")
            continue
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
        doc = nlp(doc)
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        cleaned_docs.append(" ".join(filtered_tokens))
    return cleaned_docs

def prep_question(question: str) -> Tuple[str, str, int]:
    """
    Preprocess the question by removing the question type and transform it to the query for answer and objects to search.

    Args:
        question: The question to preprocess

    Returns:
        Tuple[str, str, int]: The query, the object to search, and the question type
    """
    doc = nlp(question)
    if doc[-1].pos_ == "VERB":
        pattern = r"\b(is|are|was|were)\b"
        match = re.search(pattern, question)
        pronoun = " ".join([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]])
        verb = [token.text for token in doc if token.pos_ == "VERB"]
        if question.lower().startswith("who"):
            _ = 1
        elif question.lower().startswith("what"):
            _ = 2
        elif question.lower().startswith("where"):
            _ = 3
        elif question.lower().startswith("when"):
            _ = 4
        return f"{pronoun} {match.group(0)} {verb[0]} in", pronoun, _

    elif question.lower().startswith("who"):
        pattern = r"(?i)who\s+(is|was|are|were)\s+(.*)"
        match = re.search(pattern, question)
        return match.group(2) + " " + match.group(1), match.group(2), 1

    elif question.lower().startswith("what"):
        pattern = r"(?i)what\s+(is|was|are|were)\s+(.*)"
        match = re.search(pattern, question)
        return match.group(2) + " " + match.group(1), match.group(2), 2

    elif question.lower().startswith("where"):
        pattern = r"(?i)where\s+(is|was|are|were)\s+(.*)"
        match = re.search(pattern, question)
        return match.group(2) + " " + match.group(1) + " located in", match.group(2), 3

    elif question.lower().startswith("when"):
        pattern = r"(?i)when\s+(is|was|are|were)\s+(.*)"
        match = re.search(pattern, question)
        return match.group(2) + " " + match.group(1) + " happen in", match.group(2), 4

    else:
        print("Invalid Question")
        return None, None, None

def normalize_tfidf(tfidf_matrix):
    """
    Normalize each document vector (row) in the TF-IDF matrix using L2 norm.
    
    Parameters:
    - tfidf_matrix (numpy.ndarray): 2D array of shape (n_documents, n_terms)
    
    Returns:
    - numpy.ndarray: The normalized TF-IDF matrix
    """
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    
    norms[norms == 0] = 1
    
    normalized_matrix = tfidf_matrix / norms
    return normalized_matrix

def tfidf(documents: list[str], query: str, normalization = False) -> np.ndarray:
    """
    Calculate the tfidf for the documents and the query.

    Args:
        documents: List of documents
        query: The query

    Returns:
        np.ndarray: The tfidf matrix for the documents
        np.ndarray: The tfidf vector for the query
    """
    unique_words = {word for doc in documents for word in doc.split()}
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    tf = np.zeros((len(documents), len(unique_words)))
    for idx, doc in enumerate(documents):
        for word in doc.split():
            tf[idx, word_to_idx[word]] += 1
    df = (tf != 0).sum(axis=0)
    idf = np.log(len(documents) / df) + 1
    tfidf = tf * idf
    query_tf = np.zeros((1, len(unique_words)))
    for word in data_cleaning([query])[0].split():
        if word in word_to_idx:
            query_tf[0, word_to_idx[word]] += 1
    query_vector = query_tf * idf
    if normalization:
        tfidf = normalize_tfidf(tfidf)
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
    n_grams = ngrams(nltk.word_tokenize(text.lower()), n)
    return [' '.join(grams) for grams in n_grams]

def tile_ngrams(ngram1: list[str], ngram2: list[str]) -> list[str]:
    # Handle containment cases
    if all(x in ngram2 for x in ngram1):
        return ngram2
    if all(x in ngram1 for x in ngram2):
        return ngram1

    # Check both tiling directions
    def get_tiled(n1, n2):
        for i in range(1, min(len(n1), len(n2)) + 1):
            if n1[-i:] == n2[:i]:
                return n1 + n2[i:]
        return None

    # Try both tiling orders
    tiled1 = get_tiled(ngram1, ngram2)
    tiled2 = get_tiled(ngram2, ngram1)

    # Return the first valid tiling, or None if neither works
    return tiled1 or tiled2

def n_grams_filter(documents: list[str], question_type: int) -> list[str]:
    ngrams = [] # List of ngrams
    scores = [] # List of ngram scores

    # Generate all unigrams, bigrams and trigrams and count frequency in documents
    for n in range(1, 4):
        for document in documents:
            n_grams = generate_ngrams(document, n)

            # Find all distinct ngrams from the search document
            distinct_ngrams = []
            for n_gram in n_grams:
                if n_gram not in distinct_ngrams:
                    distinct_ngrams.append(n_gram)
            
            # Add all distinct ngrams scores to the result, increasing score by only 1 per document
            # Scores are also adjusted by positional bias, meaning ngrams appearing earlier in
            # the document are scored higher. Also score longer ngrams higher
            max_score = len(distinct_ngrams)
            for idx, n_gram in enumerate(distinct_ngrams):
                score = (max_score - idx) / max_score * n
                if n_gram not in ngrams:
                    ngrams.append(n_gram)
                    scores.append(score)
                else:
                    scores[ngrams.index(n_gram)] += score

    # Re-score ngrams based on question type
    question_bias = 1.5
    """
    Question types:
    - 1: WHO
    - 2: WHAT
    - 3: WHERE
    - 4: WHEN
    """
    for idx, ngram in enumerate(ngrams):
        token = nlp(ngram)
        for ent in token.ents:
            if question_type == 1:
                if ent.label_ in ['PERSON', 'ORG']:
                    scores[idx] *= question_bias
                    logging.info(f"{ent.text} {ent.label_}")
            elif question_type == 2:
                if ent.label_ in ['NORP', 'FAC', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                    scores[idx] *= question_bias
            elif question_type == 3:
                if ent.label_ in ['LOC', 'GPE']:
                    scores[idx] *= question_bias
            elif question_type == 4:
                if ent.label_ in ['DATE', 'TIME', 'CARDINAL']:
                    scores[idx] *= question_bias
    # Tile ngrams until it's not possible anymore
    ngrams = [ngram.split() for ngram in ngrams]
    
    # Function to remove tiled ngrams from both lists
    def remove_ngram(ngram):
        scores.remove(scores[ngrams.index(ngram)])
        ngrams.remove(ngram)

    ans_score = max(scores) # Find the ngram with the highest score as a starting point
    ans = ngrams[scores.index(ans_score)]
    sep = " "
    logging.info(f"Starting with ngram \"{sep.join(ans)}\" with score {ans_score}")
    remove_ngram(ans)
    tile = True
    while tile:
        tile = False
        max_score = max(scores) # Find the ngram with the highest score as a starting point
        max_ngram = ngrams[scores.index(max_score)]
        tile_result = tile_ngrams(max_ngram, ans)
        if tile_result:
            logging.info(f"Tiled \"{sep.join(max_ngram)}\" to \"{sep.join(ans)}\" to form\n\"{sep.join(tile_result)}\"")
            tile = True
            ans = tile_result
            ans_score += max_score
        remove_ngram(max_ngram)

    return ans

def log_write(file, text, way = "a"):
    """
    Write to the log file.

    Args:
        file: The file to write to
        text: The text to write
    """
    with open(file, way) as f:
        f.write(text)

def main():
    LOG_FILE = "qa_log.txt"
    if not os.path.exists(LOG_FILE):
        log_write(LOG_FILE, "This is a QA system by Group 5. It will try to answer questions that start with Who, What, When or Where.\nEnter 'exit' to leave the program.\n", way = "w")
    
    print("This is a QA system by Group 5. It will try to answer questions that start with Who, What, When or Where.\nEnter 'exit' to leave the program.")

    while True:
        question = input("Please enter a question: ").replace("?", "")
        log_write(LOG_FILE, f"Question: {question}\n")
        if question == "exit":
            print("Goodbye!")
            log_write(LOG_FILE, "Response: Goodbye!\n")
            break

        query, search_object, type = prep_question(question)
        query = query.lower()

        log_write(LOG_FILE, f"Search_Object: {search_object}\n")
        logging.info(f"Answer format: {query}, Search Object: {search_object}")

        documents = data_cleaning(search_wiki(search_object))
        log_write(LOG_FILE, f"Documents: {documents}\n")

        if len(documents) == 0:
            print("I am sorry, I don't know the answer.")
            log_write(LOG_FILE, "Response: I am sorry, I don't know the answer.\n")
            continue
        logging.info(f"Successfully Scrape the Number of Articles: {len(documents)}")
        documents_vector, query_vector = tfidf(documents, query, normalization=True)
        similarity = [cosine_similarity(query_vector, doc) for doc in documents_vector]
        top_k_indices = top_k_scores(similarity, k = 3)
        select_docs = [documents[idx] for idx in top_k_indices]
        answer = n_grams_filter(select_docs, type)
        logging.info(f"Answer: {answer}")
        result = []
        for sen in answer:
            for word in sen.split():
                if word not in result:
                    result.append(word)
        if len(result) > 0:
            log_write(LOG_FILE, f"Response: {query} {' '.join(result)}\n")
            print(f"{query[0].upper()}{query[1:]} {' '.join(result)}")
        else:
            log_write(LOG_FILE, "Response: I am sorry, I don't know the answer.\n")
            print("I am sorry, I don't know the answer.")


if __name__ == "__main__":
    main()