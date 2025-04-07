import wikipedia
from typing import Tuple
import requests
from pyquery import PyQuery as pq
import logging
import numpy as np
import spacy
import nltk
from nltk.util import ngrams
import os
import re

LOG_FILE = "qa_log.txt"

# Silence the GuessedAtParserWarning
import warnings
from bs4 import GuessedAtParserWarning

warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
wikipedia.set_lang("en")

# For reformulate the question
nlp = spacy.load("en_core_web_sm")


def search_wiki(search_object: str, question_type: int) -> list[str]:
    """
    Search for Wikipedia article and return the article.

    Args:
        item: The item to search for

    Returns:
        list[str]: The article
    """
    documents = []
    try:
        results = wikipedia.search(search_object, results=1)
    except Exception:
        return documents
    for result in results:
        try:
            if question_type == 2:
                response = requests.get(f"https://en.wikipedia.org/wiki/{result}")
                doc = pq(response.text)
                document = doc("div.mw-content-ltr p").text().split(".")[0]
                documents.append(document)
                log_write(LOG_FILE, f"Wikipedia Summary: {documents[0]}\n")
            else:
                documents.append(wikipedia.summary(result, sentences=3))
                log_write(LOG_FILE, f"Wikipedia Summary: {documents[0]}\n")
        except wikipedia.PageError as e:
            print(f"\nPageError: {e}")
            print("\nChange to another search object.")
            documents.append(wikipedia.summary(search_object, sentences=3))
            log_write(LOG_FILE, f"Wikipedia Summary: {documents[0]}\n")
            continue
        except Exception:
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
        filtered_tokens = [
            token.text for token in doc if not token.is_stop and not token.is_punct
        ]
        cleaned_docs.append(" ".join(filtered_tokens))
    return cleaned_docs


def prep_question(question: str) -> Tuple[str, str, int]:
    """
    Preprocess the question by removing the question type and transform it to the query for answer and objects to search.

    Args:
        question: The question to preprocess

    Returns:
        Tuple[str, str]: The query and the object to search
    """
    doc = nlp(question)
    if doc[-1].pos_ == "VERB" and not question.lower().startswith("what"):
        pattern = r"\b(is|are|was|were)\b"
        match = re.search(pattern, question)
        pronoun = " ".join(
            [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        )
        verb = [token.text for token in doc if token.pos_ == "VERB"]
        if question.lower().startswith("who"):
            _ = 1
        elif question.lower().startswith("what"):
            _ = 2
        elif question.lower().startswith("where"):
            _ = 3
        elif question.lower().startswith("when"):
            _ = 4
        return f"{pronoun} {match.group(0)} {verb[0]}", pronoun, _

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
        return match.group(2) + " " + match.group(1) + " located", match.group(2), 3

    elif question.lower().startswith("when"):
        pattern = r"(?i)when\s+(is|was|are|were)\s+(.*)"
        match = re.search(pattern, question)
        return match.group(2) + " " + match.group(1) + " happen in", match.group(2), 4

    else:
        print("Invalid Question")
        return None, None, None


def generate_ngrams(text, n) -> list[str]:
    n_grams = ngrams(nltk.word_tokenize(text), n)
    return [" ".join(grams) for grams in n_grams]


def tile_ngrams(ngram1: list[str], ans_ngrams: list[str]):
    """
    Tile two lists of strings together if ngram2 can be appended to ngram1 based on an overlapping segment.

    Args:
        ngram1: First list of strings (left side, assumed to have the higher score)
        ngram2: Second list of strings (to be appended on the right)

    Returns:
        list[str]: Combined list if tiling is possible, None otherwise

    Examples:
        ['B', 'C', 'D'], ['A', 'B', 'C'], -> ['A', 'B', 'C', 'D']
        ['I', 'will', 'always', 'love'], ['love', 'you'] -> ['love', 'you']
    """
    # Handle containment cases
    if len(ngram1) == 0:
        return ans_ngrams.copy()
    if len(ans_ngrams) == 0:
        return ngram1.copy()

    if all(elem in ans_ngrams for elem in ngram1):
        return None

    # Check for overlapping segments (only on the right side)
    def find_overlap(list1, list2):
        # Try different overlap sizes, from largest to smallest
        for overlap_size in range(min(len(list1), len(list2)), 0, -1):
            # Check if the end of list1 matches the start of list2
            if list1[-overlap_size:] == list2[:overlap_size]:
                return list1 + list2[overlap_size:]
        return None

    result = find_overlap(ans_ngrams, ngram1)
    if result and len(result) > len(ans_ngrams):
        return result

    # No valid tiling extension found; return None to indicate that no tiling occurred
    return None


def n_grams_filter(documents: list[str], question_type: int, search_object: str) -> str:
    all_ngram_dict = {}

    # Generate all unigrams, bigrams and trigrams and count frequency in documents
    for n in range(2, 4):
        for document in documents:
            n_grams = generate_ngrams(document, n)
            for n_gram in n_grams:
                if n_gram not in all_ngram_dict.keys():
                    all_ngram_dict[n_gram] = n

    # Re-score ngrams based on question type
    question_bias = 3
    """
    Question types:
    - 1: WHO
    - 2: WHAT
    - 3: WHERE
    - 4: WHEN
    """
    all_ngrams = list(all_ngram_dict.keys())
    for ngram in all_ngrams:
        if question_type == 1:
            pattern = r"\b(is|was|were|are)\b"
            if bool(re.search(pattern, ngram, re.IGNORECASE)):
                all_ngram_dict[ngram] *= 5
            if ngram[0].isupper():
                all_ngram_dict[ngram] *= question_bias

        elif question_type == 2:
            pattern = r"^(is|was|were|are)\b"
            if bool(re.search(pattern, ngram, re.IGNORECASE)):
                all_ngram_dict[ngram] *= question_bias
            if search_object in ngram:
                all_ngram_dict[ngram] *= question_bias

        elif question_type == 3:
            pattern = r"^(located|nearby|near|locate|region|country|lies|between)\b"
            if bool(re.search(pattern, ngram.lower(), re.IGNORECASE)):
                all_ngram_dict[ngram] *= 8
            if ngram[0].isupper():
                all_ngram_dict[ngram] *= question_bias

        elif question_type == 4:
            pattern = r"^(born|happen|borned|happened|occurred|occur|january|february|march|april|may|june|july|august|september|october|november|december)\b"
            if bool(re.search(pattern, ngram.lower(), re.IGNORECASE)):
                all_ngram_dict[ngram] *= 8
            if bool(re.search(r"\d+", ngram.lower(), re.IGNORECASE)):
                all_ngram_dict[ngram] *= question_bias

    median_score = np.percentile(list(all_ngram_dict.values()), 50)
    # Filter out ngrams with a score below the median
    all_ngram_dict = {k: v for k, v in all_ngram_dict.items() if v >= median_score}

    all_ngrams_list = {ngram: ngram.split() for ngram in all_ngram_dict.keys()}

    # Function to remove ngram from dictionary
    def remove_ngram(ngram):
        del all_ngram_dict[ngram]

    ans_ngrams_dict = {}
    ans_ngrams_list = []

    # Find starting ngram with highest score
    first_ngram = max(all_ngram_dict.items(), key=lambda x: x[1])[0]
    first_ngram_score = all_ngram_dict[first_ngram]
    first_ngram_tokens = all_ngrams_list[first_ngram]
    sep = " "
    logging.info(f'Starting with ngram "{first_ngram}" with score {first_ngram_score}')

    ans_ngrams_dict[first_ngram] = first_ngram_score
    ans_ngrams_list.append(first_ngram_tokens)
    remove_ngram(first_ngram)

    pattern = r"\.\s*$"

    while True:
        for ngram, _ in all_ngram_dict.items():
            tile_result = tile_ngrams(all_ngrams_list[ngram], ans_ngrams_list[0])

            if tile_result:
                ans_ngrams_list.pop()
                ans_ngrams_list.append(tile_result)
                remove_ngram(ngram)

                # If there is a senetence end in a period stop tiling and return the result
                if bool(re.search(pattern, sep.join(tile_result))):
                    return sep.join(ans_ngrams_list[0])
                break

        if tile_result is None:
            # No more ngrams to tile, return the result
            return sep.join(ans_ngrams_list[0])

def answer(question: str):
    query, search_object, question_type = prep_question(question)
    query = query.lower()
    log_write(LOG_FILE, f"Search_Object: {search_object}\n")
    logging.info(f"Answer format: {query} Search Object: {search_object}")

    documents = data_cleaning(search_wiki(search_object, question_type))
    if len(documents) == 0:
        print("I am sorry, I don't know the answer.")
        log_write(LOG_FILE, "Response: I am sorry, I don't know the answer.\n\n")
        return

    answer = n_grams_filter(documents, question_type, search_object)
    
    final_answer = tile_ngrams(answer.lower().split(" "), query.split(" "))
    if final_answer:
        print(
            "-" * 100,
            "\n",
            "Answer:",
            " ".join(final_answer).capitalize(),
            "\n",
            "-" * 100,
        )
        log_write(LOG_FILE, f"Response: {' '.join(final_answer).capitalize()}\n\n")
    else:
        if search_object.lower() in answer.lower():
            search_object_length = len(search_object.split(" "))
            if answer.lower().split(" ")[:search_object_length] == search_object.lower().split(" "):
                answer = " ".join(answer.split()[search_object_length:])
        print(
            "-" * 100,
            "\n",
            "Answer:",
            f"{query.capitalize()} {answer}",
            "\n",
            "-" * 100,
        )
        log_write(
            LOG_FILE, f"Response: {query.capitalize()} {answer}\n\n"
        )

def log_write(file, text, way="a"):
    """
    Write to the log file.

    Args:
        file: The file to write to
        text: The text to write
    """
    with open(file, way) as f:
        f.write(text)


def main():
    if not os.path.exists(LOG_FILE):
        log_write(
            LOG_FILE,
            "This is a QA system by Group 5. It will try to answer questions that start with Who, What, When or Where.\nEnter 'exit' to leave the program.\n",
            way="w",
        )

    print(
        "This is a QA system by Group 5. It will try to answer questions that start with Who, What, When or Where.\nEnter 'exit' to leave the program."
    )
    while True:
        try:
            question = input("Please enter a question: ").replace("?", "")
            log_write(LOG_FILE, f"Question: {question}\n")
            if question == "exit":
                print("Goodbye!")
                log_write(LOG_FILE, "Response: Goodbye!\n\n")
                break
            elif question == "test":
                with open("test-questions.txt", "r") as file:
                    for line in file:
                        print("*** TEST QUESTION: ", line[:-1], " ***")
                        answer(line[:-1])

            answer(question)

        except Exception as e:
            logging.info(e)
            print("Please enter a valid question.")
            log_write(LOG_FILE, "Response: Please enter a valid question.\n\n")
            continue


if __name__ == "__main__":
    main()
