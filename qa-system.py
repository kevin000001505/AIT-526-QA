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

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
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
        results = wikipedia.search(search_object, results=1)
    except Exception:
        return documents
    for result in results:
        try:
            documents.append(wikipedia.summary(result, sentences=3))
        except wikipedia.PageError as e:
            print(f"\nPageError: {e}")
            print("\nChange another search object.")
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
        Tuple[str, str, int]: The query, the object to search, and the question type
    """
    doc = nlp(question)
    if doc[-1].pos_ == "VERB":
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


def tile_ngrams(ngram1: list[str], ans_ngrams: list[str]) -> list[str]:
    """
    Tile two lists of strings together if ngram2 can be appended to ngram1 based on an overlapping segment.

    Args:
        ngram1: First list of strings (left side, assumed to have the higher score)
        ngram2: Second list of strings (to be appended on the right)

    Returns:
        list[str]: Combined list if tiling is possible, None otherwise

    Examples:
        ['A', 'B', 'C'], ['B', 'C', 'D'] -> ['A', 'B', 'C', 'D']
        ['love', 'you'], ['I', 'will', 'always', 'love'] -> None  # since tiling is not on the right
    """
    # Handle containment cases
    if len(ngram1) == 0:
        return ans_ngrams.copy()
    if len(ans_ngrams) == 0:
        return ngram1.copy()

    # Check for overlapping segments (only on the right side)
    def find_overlap(list1, list2):
        # Try different overlap sizes, from largest to smallest
        for overlap_size in range(min(len(list1), len(list2)), 0, -1):
            # Check if the end of list1 matches the start of list2
            if list1[-overlap_size:] == list2[:overlap_size]:
                return list1 + list2[overlap_size:]
        return None

    # Only attempt tiling with ans_ngrams (the higher-score ngram) as the left side and ngram1 as the candidate to append
    result = find_overlap(ans_ngrams, ngram1)
    if result and len(result) > len(ans_ngrams):
        return result

    # No valid tiling extension found; return None to indicate that no tiling occurred
    return None


def n_grams_filter(
    documents: list[str], question_type: int, search_object: str
) -> list[str]:
    ngram_dict = {}

    # Generate all unigrams, bigrams and trigrams and count frequency in documents
    for n in range(2, 5):
        for document in documents:
            n_grams = generate_ngrams(document, n)
            for n_gram in n_grams:
                if n_gram not in ngram_dict.keys():
                    ngram_dict[n_gram] = n

    # Re-score ngrams based on question type
    question_bias = 3
    """
    Question types:
    - 1: WHO
    - 2: WHAT
    - 3: WHERE
    - 4: WHEN
    """
    all_ngrams = list(ngram_dict.keys())
    for ngram in all_ngrams:
        if question_type == 1:
            pattern = r"\b(is|was|were|are)\b"
            if bool(re.search(pattern, ngram, re.IGNORECASE)):
                ngram_dict[ngram] *= 3
            if ngram[0].isupper():
                ngram_dict[ngram] *= question_bias

        elif question_type == 2:
            pattern = r"^(is|was|were|are)\b"
            if bool(re.search(pattern, ngram, re.IGNORECASE)):
                ngram_dict[ngram] *= question_bias

        elif question_type == 3:
            pattern = r"^(located|nearby|near|locate|region|country|lies|between)\b"
            if bool(re.search(pattern, ngram.lower(), re.IGNORECASE)):
                ngram_dict[ngram] *= 8
            if ngram[0].isupper():
                ngram_dict[ngram] *= question_bias

        elif question_type == 4:
            pattern = r"^(born|happen|borned|happened|occurred|occur|january|february|march|april|may|june|july|august|september|october|november|december)\b"
            if bool(re.search(pattern, ngram.lower(), re.IGNORECASE)):
                ngram_dict[ngram] *= 8
            if bool(re.search(r'\d+', ngram.lower(), re.IGNORECASE)):
                ngram_dict[ngram] *= question_bias

    median_score = np.percentile(list(ngram_dict.values()), 10)
    filter_dict = {k: v for k, v in ngram_dict.items() if v >= median_score}
    ngram_dict = filter_dict

    ngrams_list = {ngram: ngram.split() for ngram in ngram_dict.keys()}

    # Function to remove ngram from dictionary
    def remove_ngram(ngram):
        del ngram_dict[ngram]

    ans_ngrams_dict = {}
    ans_ngrams_list = []

    # Find starting ngram with highest score
    first_ngram = max(ngram_dict.items(), key=lambda x: x[1])[0]
    first_ngram_score = ngram_dict[first_ngram]
    first_ngram_tokens = ngrams_list[first_ngram]
    sep = " "
    logging.info(f'Starting with ngram "{first_ngram}" with score {first_ngram_score}')
    ans_ngrams_dict[first_ngram] = first_ngram_score
    ans_ngrams_list.append(first_ngram_tokens)
    remove_ngram(first_ngram)

    # while tile and ngram_dict:
    tile_result = True
    while ngram_dict:
        skip = False
        # Find ngram with highest score
        max_ngram = max(ngram_dict.items(), key=lambda x: x[1])[0]
        max_ngram_score = ngram_dict[max_ngram]
        max_ngram_tokens = ngrams_list[max_ngram]

        k = 0
        ans_ngrams_list = sorted(ans_ngrams_list, key=lambda x: ans_ngrams_dict[" ".join(x)], reverse=True)
        while k < len(ans_ngrams_list):
            # Check if ngram is a subset of ans_list[k]
            if all(elem in ans_ngrams_list[k] for elem in max_ngram_tokens):
                skip = True
                remove_ngram(max_ngram)
                break
            else:
                tile_result = tile_ngrams(max_ngram_tokens, ans_ngrams_list[k])
                if tile_result:
                    logging.info(
                        f'Tiling "{max_ngram}" with "{sep.join(ans_ngrams_list[k])}" to get "{sep.join(tile_result)}"'
                    )
                    ans_ngrams_list.append(tile_result)

                    ans_ngrams_dict[sep.join(tile_result)] = (
                        max_ngram_score + ans_ngrams_dict[sep.join(ans_ngrams_list[k])]
                    )

                    del ans_ngrams_dict[sep.join(ans_ngrams_list[k])]
                    ans_ngrams_list.remove(ans_ngrams_list[k])
                    remove_ngram(max_ngram)
                    k = len(ans_ngrams_list)

                else:
                    k += 1

        if not tile_result and not skip:
            ans_ngrams_list.append(max_ngram_tokens)
            ans_ngrams_dict[max_ngram] = max_ngram_score

            remove_ngram(max_ngram)
    # print("\nAnswer Dictionary:", ans_dict)
    # print('--'*50)
    # print("\nAnswer:", max(ans_dict, key = ans_dict.get))
    raw_ans = max(ans_ngrams_dict, key=ans_ngrams_dict.get)

    return raw_ans


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
    LOG_FILE = "qa_log.txt"
    if not os.path.exists(LOG_FILE):
        log_write(
            LOG_FILE,
            "This is a QA system by Group 5. It will try to answer questions that start with Who, What, When or Where.\nEnter 'exit' to leave the program.\n",
            way="w",
        )

    print(
        "This is a QA system by Group 5. It will try to answer questions that start with Who, What, When or Where.\nEnter 'exit' to leave the program."
    )
    questions = [
        "When was George Washington born",
        "Where is George Mason University",
        "Where is Taiwan",
        "Where is Japan",
        "Who is Donald Trump",
        "Who is the first president of the United States",
        "Who is Barack Obama",
        "When was the first iPhone released",
    ]
    # while True:
    for question in questions:
        # question = input("Please enter a question: ").replace("?", "")
        log_write(LOG_FILE, f"Question: {question}\n")
        if question == "exit":
            print("Goodbye!")
            log_write(LOG_FILE, "Response: Goodbye!\n")
            break

        query, search_object, question_type = prep_question(question)
        query = query.lower()

        log_write(LOG_FILE, f"Search_Object: {search_object}\n")
        logging.info(f"Answer format: {query}, Search Object: {search_object}")
        documents = data_cleaning(search_wiki(search_object))

        if len(documents) == 0:
            print("I am sorry, I don't know the answer.")
            log_write(LOG_FILE, "Response: I am sorry, I don't know the answer.\n")
            continue
        answer = n_grams_filter(documents, question_type, search_object)
        answer = answer.replace(". ", "")
        logging.info(f"Answer: {answer}")
        Ans = tile_ngrams(answer.lower().split(" "), query.split(" "))
        if Ans:
            print("\nAnswer:", " ".join(Ans), "\n---------------")
        else:
            print(
                "\nAnswer:",
                f"{query[0].upper()}{query[1:]} {answer}",
                "\n",
                "---------------" * 10,
            )
        breakpoint()


if __name__ == "__main__":
    main()
