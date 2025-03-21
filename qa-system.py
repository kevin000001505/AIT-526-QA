import requests
from pyquery import PyQuery as pq
import spacy
import re


url = "https://en.wikipedia.org/wiki/"
# For reformulate the question
nlp = spacy.load("en_core_web_sm")

def search_wiki(item) -> list:
    """Search for Wikipedia article and return the article"""
    response = requests.get(url + item)
    doc = pq(response.text)
    items = doc("#mw-content-text > div > p")
    documents = []
    for item in items.items():
        documents.append(item.text())
    return documents

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