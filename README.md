# AIT 526

### We directly scrape the waht-question because wikipedia summary can't get the result by single object, such as: coffee, train or truck.
### We score the ngrams by it n, question-related type and the keywords.
### We get the highest score of ngrams and start to tile it with other ngrams until there is no ngrams can tile with it. 
### We only tile the reight side of the highest score ngram.

```python
# Test questioins
questions = [
    "What is coffee",
    "What is computer",
    "What is train",
    "What is Google",
    "What is NBA",
    "Who is Donald Trump",
    "Who is Barack Obama",
    "Who is Gordon James Ramsay",
    "Who is Kobe Bean Bryant",
    "Who is Giannis Sina Ugo Antetokounmpo",
    "Where is George Mason University",
    "Where is Taiwan",
    "Where is Japan",
    "Where is Fairfax city",
    "Where is The Metropolitan Museum of Art",
    "When was George Washington born",
    "When was the first iphone released",
    "When was the World War one",
    "When was Mahatma Gandhi assassinated",
    "When was the Titanic sink"
]
```

```bash
# Run test
python qa-system
# type test in the qa
Please enter a question: test
```