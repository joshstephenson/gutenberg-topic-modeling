# Topic Modeling Literature
#### Topic modeling books from The Gutenberg Project
This project is an exploration of topic modeling with Latent Dirachlet Allocation for literature. 

## book_cleaner.py
A bulk package of Gutenberg books was obtained. Each book was an obscurely named text file, so the first step was to write `book_cleaner.py` to parse the books. Each file has inconsistent headers and footers containing attributions and transcription notes that needed to be stripped out. Due to the overwhelming inconsistency, some books will be skipped and successful parsing with dump the user in `vim` to edit any remaining text they don't want included in the topic modeling. Saved books will be placed in `./Books` subdirectory.

## book_modeler.py
This file will parse books saved to `./Books`. It will tokenize the text in one of 6 ways depending on how the `METHOD` constant is set. Each of these methods is added to the previous one.
- `basic`: Basic `NLTK` word tokenization.
- `trimmed`: Will trim the book text to the nearest sentence boundary after 1000 words. This is inspired by [Jockers and Mimno (2013)](https://www.sciencedirect.com/science/article/abs/pii/S0304422X13000673) who tested this on 19th century literature.
- `stopwords`: Will remove all stopwords in `NLTK` as well as some others that were identified during the project. It will also remove any roman numerals.
- `onlynouns`: Will remove all tokens that aren't nouns. Proper nouns are also removed.
- `lemmatize`: Will lemmatize tokens using `WordNetLemmatizer` from `nltk.stem.wordnet`.
- `bigrams`: Will use only bigrams. Warning, this needs a lot more work to be useful.

After running this, it will output a visualization from `PyLDAvis` with a name based on the chosen method. For example: `books-lemmatize-32.html` where 32 is the selected number of topics to find. This can be changed using the `NUM_TOPICS` constant. If you want to run a coherence plot, then set `FIND_COHERENCE` to `True`.

Example LDAvis:

<img width="1475" alt="image" src="https://github.com/joshstephenson/gutenberg-topic-modeling/assets/11002/7a468a1a-96d0-4f8d-87ab-6153956d15e2">
