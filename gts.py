#!/usr/bin/env python3

from enum import IntEnum
import copy
import re
import os.path
from pathlib import Path
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag, FreqDist, ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import MWETokenizer
from nltk.lm.preprocessing import flatten
import json
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, Phrases, LdaModel
from gensim.utils import tokenize
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, stem_text
import pyLDAvis
import pyLDAvis.gensim
#import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle

def setup_mwes():
    mwe_body_lines = []
    with open('mwelex/wordnet_mwes.json', 'r') as f:
        mwe_lines = f.readlines()
    f.close()
    mwes = []
    mwe_lookup = {}
    for line in mwe_lines:
        js = json.loads(line)
        lemma = js['wnlemmas']
        words = js['words']
        key = list(lemma.keys())[0]
        supersense = lemma[key]['supersense'].strip()
        mwes.append(tuple(words))
        lookup_key = '_'.join(words)
        pos = supersense.split('.')[0].strip()
        mwe_lookup[lookup_key] = {'supersense': supersense, 'POS_GROUP': pos}

    return MWETokenizer(mwes)

stop = stopwords.words('english')
stop.extend(['thou', 'thing', 'hath', 'part', 'ye', 'thee', 'thy', 'ere', 'nay', 'hast', 'ha', 'wilt', 'thereafter', '....', 'en', 'dat', 'ai', 'gwine', 'aye', 'quoth', 'beheld', 'behold', 'thine', 'unto', 'chapter', 'book', 'th', 'yo', 'wi', 'de', 'yet'])

def roman_numeral(number):
    num = [1, 4, 5, 9, 10, 40, 50, 90,
        100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL",
        "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12

    numeral = ""
    while number:
        div = number // num[i]
        number %= num[i]

        while div:
            numeral += sym[i]
#            print(sym[i], end = "")
            div -= 1
        i -= 1
    return numeral

numerals = [roman_numeral(i).lower() for i in range(100)]
stop.extend(numerals)
#nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
#nlp.max_length = 100000000
lemmatizer = WordNetLemmatizer()

POS = ['NOUN']
TITLE_RE = r"^Title: "
AUTHOR_RE = r"^Author: "
START_RE = r'\*+\s*START OF'
END_RE = r'\*+\s*END OF'
PARSE_MWES = True
SUMMARIZE = False
TARGET_WORDCOUNT = 1000

if PARSE_MWES:
    mwe_tokenizer = setup_mwes()

class MethodStep(IntEnum):
    basic = 0
    trimwords = 1
    stopwords = 2
    onlynouns = 3
    lemmatize = 4
    bigrams = 5
    mwes = 6

METHOD = MethodStep.basic
class Book:

    @classmethod
    def load(self, dirname = 'PG', matching = None, cleaned = False):
        path = dirname
        files = [str(child.resolve()) for child in sorted(Path.iterdir(Path(path)))]
        if matching:
            files = [file for file in files if file.endswith('.txt') and matching in file]
        else:
            files = [file for file in files if file.endswith('.txt')]

        books  = [Book(f, cleaned) for f in files]
        valid = [b for b in books if b.has_body()]
        print(f'{len(valid)} valid books out of total {len(books)}')

        return valid

    def __init__(self, file, cleaned = False):
        self.title = ''
        self.author = ''
        self.file = file
        self.body = ''
        self.body_lines = []
        self.lemma = []
        self.process(cleaned)

    def has_body(self):
        return bool(len(self.body_lines))

    def process(self, cleaned):
        def is_word(word):
            word = word.lower()
            return len(word) > 1 and word.isalpha() and word not in stop

        def is_noun(pos):
            return pos.startswith('NN') and pos not in ['NNP', 'NNPS'] # no proper nouns

        def _tokenize(body, trim = False, only_nouns = False):
            if trim:
                # We only want first 1000 words trimmed at nearest sentence.
                sentences = sent_tokenize(body)
                count = 0
                words = []
                for sent in sentences:
                    if only_nouns:
                        tokenized = list(tokenize(sent, lower = False))
                        tagged = pos_tag(tokenized)
                        w = [word.lower() for word, pos in tagged if pos.startswith('NN') and is_noun(pos) and is_word(word)]
                    else:
                        w = list(tokenize(sent, lower = False))
                    count += len(w)
                    words = words + w
                    if count >= TARGET_WORDCOUNT:
                        break
                return words
            else:
                return [w.lower() for w in word_tokenize(body) if w.isalpha()]

        def _remove_stopwords(words):
            return [word for word in words if is_word(word)]

        def remove_all_but_nouns(body):
            return _tokenize(body, trim = True, only_nouns = True)

        def lemmatize(body):
            words = remove_all_but_nouns(body)
            return [lemmatizer.lemmatize(word) for word in words]

        def mwes(body):
            sentences = sent_tokenize(body.replace('_', '').replace('\n', ''))
            tokenized = [mwe_tokenizer.tokenize(list(tokenize(sent, lower = False))) for sent in sentences]
            tagged = [pos_tag(sent) for sent in tokenized]
            sentences = [[lemmatizer.lemmatize(word.lower()) for word, pos in sent if is_noun(pos) and is_word(word)] for sent in tagged]
            sent = list(flatten(sentences))
            return sent

        def bigrams(body):
            # Add bigrams
            sentences = sent_tokenize(body.replace('_', '').replace('\n', ''))
            tokenized = [list(tokenize(sent, lower = False)) for sent in sentences]
            tagged = [pos_tag(sent) for sent in tokenized]
            sentences = [[lemmatizer.lemmatize(word.lower()) for word, pos in sent if is_noun(pos) and is_word(word)] for sent in tagged]
            model = Phrases(sentences, min_count = 3, threshold = 10)
            tokens = [preprocess_string(" ".join(sent), []) for sent in sentences]
            bigrams = model[tokens]
            rtn = []
            for b in bigrams:
                for w in b:
                    if '_' in w:
                        rtn.append(w.replace('_', ' '))
            return rtn

        self.parse_book(cleaned)

        if self.has_body():
            print(f'Parsed book: {self.title} by {self.author} w/ {len(self.body_lines)} lines of text.')

            self.lemma = _tokenize(self.body, METHOD >= MethodStep.trimwords)
            match METHOD:
                case MethodStep.basic:
                    pass
                    # nothing, handled above
                case MethodStep.trimwords:
                    self.lemma = _tokenize(self.body, True)
                case MethodStep.stopwords:
                    self.lemma = _remove_stopwords(self.lemma)
                case MethodStep.onlynouns:
                    self.lemma = remove_all_but_nouns(self.body)
                case MethodStep.lemmatize:
                    self.lemma = lemmatize(self.body)
                case MethodStep.bigrams:
                    self.lemma = bigrams(self.body)
                case MethodStep.mwes:
                    self.lemma = mwes(self.body)

        return self

    def __str__(self):
        return f'{self.title} ({len(self.body)}) by {self.author}'

    def parse_book(self, cleaned = False):
        if cleaned:
            filename = self.file.split('/')[-1]
            parts = filename.replace('.txt', '').split('-by-')
            title = parts[0].replace('_', ' ')
            author = parts[1].replace('_', ' ')
            self.title = title
            self.author = author
            with open(self.file, 'r', encoding = 'latin') as f:
                for line in f.readlines():
                    self.body_lines.append(line)
        else:
            is_reading = False
            found_title = False
            found_author = False
            with open(self.file, 'r', encoding = 'latin') as f:
                for line in f.readlines():
                    if not found_title and re.search(TITLE_RE, line):
                        self.title = re.sub(TITLE_RE, "", line).strip()
                        found_title = True

                    if not found_author and re.search(AUTHOR_RE, line):
                        self.author = re.sub(AUTHOR_RE, "", line).strip()
                        found_author = True

                    # Don't save header content which is a lot of messaging from Gutenberg
                    if not is_reading and re.search(START_RE, line):
                        is_reading = True
                        continue # skip this line

                    # Don't save footer content either
                    elif is_reading and re.search(END_RE, line):
                        is_reading = False

                    line = line.strip()
                    if is_reading and len(line) > 0:
                        self.body_lines.append(line)

        f.close()
        self.body = ' '.join(self.body_lines)

FIND_COHERENCE = False
NUM_TOPICS = 32
if __name__ == '__main__':
    def compute_coherence_values(dictionary, corpus, texts, nums):
        coherence_values = []
        model_list = []
        for num_topics in nums:
            print(f'num_topics: {num_topics}')
            model = LdaModel(corpus = corpus,
                id2word = dictionary.id2token,
                chunksize = 20,
                alpha = 'auto',
                eta = 'auto',
                iterations = 50,
                num_topics = num_topics,
                passes = 20,
                eval_every = None)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def run_analysis(start = 2, limit = 41, step = 3):
        x = list(range(start, limit+1, step))
        docs = [book.lemma for book in books]

        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below = 3, no_above = 0.5)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        temp = dictionary[0] # prime the dictionary ???
        id2word = dictionary.id2token

        model_list, coherence_values = compute_coherence_values(dictionary = dictionary, corpus = corpus, texts = docs, nums = x)
        print(coherence_values)
        plt.figure()
        plt.plot(x, coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.savefig('coherence.png')

    method_str = str(METHOD).split('.')[1]
    cache_name = f'books-{NUM_TOPICS}-{method_str}.pkl'
    if os.path.isfile(cache_name):
        print('already have books cached')
        pickle_file = open(cache_name, 'rb')
        books = pickle.load(pickle_file)
        pickle_file.close()
        print(f'Loaded {len(books)} cached books')
    else:
        books = Book.load('Books', matching = None, cleaned = True)
        pickle_file = open(cache_name, 'wb')
        pickle.dump(books, pickle_file)
        pickle_file.close()


    if FIND_COHERENCE:
        run_analysis(2, 60, 2)
    else:
        docs = [book.lemma for book in books]

        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below = 3, no_above = 0.5)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        temp = dictionary[0] # prime the dictionary ???
        id2word = dictionary.id2token

        model = LdaModel(corpus = corpus,
            id2word = id2word,
            chunksize = 20,
            alpha = 'auto',
            eta = 'auto',
            iterations = 50,
            num_topics = NUM_TOPICS,
            passes = 20,
            eval_every = None)

        top_topics = model.top_topics(corpus)

        prepared = pyLDAvis.gensim.prepare(model, corpus, dictionary, mds = 'mmds')
        filename = './books-'+ str(NUM_TOPICS) + '-' + method_str + '.html'
        print(filename)
        pyLDAvis.save_html(prepared, filename)

        for i in range(5):
            print(model.get_document_topics

#        print("Making word clouds")
#        for i in range(NUM_TOPICS):
#            w = WordCloud().fit_words(dict(model.show_topic(i, 50)))
#            plt.imshow(w)
#            plt.savefig('book-wordcloud-' + method_str +'-topic-' + str(i) + '.png')
#            plt.axis("off")
#
#        print(f'Finding topics for first 10 books')
#        for b in books:
#            print(b.title)
#            topics = model.get_document_topics(dictionary.doc2bow(b.lemma))
#            ids = [a for a,b in topics]
#            print(model.show_topic(ids[0]))
