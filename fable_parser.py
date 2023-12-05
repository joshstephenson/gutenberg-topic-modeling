from enum import Enum
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag, FreqDist, ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import MWETokenizer
from nltk.lm.preprocessing import flatten
import json
from sys import stderr
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, Phrases, LdaModel
from gensim.utils import simple_preprocess
import pyLDAvis
import pyLDAvis.gensim
#import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def is_title(text):
    return text.upper() == text

def setup_mwes():
    mwe_lines = []
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

PARSE_MWES = False
ONLY_MWES = False

stop = stopwords.words('english')
stop.extend(['one'])
lemmatizer = WordNetLemmatizer()
#spacy_model = 'en_core_web_lg'
#try:
#    nlp = spacy.load(spacy_model, disable = ['parser', 'ner'])
#except OSError:
#    print('Downloading language model for the spaCy POS tagger\n'
#        "(don't worry, this will only happen once)", file=stderr)
#    from spacy.cli import download
#    download(spacy_model)
#    nlp = spacy.load(spacy_model, disable = ['parser', 'ner'])

if PARSE_MWES:
    mwe_tokenizer = setup_mwes()


class ParseMode(Enum):
    TITLE = 0
    BODY = 1
    MORAL = 2

class Fable:

    @classmethod
    def load(self, file = "/Users/joshuastephenson/Documents/MSU/Classes/Data Mining/Final_Project/aesop_fables.txt"):
        fables = []
        mode = ParseMode.TITLE
        current = None

        with open(file, 'r', encoding = 'latin') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line):
                    if mode == ParseMode.TITLE and is_title(line):
                        current = Fable(line)
                        mode = ParseMode.BODY
                    elif mode == ParseMode.BODY:
                        current.body_lines.append(line)
                    elif mode == ParseMode.MORAL:
                        # sometimes a fable doesn't have a moral and we go straight into another title
                        if is_title(line):
                            fables.append(current.process())
                            current = Fable(line)
                            mode = ParseMode.BODY
                        else:
                            current.moral_lines.append(line)
                else: # line is blank
                    if mode == ParseMode.BODY and len(current.body_lines): # we've reached the end of the body
                        mode = ParseMode.MORAL
                    elif mode == ParseMode.MORAL and len(current.moral_lines): # we've reached the end of the moral
                        fables.append(current.process())
                        mode = ParseMode.TITLE
            fables.append(current.process())

        f.close()
        return fables

    def __init__(self, title):
        self.title = title.title()
        self.body = None
        self.moral = None
        self.body_lines = []
        self.moral_lines = []

    def process(self):
        def is_word(word):
            word = word.lower()
            return word.isalpha() and len(word) > 1 #and word not in stop

        self.body = ' '.join([self.title] + self.body_lines)
        self.moral = ' '.join(self.moral_lines)
        self.words = [word.lower() for word in word_tokenize(self.body) if is_word(word)]
        self.lemma = [lemmatizer.lemmatize(word) for word in self.words]

        if PARSE_MWES:
            self.words = [word for word in mwe_tokenizer.tokenize(self.words)]
        if ONLY_MWES:
            self.words = [word for word in self.words if '_' in word]

        return self


    def __str__(self):
        return f'{self.title} ({len(self.body)})\n\n\t{self.moral}'

NUM_TOPICS = 7
if __name__ == '__main__':
    fables = Fable.load()[:10]

    docs = [fable.lemma for fable in fables]
    bigram = Phrases(docs, min_count = 3)
    for d in docs:
        for b in bigram[d]:
            if '_' in b: # bigram
                d.append(b)

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above = 0.5)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    temp = dictionary[0]
    id2word = dictionary.id2token
    model = LdaModel(corpus = corpus,
                     id2word = id2word,
                     chunksize = 20,
                     alpha = 'auto',
                     eta = 'auto',
                     iterations = 50,
                     num_topics = NUM_TOPICS,
                     passes = 10,
                     eval_every = None)

    top_topics = model.top_topics(corpus)
    print(top_topics)
#    corpus = []
#    id2word = corpora.Dictionary()
#    for fable in fables:
#        id2word.add_documents(fable.lemma)
#        all_lemma.append(fable.lemma)
#
#    corpus = [id2word.doc2bow(text) for text in all_lemma]
#
#    num_topics = 7
#    lda_model = gensim.models.LdaMulticore(corpus = corpus, id2word = id2word, num_topics = num_topics)
#
    prepared = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(prepared, './fables_prepared_'+ str(NUM_TOPICS) +'.html')

    print(f'Finding topics for first fable')
    for doc, fable in zip(docs, fables):
        print(fable.body)
        topic_ids = [a for a,b in model.get_document_topics(dictionary.doc2bow(doc))]
        print(model.show_topic(topic_ids[0]))
#
#
#    for i in range(num_topics):
#        w = WordCloud().fit_words(dict(lda_model.show_topic(i, 20)))
#        plt.imshow(w)
#        plt.savefig('wordcloud-topic-' + str(i) + '.png')
#        plt.axis("off")

    def compute_coherence_values(dictionary, corpus, texts, nums):
        coherence_values = []
        model_list = []
        for num_topics in nums:
            print(f'num_topics: {num_topics}')
            model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def run_analysis():
        limit=4; start=2; step=2;
        x = list(range(2, 41, 1))
        model_list, coherence_values = compute_coherence_values(dictionary = id2word, corpus = corpus, texts = all_lemma, nums = x)
        print(coherence_values)
        plt.figure()
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.savefig('coherence.png')
