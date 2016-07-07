# Use Python 3, or face the Unicode horror!

# Standard library:
import glob
import gzip

# Additional modules:
import nltk.data
from nltk.tokenize import word_tokenize
from lxml import etree,html

sent_tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle')

def sentence_generator(location):
    for path in glob.glob(location):
        with open(path) as f:
            contents = '<contents>'+f.read()+'</contents>'
            root = html.fromstring(contents)
            for doc in root.findall('doc'):
                for chunk in doc.text.split('\n'):
                    for sentence in sent_tokenizer.tokenize(chunk):
                        yield sentence

def yield_number(location, n):
    i = 0
    for sentence in sentence_generator(location):
        tokenized = word_tokenize(sentence)
        if len(tokenized) > 4:
            yield sentence
            i+=1
            if i > n:
                break

def write_plain(location, filename, max_sents):
    with open(filename,'wt') as f:
        for sentence in yield_number(location, n=max_sents):
            f.write(sentence + '\n')

write_plain('./dutch_text/*/*', 'nl_wiki_plain.txt', max_sents=20000)
write_plain('./english_text/*/*', 'en_wiki_plain.txt', max_sents=20000)
