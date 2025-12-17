import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.corpus import reuters

# Create many sentences using a larger corpus (brown, gutenberg, reuters)
for corpus in [brown, gutenberg, reuters]:
    for sent in corpus.sents():
        line = (" ".join(sent)).lower()
        print(line)
