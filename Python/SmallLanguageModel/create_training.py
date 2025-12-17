#!/usr/bin/env python3

from nltk.corpus import brown

# Uses the brown corpus to extract sentences for training
for sent in brown.sents():
    print(" ".join(sent).lower())
