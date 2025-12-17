import numpy as np
from collections import defaultdict

# Maximum width of the word embedding matrix
EMBEDDING_WIDTH = 100

def embedding_cosine(word_embeddings, word1, word2):
    """ Gets the cosine of two word embeddings (how close a word is related to another """
    embedding1 = word_embeddings[word1]
    embedding2 = word_embeddings[word2]
    # Get the dot product of both the word embeddings, how aligned they are
    dotproduct = np.dot(embedding1, embedding2)
    # Divides the product od their magnitudes to normalize
    cosine = dotproduct / (np.linalg.norm(embedding1) *
                           np.linalg.norm(embedding2))
    # If the value is close to 1 then they are closely related, closer to 0 means unrelated
    return cosine

def find_most_similar(word_embeddings, query):
    """ Builds a list of the 5 most similar words. Points in the same direction as query word"""
    best_ones = []
    # Loops over the all the words in the vocabulary
    for key in word_embeddings:
        # If the word is the same, we skip
        if key == query: continue
        # Get the score of the two words of interest
        score = embedding_cosine(word_embeddings, query, key)
        # Add it to the best scores
        best_ones.append((score, key))
        # Begin sorting before filtering the values
        best_ones.sort(reverse=True)
        # Only keep the 5 best words (most similar)
        best_ones = best_ones[:5]
    return best_ones

def main():
    # Initialize the embedding matrix
    word_embeddings = defaultdict(lambda: np.zeros(EMBEDDING_WIDTH))
    # Open the inputfile
    with open("vec.txt") as infile:
        # Skip the first line
        next(infile)
        # Parse every remianing line
        for line in infile:
            line = line.strip()
            # The first token is the word, the rest are numbers
            fields = line.split()
            word = fields[0]
            assert len(fields[1:]) == EMBEDDING_WIDTH, "we have a problem"
            # Converts the numbers to a numpy vector
            floats = [float(field) for field in fields[1:]]
            embedding = np.array(floats)
            word_embeddings[word] = embedding

    print("The Cosine between freak and freak")
    print(embedding_cosine(word_embeddings, "freak", "freak"))

    print("The Cosine between dog and cat")
    print(embedding_cosine(word_embeddings, "dog", "cat"))

    print("Thee Cosine between credible and urged")
    print(embedding_cosine(word_embeddings, "credible", "urged"))

    # Commandline interface
    while True:
        try:
            query = input("What words are most similar?: ")
            embedding = word_embeddings[query]

            most_similar = find_most_similar(word_embeddings, query)
            print(f"The most similar are: {most_similar}")

        except EOFError as e:
            print()
            print("See Ya!")
            break


if __name__ == "__main__": main()

