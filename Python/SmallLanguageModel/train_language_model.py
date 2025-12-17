import random
import numpy as np
import sklearn
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from collections import defaultdict

# Constants for max embedding size and sentence batch size
EMBEDDING_WIDTH = 100
MINIBATCH_SIZE = 5000


"""
Note that there are three special tokens we need to take into account here.
 Special Token: <START> -- the special token denoting the beginning of a sentence.
   - It happens twice to start every sentence.
   - Its word ID number is 0.
   - Its embedding is defined to be [1] * EMBEDDING_WIDTH
 Special Token: </s> -- the end of every sentence!
   - It's how we know to stop generating, and it will be the last entry in a sequence.
   - Its word ID number is 1.
   - This actually does not have an embedding since it will never be in the context.
 Special Token: <UNK>
   - This is our representation for a token we do not have in our vocabulary! This will happen sometimes,
     since our vocabulary is just words that occurred 5 or more times in the training data.
   - Unknown words get the word ID number 2.
   - Our embedding for <UNK> is defined to be np.zeros(EMBEDDING_WIDTH)
"""

def load_embeddings(filename):
    """ Returns a dictionary mapping from words to their embeddings, loading
    them from a word2vec vec.txt file """
    # Dictionary to store the words and their embeddings
    words_to_embeddings = {}
    # Creating the embedding array
    words_to_embeddings["<START>"] = np.array([1.0] * EMBEDDING_WIDTH)
    words_to_embeddings["<UNK>"] = np.zeros((EMBEDDING_WIDTH,))
    # Open the provided file
    with open(filename) as infile:
        # skip the first line
        firstline = next(infile)
        for line in infile:
            # Splits the line into a list of strings
            tokens = line.strip().split()
            # Takes the first token as the word
            word = tokens[0]
            # Takes the rest as vector components
            embedding = tokens[1:]
            # Converts the vector components into floats
            embedding = [float(token) for token in embedding]
            # Converts the list into a numpy array
            embedding = np.array(embedding)
            # Stores it in a dictionary
            words_to_embeddings[word] = embedding
    return words_to_embeddings


def load_vocabulary(filename):
    """ Returns two dictionaries -- the first maps from words to their IDs and
    the second maps from IDs to words. These are also loaded from a word2vec
    vec.txt file, for simplicity. This allows us to turn text into numeric
    labels and convert model outputs back into words """

    # Building a lookup table for word to integer ID
    vocab_lookup = {}
    vocab_lookup["<START>"] = 0
    vocab_lookup["</s>"] = 1
    vocab_lookup["<UNK>"] = 2
    # Building a lookup table for integer ID to word
    index_to_word = {}
    index_to_word[0] = "<START>"
    index_to_word[1] = "</s>"
    index_to_word[2] = "<UNK>"
    # Open our vocabulary file
    with open(filename) as infile:
        # skip the first line
        firstline = next(infile)
        # Assigning an ID to every word in the file
        for position, line in enumerate(infile):
            tokens = line.strip().split()
            word = tokens[0]
            # 0th word in the file gets ID 3 and onwards
            word_id = 3 + position
            # Stores in both directions
            vocab_lookup[word] = word_id
            index_to_word[word_id] = word
    return vocab_lookup, index_to_word


def get_embedding(embeddings, word):
    """ Get the embedding of a specified word in the embeddings we have """
    if word in embeddings:
        return embeddings[word]
    else:
        # The word does not exist in our data
        return embeddings["<UNK>"]


def get_word_id(word_to_id, word):
    """ Get the word ID of a specified word """
    if word in word_to_id:
        return word_to_id[word]
    else:
        # The ID does not exist in our data
        return word_to_id["<UNK>"]


def X_y_for_sentence(sentence, embeddings, word_to_id):
    """ Prepare a single sentence for training.
    Returns a list of concatenated embeddings (so 200-wide numpy vectors) and
    a corresponding list of word labels """
    X_list = []
    y_list = []
    # prior-prior word
    prevprev = "<START>"
    # prior word
    prev = "<START>"
    # For every word in a sentence
    for token in sentence + ["</s>"]:
        # Look up the embeddings for the previous and previous-previous words
        prevprev_embedding = get_embedding(embeddings, prevprev)
        prev_embedding = get_embedding(embeddings, prev)
        # Add both embeddings into a single vector
        both_embeddings = np.concatenate([prevprev_embedding, prev_embedding])
        X_list.append(both_embeddings)
        # Find the target word id
        target = get_word_id(word_to_id, token)
        y_list.append(target)
        # The previous-previous word now becomes the previous word
        prevprev = prev
        # The previous word is our next word in the sentence
        prev = token
    return X_list, y_list


def generate_minibatches(filename, embeddings, vocabulary):
    """ Generator that yields little batches, suitable for training with partial_fit """
    X_list = []
    y_list = []
    with open(filename) as infile:
        for line in infile:
            # Normalize
            line = line.strip()
            # Get the words in that line
            tokens = line.split()
            # Turn the sentence into training pairs
            X_add, y_add = X_y_for_sentence(tokens, embeddings, vocabulary)
            # Add the context vector with prevprev and prev embeddings to the end of the list
            X_list.extend(X_add)
            # y_list holds the target for the next word ID
            y_list.extend(y_add)
            #
            if len(X_list) >= MINIBATCH_SIZE:
                # Makes sure they are the same size before converting them to numpy arrasys
                assert len(X_list) == len(y_list)
                X = np.array(X_list)
                y = np.array(y_list)
                # Resets the list
                X_list = []
                y_list = []
                # Uses yield so the caller can train on that batch
                yield (X, y)

    # After the file ends, it yields the final leftover batch
    assert len(X_list) == len(y_list)
    X = np.array(X_list)
    y = np.array(y_list)
    yield (X, y)


def batch_for_file(filename, embeddings, vocabulary):
    """ Creates a training batch for the entire file, same as minibatches except the entire file """
    X_list = []
    y_list = []
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            tokens = line.split()
            X_add, y_add = X_y_for_sentence(tokens, embeddings, vocabulary)
            X_list.extend(X_add)
            y_list.extend(y_add)
    assert len(X_list) == len(y_list)
    X = np.array(X_list)
    y = np.array(y_list)
    return (X, y)


def train_neural_network(clf, embeddings, vocabulary, theclasses):
    """ Train a neural network based on the clf (classifier) """
    # Number of Epochs for training. More = longer training and maybe more accuracy/less loss. (training length)
    NUM_EPOCHS = 50
    # Creating validation batches to measure progress after each epoch
    X_validation, y_validation = batch_for_file("sentences_validation", embeddings, vocabulary)
    print("training over minibatches...")
    # List of accuracy and loss that the language model will have
    accuracies = []
    losses = []
    # In every epoch, we will:
    for epoch in range(NUM_EPOCHS):
        print("epoch", epoch)
        # Iterate over the minibatches from sentences_train file
        for batchnum, (X_batch, y_batch) in tqdm.tqdm(
            enumerate(generate_minibatches("sentences_train", embeddings, vocabulary))
        ):
            # partial_fit updates the model weights incrementally per batch
            clf.partial_fit(X_batch, y_batch, classes=theclasses)
        # Computes validation predictions
        y_pred = clf.predict_proba(X_validation)
        # Calculates validation loss (cross-entropy)
        validation_loss = sklearn.metrics.log_loss(
            y_validation, y_pred, labels=theclasses
        )
        # Calculates validation loss
        validation_acc = clf.score(X_validation, y_validation)
        # Save both the losses and accuracy for logging
        accuracies.append(validation_acc)
        losses.append(validation_loss)
        print("validation loss", validation_loss)
        print("validation accuracy", validation_acc)
    # Show a plot of the loss (should be decreasing every epoch)
    plt.plot(losses)
    plt.show()


def negative_log_probability_for_sequence(
    clf, embeddings, vocabulary, index_to_word, sequence
):
    # Initilize the total of the log prob
    totalLogProb = 0.0
    # Initlize the tokens for context
    secondLastWord = "<START>"
    lastWord = "<START>"
    # We need to predict every word in the sequence + the end token
    targetWords = sequence + ["</s>"]
    # Now we begin the main loop
    for word in targetWords:
        # Get the embeddings from the previous two words
        embedSecondLast = get_embedding(embeddings, secondLastWord)
        embedLast = get_embedding(embeddings, lastWord)
        # Make them into an inut vector sklearn accepts
        inputVec = np.concatenate([embedSecondLast, embedLast]).reshape(1, -1)
        # This returns a list of probabilites for every word in the vocab
        wholeVocab = clf.predict_proba(inputVec)
        # Find what the id for the word of interest
        wordId = get_word_id(vocabulary, word)
        # Get the probability for that word Id
        currentProb = wholeVocab[0][wordId]
        # We need to get the surprise of the model, set to very small instead of 0
        if currentProb == 0:
            currentProb = 1e-16
        # Taking the negative log base 2 of the word probability for surprise
        surprise = -np.log2(currentProb)
        # Add this to the total
        totalLogProb += surprise
        # Update the word context during the next loop
        secondLastWord = lastWord
        lastWord = word
    return totalLogProb


def sequences_from_file(filename, shuffled=False):
    """ Get the sequences of words from the input file, either shuffled or not.
    This is to test whether our model is surprised or not by the data """
    output = []
    # Go into the input file
    with open(filename) as infile:
        for line in infile:
            # Normalize the lines
            line = line.strip()
            # Split the line up by words
            tokens = line.split()
            # Shuffle the words in that sentence if needed
            if shuffled:
                random.shuffle(tokens)
            output.append(tokens)
    return output


def model_suprise(clf, embeddings, vocabulary, index_to_word):
    """ Calculating whether the model is suprised by the sequence of words in the sentence"""
    # Get the normal and shuffled word sequences
    sequences = sequences_from_file("sentences_test")
    sequences_shuffled = sequences_from_file("sentences_test", shuffled=True)
    # We will store the scores in a list
    realScore = []
    shuffleScore = []
    # Calculate surprise for the normal sentences
    for sequence in sequences:
        # Getting the negative log probability for the non-shuffled sentences, this will be our score
        score = negative_log_probability_for_sequence(
            clf, embeddings, vocabulary, index_to_word, sequence
        )
        # Add the score of the real sentences to the list
        realScore.append(score)
    # Calculate surprise for the shuffled sentences
    for sequence in sequences_shuffled:
        # Getting the negative lof probability for the shuffled sentences
        score = negative_log_probability_for_sequence(
            clf, embeddings, vocabulary, index_to_word, sequence
        )
        # Add the score of the shuffled sentences to the list
        shuffleScore.append(score)
    # Outputting the results
    print("Average surprise for the real sentences: ", np.mean(realScore))
    print("Average surprise for the shuffled sentences: ", np.mean(shuffleScore))


def sample_from_model(clf, embeddings, vocabulary, index_to_word):
    """ This function will sample some sentences from the model. It will generate these sentences """
    secondLastWord = "<START>"
    lastWord = "<START>"
    generatedWords = []

    # Make sure the loop does not run forever just in case
    MAX_LENGTH = 50

    for _ in range(MAX_LENGTH):
        # Get the input ready
        embedSecondLast = get_embedding(embeddings, secondLastWord)
        embedLast = get_embedding(embeddings, lastWord)
        inputVec = np.concatenate([embedSecondLast, embedLast]).reshape(1, -1)
        # Get the probabilities
        probabilities = clf.predict_proba(inputVec)[0]
        # Sampling a word
        indices = np.arange(len(probabilities))
        chosenIdx = np.random.choice(indices, p=probabilities)
        # Convert the ID back into the word of interest
        chosenWord = index_to_word[chosenIdx]
        # Stop if needed
        if chosenWord == "</s>":
            break
        # Update the context and add generated words
        generatedWords.append(chosenWord)
        secondLastWord = lastWord
        lastWord = chosenWord
    return generatedWords


def main():
    """ Runs the main model training and evalution """
    # Load embeddings + vocab that everything else needs
    embeddings = load_embeddings("vec.txt")
    vocabulary, index_to_word = load_vocabulary("vec.txt")

    # Create classifier instance. Update the hidden layer/activation function
    clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(300, 600), activation="relu")

    # Build the classes for partial_fit / log_loss
    the_classes = np.arange(max(index_to_word.keys()) + 1)

    # Train first so generation makes sense
    train_neural_network(clf, embeddings, vocabulary, the_classes)

    print("Generating 20 sentences")
    # Going to generate 20 sentences from the model to see what it learned
    for i in range(20):
        # Get a sentence from the model
        sentences = sample_from_model(clf, embeddings, vocabulary, index_to_word)
        # Join the list to a string
        print(f"{i + 1}: {' '.join(sentences)}")

    # Next word prediction
    both_embeddings = np.concatenate([embeddings["<START>"], embeddings["<START>"]])
    res = clf.predict_proba(np.array([both_embeddings]))
    # Bring the most proabable word indexes to the front
    most_prob_indices = np.argsort(-res)
    for i in range(20):
        # Takes the i-th highest probability word ID
        index = most_prob_indices[0][i]
        # Converts the id back into the word string
        print("word", index, index_to_word[index])

    # Surprise comparison for shuffled and non-shuffled sentences
    model_suprise(clf, embeddings, vocabulary, index_to_word)


if __name__ == "__main__":
    main()
