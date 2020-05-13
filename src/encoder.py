import numpy as np
import re


class Encoder(object):

    # Constructor
    def __init__(self, dim, words):
        raise NotImplementedError

    # Shape getter
    @property
    def shape(self):
        return (len(self.vectors), self.dim)

    # Encode word to vector
    def encode(self, word):
        raise NotImplementedError

    # Decode vector to word
    def decode(self, vector):
        raise NotImplementedError


class OneHotEncoder(Encoder):

    # Constructor
    def __init__(self, dim, words):
        # Store dimension
        self.dim = dim
        # Ensure words uniqueness
        words = set(words)
        # Set word to index
        self.w2i = {w: i for i, w in enumerate(words)}
        # Set index to word
        self.i2w = {i: w for i, w in enumerate(words)}
        # Initialize vectors (as lists, more memory efficient)
        self.vectors = [[int(j == i) for j in len(words)] for w, i in self.w2i]

    # Word to vector
    def encode(self, word):
        return np.array(self.w2i[word], dtype=np.float)

    # Vector to word
    def decode(self, vector):
        # If no value is set, return all zeroes
        if not np.sum(vector):
            return None
        # Get index of selected word
        i = np.arange(0, self.dim) * vector
        # Return decoded word
        return self.i2w[i]


class WordToVector(Encoder):

    # Constructor
    def __init__(self, dim, words, w2v={}):
        # Store vectors dimension
        self.dim = dim
        # Initialize vectors as numpy matrix
        vectors, vectors_mean, vectors_std = WordToVector.vectorize(w2v, dim)
        # Get all the words available
        words = list(set(words) | set(w2v.keys()))
        # Set word to index mapping and vice-versa
        self.w2i = {w: i for i, w in enumerate(words)}
        self.i2w = {i: w for i, w in enumerate(words)}
        # Get vector for every words (eventually make it from distributions)
        self.vectors = list()
        for i, word in enumerate(words):
            # Define i-th word vector
            vector = np.random.normal(vectors_mean, vectors_std).tolist()
            vector = [float(x) for x in w2v.get(word, vector)]
            # Store i-th word vector
            self.vectors.append(vector)
        # Store distributions parameters
        self.vectors_mean, self.vectors_std = vectors_mean, vectors_std

    def encode(self, word):
        # If there is a vector associated with current word, get it
        if word in set(self.w2i.keys()):
            vector = self.vectors[self.w2i[word]]
        # Otherwise, generate a random vector from distributions
        else:
            vector = np.random.normal(self.vectors_mean, self.vectors_std)
        # Return retrieved vector
        return vector

    def decode(self, vector):
        # Get vectors matrix
        vectors = np.array(self.vectors, dtype=np.float)
        # Compute euclidean istances from each known word vector
        distances = np.sqrt(np.sum((vector - vectors) ** 2, axis=1))
        # Get index of closest vector (first if many)
        i = np.argmin(distances)
        # Return word associated with closest vector
        return self.i2w[i]

    # Vectorize a dictionary of vectors, returns either distributions
    @staticmethod
    def vectorize(w2v, dim):
        # Make vectors matrix
        vectors = np.array(list(w2v.values()), dtype=np.float)
        vectors_mean = np.mean(vectors[:, :dim], axis=0)  # Compute mean
        vectors_std = np.std(vectors[:, :dim], axis=0)  # Compute std dev
        # Return either vectorized dictionary, means and standard deviations
        return vectors, vectors_mean, vectors_std

    # Load dictionary from file (e.g. glove)
    @staticmethod
    def from_csv(in_path, dim, words, sep=' '):
        # Initialize word to vector dictionary
        w2v = dict()
        # Open tsv containing
        with open(in_path, 'r') as in_file:
            # Loop through each line in file
            for line in in_file:
                # Clean line
                line = re.sub(r'[\n\r]+', '', line)
                # Split line using separator
                line = line.split(sep)
                # First column is word, others are vector dimensions
                word, vector = line[0], line[1:dim+1]
                # Store word its associated vector
                w2v[word] = vector
        # Generate a new istance of W2V encoder
        return WordToVector(dim, words, w2v)
