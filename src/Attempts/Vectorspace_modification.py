"""Abstract data type definitions for vector space model that supports
   cosine similarity queries using TF-IDF matrix built from the corpus.
"""

import sys
import concurrent.futures

from math import sqrt, log10
from typing import Callable, Iterable, List, Any
# from nltk.stem import StemmerI
from nltk import SnowballStemmer
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

__author__ = "Luke Rozinskas"
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Luke Rozinskas", "Mike Ryu"]
__license__ = "MIT"
__email__ = "lrozinskas@westmont.edu"


class Vector:
    def __init__(self, elements: list[float] | None = None):
        self._vec = elements if elements else []

    def __getitem__(self, index: int) -> float:
        if index < 0 or index >= len(self._vec):
            raise IndexError(f"Index out of range: {index}")
        else:
            return self._vec[index]

    def __setitem__(self, index: int, element: float) -> None:
        if 0 <= index < len(self._vec):
            self._vec[index] = element
        else:
            raise IndexError(f"Index out of range: {index}")

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        elif other is None or not isinstance(other, Vector):
            return False
        else:
            return self._vec == other.vec

    def __str__(self) -> str:
        return str(self._vec)

    @property
    def vec(self):
        return self._vec

    @staticmethod
    def _get_cannot_compute_msg(computation: str, instance: object):
        return f"Cannot compute {computation} with an instance that is not a DocumentVector: {instance}"

    def norm(self) -> float:
        """ Euclidean norm of self"""
        if not self._vec:
            raise ValueError
        else:
            norm = sqrt(sum([x ** 2 for x in self._vec]))
        return norm

    def dot(self, other: object) -> float:
        """ Dot product of `self` and `other` vectors."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("dot product", other))
        else:
            if len(self._vec) != len(other._vec):
                raise IndexError
            else:
                dot = sum([x1 * x2 for x1, x2 in zip(self._vec, other._vec)])
                return dot

    def cossim(self, other: object) -> float:
        """Cosine similarity of `self` and `other` vectors."""

        if not isinstance(other, Vector) or self._vec is None or not self._vec:
            raise ValueError(self._get_cannot_compute_msg("cosine similarity", other))
        else:
            if len(self._vec) != len(other._vec):
                raise ValueError
            else:
                denominator = (self.norm() * other.norm())
                if denominator == 0.0:
                    return 0.0
                else:
                    return self.dot(other)/denominator

    def boolean_intersect(self, other: object) -> list[tuple[float, float]]:
        """Returns a list of tuples of elements where both `self` and `other` had nonzero values."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("boolean intersection", other))
        else:
            return [(e1, e2) for e1, e2 in zip(self._vec, other._vec) if e1 and e2]


# class Document:
#     _iid = 0
#
#     def __init__(self, title: str = None, words: list[str] = None, processors: tuple[set[str], SnowballStemmer] = None):
#         Document._iid += 1
#         self._iid = Document._iid
#         self._title = title if title else f"(Untitled {self._iid})"
#         self._words = list(words) if words else []
#
#         if processors:
#             exclude_words = processors[0]
#             stemmer = processors[1]
#             if not isinstance(exclude_words, set) or not isinstance(stemmer, SnowballStemmer):
#                 raise ValueError(f"Invalid processor type(s): ({type(exclude_words)}, {type(stemmer)})")
#             else:
#                 self.stem_words(stemmer)
#                 self.filter_words(exclude_words)

class Document:
    _iid = 0

    def __init__(self, title=None, words=None, timestamp=None, processors: tuple[set[str], SnowballStemmer] = None, ):
        Document._iid += 1
        self._iid = Document._iid
        self._title = title if title else f"(Untitled {self._iid})"
        self._words = list(words) if words else []
        self._timestamp = timestamp

        if processors:
            exclude_words, stemmer = processors
            self.stem_words(stemmer)
            self.filter_words(exclude_words)

    @property
    def timestamp(self):
        return self._timestamp

    def __iter__(self):
        return iter(self._words)

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        elif other is None or not isinstance(other, Document):
            return False
        else:
            return self._title == other.title and self._words == other.words

    def __hash__(self) -> int:
        return hash((self._title, tuple(self._words)))

    def __str__(self) -> str:
        words_preview = ["["]
        preview_size = 5
        index = 0

        while index < len(self._words) and index < preview_size:
            words_preview.append(f"{self._words[index]}, ")
            index += 1
        words_preview.append("... ]")

        return "[{i:04d}]: {title} {words}".format(
            i=self._iid,
            title=self._title,
            words="".join(words_preview)
        )

    @property
    def iid(self):
        return self._iid

    @property
    def title(self):
        return self._title

    @property
    def words(self):
        return self._words

    def filter_words(self, exclude_words: set[str]) -> None:
        """ Remove any words from `_words` that appear in `exclude_words` passed in."""
        self._words = self._words

        # Creates list for the words
        word_list = []

        for word in self._words:
            # checks if words are more than 1 char and alphabetical letters
            if word.isalpha() and word not in exclude_words:
                word_list.append(word)
        self._words = word_list

    def stem_words(self, stemmer: SnowballStemmer) -> None:
        """ Stem each word in `_words` using the `stemmer` passed in."""

        self._words = self._words

        # Creates list for stemmed words
        stem_list = []

        for word in self._words:
            # uses stemmer to stem the words
            stem = stemmer.stem(word)
            # adds them to the list
            stem_list.append(stem)
        self._words = stem_list

    def tf(self, term: str) -> int:
        """ Compute and return the term frequency of the `term` passed in among `_words`."""

        # Counter for the tf
        term_freq = 0
        the_words = self.words
        for word in the_words:
            # if the word matches a term in the doc add to count
            if word == term:
                term_freq += 1
        return term_freq


# New Code is here
def analyze_watch_times(corpus, keyword):
    # Create a list of tuples (document_index, timestamp) where the keyword is present
    # keyword_occurrences = [(idx, doc.timestamp) for idx, doc in enumerate(corpus.docs) if keyword in doc.words]
    keyword_occurrences = [(idx, doc.timestamp) for idx, doc in enumerate(corpus) if keyword in doc.words]

    # Sort the list based on timestamps
    keyword_occurrences.sort(key=lambda x: x[1])

    # Display the sorted list of (document_index, timestamp)
    print(f"Watch times for videos related to '{keyword}':")
    for idx, timestamp in keyword_occurrences:
        print(f"Document Index: {idx}, Timestamp: {timestamp}")

# Specify the path to your JSON file
json_file_path_wsl = '/home/lrozinskas/CS128/Data_folder/history/watch-history.json'

# Convert the WSL path to a Windows path
json_file_path_windows = os.path.normpath(json_file_path_wsl)

# Check if the file exists
if os.path.exists(json_file_path_windows):
    # Read data from the JSON file
    with open(json_file_path_windows, 'r') as json_file:
        data = json.load(json_file)

    # # Extract text data and timestamps from the dataset
    corpus = [Document(title=document['title'], words=document.get('description', ''), timestamp=document['time'])
              for document in data]

    # Create a list of Document objects
    # # documents = [Document(title=document['title'], words=document.get('description', ''), timestamp=document['time'])
    #              for document in data]

    # Create an instance of the Corpus class
    # corpus = Corpus(documents=documents)





    # TF-IDF calculation using scikit-learn
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([f"{document.title} {document.words}" for document in corpus])

    print("TF-IDF Matrix:")
    print(tfidf_matrix)

    # Print the terms (features) learned by the vectorizer
    # print("Terms (features):", vectorizer.get_feature_names())

    # def rank_documents(query, tfidf_matrix, corpus):
    #     query_vector = vectorizer.transform([query])
    #     similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    #     ranked_documents = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    #     return ranked_documents

    # def rank_documents(query, corpus):
    #     # TF-IDF calculation using scikit-learn
    #     vectorizer = TfidfVectorizer(stop_words='english')
    #     tfidf_matrix = vectorizer.fit_transform([f"{document.title} {document.words}" for document in corpus])
    #
    #     query_vector = vectorizer.transform([query])
    #     similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    #     ranked_documents = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    #
    #     return ranked_documents

    def rank_documents(query, vectorizer, tfidf_matrix, corpus):
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        ranked_documents = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return ranked_documents

    # # Example usage
    # query = "interesting video"
    # ranked_results = rank_documents(query, tfidf_matrix, corpus)
    #
    # # Display the ranked results
    # print("Ranked Results:")
    # for idx, score in ranked_results:
    #     print(f"Document Index: {idx}, Score: {score}")
    #     print("Title:", corpus[idx].title)
    #     print("Description:", corpus[idx].words)
    #     print("=" * 30)

    # Example of analyzing watch times for a specific keyword
    # analyze_watch_times(corpus, "sports")



else:
    print(f"File not found at path: {json_file_path_windows}")

class Corpus:
    def __init__(self, documents: list[Document], threads=1, debug=False):
        self._docs: list[Document] = documents

        # Setting flags.
        self._threads: int = threads
        self._debug: bool = debug

        # Bulk of the processing (and runtime) occurs here.
        self._terms = self._compute_terms()
        self._dfs = self._compute_dfs()
        self._tf_idf = self._compute_tf_idf_matrix()

    def __getitem__(self, index) -> Document:
        if 0 <= index < len(self._docs):
            return self._docs[index]
        else:
            raise IndexError(f"Index out of range: {index}")

    def __iter__(self):
        return iter(self._docs)

    def __len__(self):
        return len(self._docs)

    @property
    def docs(self):
        return self._docs

    @property
    def terms(self):
        return self._terms

    @property
    def dfs(self):
        return self._dfs

    @property
    def tf_idf(self):
        return self._tf_idf

    def _compute_terms(self) -> dict[str, int]:
        """ Computes and returns the terms (unique, stemmed, and filtered words) of the corpus."""

        filtered_words = []
        # iterates through each doc, and each word in those docs, checks if they're words and then adds to list
        for doc in self._docs:
            for word in doc.words:
                if word.isalpha():
                    filtered_words.append(word)
        return self._build_index_dict(filtered_words)

    def _compute_df(self, term) -> int:
        """Computes and returns the document frequency of the `term` in the context of this corpus (`self`)."""
        if self._debug:
            print(f"Started working on DF for '{term}'")
            sys.stdout.flush()

        def check_membership(t: str, doc: Document) -> bool:
            """ An efficient method to check if the term `t` occurs in a list of words `doc`."""
            for word in doc:
                if t == word:
                    return True

        return sum([1 if check_membership(term, doc) else 0 for doc in self._docs])

    def _compute_dfs(self) -> dict[str, int]:
        """Computes document frequencies for each term in this corpus and returns a dictionary of {term: df}s."""
        if self._threads > 1:
            return Corpus._compute_dict_multithread(self._threads, self._compute_df, self._terms.keys())
        else:
            return {term: self._compute_df(term) for term in self._terms.keys()}  # HINT: Using dictionary comprehension makes this a single-liner.

    def _compute_tf_idf(self, term, doc=None, index=None):
        """ Computes and returns the TF-IDF score for the term and a given document.
        An arbitrary document may be passed in directly (`doc`) or be passed as an `index` within the corpus.
        """
        dfs = self._dfs
        doc = self._get_doc(doc, index)

        dfs_terms = dfs.get(term)
        if dfs_terms is None:
            return 0.0
        elif term in dfs.keys():
            # gets the tf term of the formula
            tf_term = log10(1 + doc.tf(term))
            corpus_size = len(self.docs)
            # gets the idf portion of formula
            idf_term = log10(corpus_size / (1 + dfs_terms))
            tf_idf = tf_term * idf_term
            # makes sure that the term is actually in the doc
            if term in self.terms and len(self.docs) > 1:
                return tf_idf
            else:
                return 0

    def compute_tf_idf_vector(self, doc=None, index=None) -> Vector:
        """Computes and returns the TF-IDF vector for the given document.

        An arbitrary document may be passed in directly (`doc`) or be passed as an `index` within the corpus.

        """
        doc = self._get_doc(doc, index)

        # creates a list to be turned into a vector
        tf_idf_vector = []
        # iterates through all terms in the corpus
        for term in self._terms:
            # gets a score for each term in the doc
            tf_idf_score = self._compute_tf_idf(term, doc)
            # append it
            tf_idf_vector.append(tf_idf_score)

        return Vector(tf_idf_vector)

    def _compute_tf_idf_matrix(self) -> dict[str, Vector]:
        """Computes and returns the TF-IDF matrix for the whole corpus.

        The TF-IDF matrix is a dictionary of {document title: TF-IDF vector for the document}.

        """
        def tf_idf(document):
            if self._debug:
                print(f"Processing '{document.title}'")
                sys.stdout.flush()
            vector = self.compute_tf_idf_vector(doc=document)
            return vector

        matrix = {}
        if self._threads > 1:
            matrix = Corpus._compute_dict_multithread(self._threads, tf_idf, self._docs,
                                                      lambda d: d, lambda d: d.title)
        else:
            for doc in self._docs:

                matrix[doc.title] = tf_idf(doc)

                if self._debug:
                    print(f"Done with doc {doc.title}")
        return matrix

    def _get_doc(self, document, index):
        """A helper function to None-guard the `document` argument and fetch documents per `index` argument."""
        if document is not None and index is None:
            return document
        elif index is not None and document is None:
            if 0 <= index < len(self):
                return self._docs[index]
            else:
                raise IndexError(f"Index out of range: {index}")

        elif document is None and index is None:
            raise ValueError("Either document or index is required")
        else:
            raise ValueError("Either document or index must be passed in, not both")

    @staticmethod
    def _compute_dict_multithread(num_threads: int, op: Callable, iterable: Iterable,
                                  op_arg_func= lambda x: x, key_arg_func=lambda x: x) -> dict:
        """Experimental generic multithreading dispatcher and collector to parallelize dictionary construction.

        Args:
            num_threads (int): maximum number of threads (workers) to utilize.
            op: (Callable): operation (function or method) to execute.
            iterable: (Iterable): iterable to call the `op` on each item.
            op_arg_func: a function that maps an item of the `iterable` to an argument for the `op`.
            key_arg_func: a function that maps an item of the `iterable` to the key to use in the resulting dict.

        Returns:
            A dictionary of {key_arg_func(an item of `iterable`): op(p_arg_func(an item of `iterable`))}.

        """
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_keys = {executor.submit(op, op_arg_func(item)): key_arg_func(item) for item in iterable}
            for future in concurrent.futures.as_completed(future_to_keys):
                key = future_to_keys[future]
                try:
                    result[key] = future.result()
                except Exception as e:
                    print(f"Key '{key}' generated exception:", e, file=sys.stderr)
        return result

    @staticmethod
    def _build_index_dict(lst: list) -> dict:
        """Given a list, returns a dictionary of {item from list: index of item}."""
        return {item: index for (index, item) in enumerate(lst)}

