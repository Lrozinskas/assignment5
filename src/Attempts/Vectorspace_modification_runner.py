# # import sys
# # import time
# # import pickle
# # import argparse
# # import os
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity
# # import concurrent.futures
# # from datetime import datetime
# # import json
# #
# # from nltk import SnowballStemmer
# #
# # from Vectorspace_modification import Document, Corpus, Vector, analyze_watch_times
# #
# # __author__ = "Mike Ryu"
# # __copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
# # __credits__ = ["Mike Ryu"]
# # __license__ = "MIT"
# # __email__ = "mryu@westmont.edu"
# #
# #
# # def main() -> None:
# #     pars = setup_argument_parser()
# #     args = pars.parse_args()
# #     timer = Timer()
# #
# #     document_processors = (set(), SnowballStemmer('english'))  # Update with your stopwords if needed
# #
# #     try:
# #         with open(args.pickle_file_path, "rb") as pickle_file:
# #             corpus = timer.run_with_timer(pickle.load, [pickle_file], label="corpus load from pickle")
# #     except FileNotFoundError:
# #         # Read data from the JSON file
# #         with open(args.json_file_path_windows, 'r') as json_file:
# #             data = json.load(json_file)
# #
# #         # Extract text data and timestamps from the dataset
# #         corpus_documents = [Document(title=document['title'], words=document.get('description', ''),
# #                                      timestamp=document['time'], processors=document_processors)
# #                             for document in data]
# #
# #         corpus = timer.run_with_timer(Corpus, [corpus_documents, args.num_threads, args.debug],
# #                                       label="corpus instantiation (includes TF-IDF matrix)")
# #
# #         with open(args.pickle_file_path, "wb") as pickle_file:
# #             pickle.dump(corpus, pickle_file)
# #
# #     # Use the existing querying function
# #     keep_querying(corpus, document_processors, 10)
# #
# #
# # def setup_argument_parser() -> argparse.ArgumentParser:
# #     pars = argparse.ArgumentParser(prog="python3 -m your_module_name")  # Replace with your actual module name
# #     pars.add_argument("num_threads", type=int, help="required integer indicating how many threads to utilize")
# #     pars.add_argument("json_file_path_windows", type=str, help="required string containing the path to a JSON file")
# #     pars.add_argument("pickle_file_path", type=str, help="required string containing the path to a pickle (data) file")
# #     pars.add_argument("-d", "--debug", action="store_true", help="flag to enable printing debug statements to console output")
# #     return pars
# #
# #
# # def keep_querying(corpus: Corpus, processors: tuple[set[str], SnowballStemmer], num_results: int) -> None:
# #     again_response = 'y'
# #
# #     while again_response == 'y':
# #         raw_query = input("Your query? ")
# #         query_document = Document(title="query", words=raw_query.split(), processors=processors)
# #         query_vector = corpus.compute_tf_idf_vector(doc=query_document)
# #
# #         query_result = {}
# #         for title, doc_vector in corpus.tf_idf.items():
# #             query_result[title] = doc_vector.cossim(query_vector)
# #
# #         display_query_result(raw_query, query_result, corpus, num_results)
# #         again_response = input("Again (y/N)? ").lower()
# #
# #
# # def display_query_result(query: str, query_result: dict, corpus: Corpus, num_results) -> None:
# #     if num_results > len(corpus):
# #         num_results = len(corpus)
# #
# #     sorted_result = sorted([(title, score) for title, score in query_result.items()],
# #                            key=lambda item: item[1], reverse=True)
# #
# #     print(f"\nFor query: {query}")
# #     for i in range(num_results):
# #         title, score = sorted_result[i]
# #         print(f"Result {i + 1:02d} : [{score:0.6f}] {title}")
# #     print()
# #
# #
# # class Timer:
# #     def __init__(self):
# #         self._start = 0.0
# #         self._stop = 0.0
# #
# #     def run_with_timer(self, op, op_args=None, label="operation"):
# #         if not op_args:
# #             op_args = []
# #
# #         self.start()
# #         result = op(*op_args)
# #         self.stop()
# #
# #         self.print_elapsed(label=label)
# #         return result
# #
# #     def print_elapsed(self, label: str = "operation", file=sys.stdout):
# #         print(f"Elapsed time for {label}: {self.get_elapsed():0.4f} seconds", file=file)
# #
# #     def get_elapsed(self) -> float:
# #         return self._stop - self._start
# #
# #     def start(self) -> None:
# #         self._start = time.time()
# #
# #     def stop(self) -> None:
# #         self._stop = time.time()
# #
# #
# # if __name__ == '__main__':
# #     main()
#









import sys
import time
import pickle
import argparse
import json
from nltk import SnowballStemmer

from Attempts.Vectorspace_modification import Document, Corpus, Vector, analyze_watch_times

__author__ = "Mike Ryu"
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Mike Ryu"]
__license__ = "MIT"
__email__ = "mryu@westmont.edu"


def main() -> None:
    pars = setup_argument_parser()
    args = pars.parse_args()
    timer = Timer()

    document_processors = (set(), SnowballStemmer('english'))  # Update with your stopwords if needed

    try:
        with open(args.pickle_file_path, "rb") as pickle_file:
            corpus = timer.run_with_timer(pickle.load, [pickle_file], label="corpus load from pickle")
    except FileNotFoundError:
        # Read data from the JSON file
        with open(args.json_file_path_windows, 'r') as json_file:
            data = json.load(json_file)

        # Extract text data and timestamps from the dataset
        corpus_documents = [Document(title=document['title'], words=document.get('description', ''),
                                     timestamp=document['time'], processors=document_processors)
                            for document in data]

        corpus = timer.run_with_timer(Corpus, [corpus_documents, args.num_threads, args.debug],
                                      label="corpus instantiation (includes TF-IDF matrix)")

        with open(args.pickle_file_path, "wb") as pickle_file:
            pickle.dump(corpus, pickle_file)

    # Use the existing querying function
    keep_querying(corpus, document_processors, 10)

    # Utilize the Vector class
    example_vector = Vector([1.0, 2.0, 3.0])
    print("Example Vector:", example_vector.vec)

    # Utilize analyze_watch_times
    analyze_watch_times(corpus, "soccer")


def setup_argument_parser() -> argparse.ArgumentParser:
    pars = argparse.ArgumentParser(prog="python3 -m your_module_name")  # Replace with your actual module name
    pars.add_argument("num_threads", type=int, help="required integer indicating how many threads to utilize")
    pars.add_argument("json_file_path_windows", type=str, help="required string containing the path to a JSON file")
    pars.add_argument("pickle_file_path", type=str, help="required string containing the path to a pickle (data) file")
    pars.add_argument("-d", "--debug", action="store_true", help="flag to enable printing debug statements to console output")
    return pars


def keep_querying(corpus: Corpus, processors: tuple[set[str], SnowballStemmer], num_results: int) -> None:
    again_response = 'y'

    while again_response == 'y':
        raw_query = input("Your query? ")
        query_document = Document(title="query", words=raw_query.split(), processors=processors)
        query_vector = corpus.compute_tf_idf_vector(doc=query_document)

        query_result = {}
        for title, doc_vector in corpus.tf_idf.items():
            query_result[title] = doc_vector.cossim(query_vector)

        display_query_result(raw_query, query_result, corpus, num_results)
        again_response = input("Again (y/N)? ").lower()


def display_query_result(query: str, query_result: dict, corpus: Corpus, num_results) -> None:
    if num_results > len(corpus):
        num_results = len(corpus)

    sorted_result = sorted([(title, score) for title, score in query_result.items()],
                           key=lambda item: item[1], reverse=True)

    print(f"\nFor query: {query}")
    for i in range(num_results):
        title, score = sorted_result[i]
        print(f"Result {i + 1:02d} : [{score:0.6f}] {title}")
    print()


class Timer:
    def __init__(self):
        self._start = 0.0
        self._stop = 0.0

    def run_with_timer(self, op, op_args=None, label="operation"):
        if not op_args:
            op_args = []

        self.start()
        result = op(*op_args)
        self.stop()

        self.print_elapsed(label=label)
        return result

    def print_elapsed(self, label: str = "operation", file=sys.stdout):
        print(f"Elapsed time for {label}: {self.get_elapsed():0.4f} seconds", file=file)

    def get_elapsed(self) -> float:
        return self._stop - self._start

    def start(self) -> None:
        self._start = time.time()

    def stop(self) -> None:
        self._stop = time.time()


if __name__ == '__main__':
    main()




# def main():
#     # Specify the path to your JSON file
#     json_file_path_wsl = '/home/lrozinskas/CS128/Data_folder/history/watch-history.json'
#
#     # Convert the WSL path to a Windows path
#     json_file_path_windows = os.path.normpath(json_file_path_wsl)
#
#     # Check if the file exists
#     if os.path.exists(json_file_path_windows):
#         # Read data from the JSON file
#         with open(json_file_path_windows, 'r') as json_file:
#             data = json.load(json_file)
#
#         # Create a list of Document objects
#         documents = [
#             Document(title=document['title'], words=document.get('description', ''), timestamp=document['time'])
#             for document in data]
#
#         # Create an instance of the Corpus class
#         corpus = Corpus(documents=documents)
#
#         while True:
#             # Get user input for the query
#             query = input("Enter your query (or 'exit' to quit): ")
#
#             if query.lower() == 'exit':
#                 break
#
#             # Example of analyzing watch times for the user-input query
#             analyze_watch_times(corpus, query)
#
#             # Example usage for ranking documents based on user-input query
#             ranked_results = rank_documents(query, corpus)
#
#             # Display the ranked results
#             print("Ranked Results:")
#             for idx, score in ranked_results:
#                 print(f"Document Index: {idx}, Score: {score}")
#                 print("Title:", corpus[idx].title)
#                 print("Description:", corpus[idx].words)
#                 print("=" * 30)
#
#     else:
#         print(f"File not found at path: {json_file_path_windows}")
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#

