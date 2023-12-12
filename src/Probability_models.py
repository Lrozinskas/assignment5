# import math
# from collections import Counter
#
# # Sample dataset (replace this with your actual dataset)
# data = [
#     {"title": "Watched 120 Shot", "description": "Sample video description"},
#     {"title": "Watched She Is A Star (Original) by Mike Ryu", "description": "Another sample description"},
#     # Add more data entries as needed
# ]
#
# # Function to preprocess text and calculate TF-IDF
# def calculate_tfidf(documents):
#     # Tokenize and preprocess text
#     terms_per_doc = [set(document["title"].lower().split() + document.get("description", "").lower().split()) for document in documents]
#
#     # Calculate term frequency (TF)
#     term_frequency = [Counter(terms) for terms in terms_per_doc]
#
#     # Calculate document frequency (DF)
#     document_frequency = Counter(term for terms in terms_per_doc for term in set(terms))
#
#     # Calculate inverse document frequency (IDF)
#     total_documents = len(documents)
#     idf = {term: math.log(total_documents / (1 + document_frequency[term])) for term in document_frequency}
#
#     # Calculate TF-IDF
#     tfidf = [{term: tf[term] * idf[term] for term in tf} for tf in term_frequency]
#
#     return tfidf
#
# # Function to rank documents using Binary Independence Model (BIM)
# def rank_documents(query, documents, tfidf):
#     query_terms = set(query.lower().split())
#     scores = []
#
#     for i, document in enumerate(documents):
#         document_score = 1.0
#
#         for term in query_terms:
#             # Using binary independence model (BIM)
#             if term in tfidf[i]:
#                 document_score *= tfidf[i][term]
#             else:
#                 document_score *= (1 - tfidf[i].get(term, 0))
#
#         scores.append((i, document_score))
#
#     # Rank documents based on scores
#     ranked_documents = sorted(scores, key=lambda x: x[1], reverse=True)
#     return ranked_documents
#
# # Example usage
# query = "interesting video"
# tfidf_scores = calculate_tfidf(data)
# ranked_results = rank_documents(query, data, tfidf_scores)
#
# # Display the ranked results
# print("Ranked Results:")
# for idx, score in ranked_results:
#     print(f"Document Index: {idx}, Score: {score}")
#     print("Title:", data[idx]["title"])
#     print("Description:", data[idx].get("description", ""))
#     print("=" * 30)



## Iteration 2

# import math
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Sample dataset (replace this with your actual dataset)
# data = [
#     {"title": "Watched 120 Shot", "description": "Sample video description"},
#     {"title": "Watched She Is A Star (Original) by Mike Ryu", "description": "Another sample description"},
#     # Add more data entries as needed
# ]
#
# # Extract text data from the dataset
# corpus = [f"{document['title']} {document.get('description', '')}" for document in data]
#
# # TF-IDF calculation using scikit-learn
# vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(corpus)
#
# # Function to rank documents using cosine similarity
# def rank_documents(query, tfidf_matrix, documents):
#     query_vector = vectorizer.transform([query])
#
#     # Calculate cosine similarity between the query and documents
#     similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
#
#     # Rank documents based on similarities
#     ranked_documents = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
#     return ranked_documents
#
# # Example usage
# query = "interesting video"
# ranked_results = rank_documents(query, tfidf_matrix, data)
#
# # Display the ranked results
# print("Ranked Results:")
# for idx, score in ranked_results:
#     print(f"Document Index: {idx}, Score: {score}")
#     print("Title:", data[idx]["title"])
#     print("Description:", data[idx].get("description", ""))
#     print("=" * 30)


# import json
# import math
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Specify the path to your JSON file
# file_path = 'C:\Users\laroz\Downloads\takeout-20231209T005715Z-001.zip\Takeout\YouTube and YouTube Music\history'
#
# # Read data from the local file
# with open(file_path, 'r') as file:
#     data = json.load(file)
#
# # Extract text data from the dataset
# corpus = [f"{document['title']} {document.get('description', '')}" for document in data]
#
# # TF-IDF calculation using scikit-learn
# vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(corpus)
#
# # Function to rank documents using cosine similarity
# def rank_documents(query, tfidf_matrix, documents):
#     query_vector = vectorizer.transform([query])
#
#     # Calculate cosine similarity between the query and documents
#     similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
#
#     # Rank documents based on similarities
#     ranked_documents = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
#     return ranked_documents
#
# # Example usage
# query = "interesting video"
# ranked_results = rank_documents(query, tfidf_matrix, data)
#
# # Display the ranked results
# print("Ranked Results:")
# for idx, score in ranked_results:
#     print(f"Document Index: {idx}, Score: {score}")
#     print("Title:", data[idx]["title"])
#     print("Description:", data[idx].get("description", ""))
#     print("=" * 30)
#


import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Specify the WSL path to your JSON file
# json_file_path_wsl = r'Ubuntu\home\lrozinskas\CS128\Data_folder\history\watch-history.json'
# json_file_path_wsl = r'\\wsl.localhost\Ubuntu\home\lrozinskas\CS128\Data_folder\history\watch-history.json'
# json_file_path_wsl = '/wsl.localhost/Ubuntu/home/lrozinskas/CS128/Data_folder/history/watch-history.json'
json_file_path_wsl = '/home/lrozinskas/CS128/Data_folder/history/watch-history.json'


# Convert the WSL path to a Windows path
json_file_path_windows = os.path.normpath(json_file_path_wsl)

print("Current Working Directory:", os.getcwd())

# Check if the file exists
if os.path.exists(json_file_path_windows):
    # Read data from the JSON file
    with open(json_file_path_windows, 'r') as json_file:
        data = json.load(json_file)

    # Extract text data from the dataset
    corpus = [f"{document['title']} {document.get('description', '')}" for document in data]

    # TF-IDF calculation using scikit-learn
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Function to rank documents using cosine similarity
    def rank_documents(query, tfidf_matrix, documents):
        query_vector = vectorizer.transform([query])

        # Calculate cosine similarity between the query and documents
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Rank documents based on similarities
        ranked_documents = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return ranked_documents

    # Example usage
    query = "interesting video"
    ranked_results = rank_documents(query, tfidf_matrix, data)

    # Display the ranked results
    print("Ranked Results:")
    for idx, score in ranked_results:
        print(f"Document Index: {idx}, Score: {score}")
        print("Title:", data[idx]["title"])
        print("Description:", data[idx].get("description", ""))
        print("=" * 30)
else:
    print(f"File not found at path: {json_file_path_windows}")
