import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime
import pytz

def load_watch_history(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def convert_to_pst(time_str):
    utc_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    utc_time = pytz.utc.localize(utc_time)
    pst_time = utc_time.astimezone(pytz.timezone("America/Los_Angeles"))
    return pst_time

def get_tfidf_matrix(titles, vectorizer):
    tfidf_matrix = vectorizer.fit_transform(titles)
    return tfidf_matrix

def cluster_analysis(titles, vectorizer, num_clusters=5):
    tfidf_matrix = get_tfidf_matrix(titles, vectorizer)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)

    return clusters

def display_cluster_analysis(clusters, watch_history):
    unique_clusters = set(clusters)
    print("\nCluster Analysis:")
    for cluster_num in unique_clusters:
        cluster_videos = [watch_history[i] for i in range(len(clusters)) if clusters[i] == cluster_num]
        print(f"\nCluster {cluster_num + 1} - {len(cluster_videos)} videos:")
        for video in cluster_videos[:min(10, len(cluster_videos))]:
            title = video.get('title', 'N/A')
            time_str = video.get('time', 'N/A')
            watched_time = convert_to_pst(time_str).strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"  - Title: {title}, Watched Time: {watched_time}")

def main():
    watch_history_file = '/home/lrozinskas/CS128/Data_folder/history/watch-history.json'
    watch_history = load_watch_history(watch_history_file)

    titles = [entry['title'] for entry in watch_history]
    vectorizer = TfidfVectorizer(stop_words='english')

    clusters = cluster_analysis(titles, vectorizer, num_clusters=5)
    display_cluster_analysis(clusters, watch_history)

if __name__ == "__main__":
    main()

