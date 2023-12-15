from sklearn.metrics.pairwise import linear_kernel
from prettytable import PrettyTable

import json
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pytz

def load_watch_history(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# def convert_to_pst(time_str):
#     utc_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
#     utc_time = pytz.utc.localize(utc_time)
#     pst_time = utc_time.astimezone(pytz.timezone("America/Los_Angeles"))
#     return pst_time

def convert_to_pst(time_str):
    # Handle milliseconds in the timestamp
    try:
        utc_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        utc_time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")

    utc_time = pytz.utc.localize(utc_time)
    pst_time = utc_time.astimezone(pytz.timezone("America/Los_Angeles"))
    return pst_time

def get_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    else:
        return "Evening/Night"



def analyze_watch_times(watch_history):
    watch_times = [convert_to_pst(entry['time']).hour for entry in watch_history]
    return watch_times

def display_watch_time_analysis(watch_times):
    plt.figure(figsize=(10, 6))
    sns.histplot(watch_times, bins=24, kde=True)
    plt.title('Distribution of Watch Times')
    plt.xlabel('Hour of the Day (PST)')
    plt.ylabel('Frequency')
    plt.xticks(range(24), [f"{hour:02}:00" for hour in range(24)])
    plt.show()



def analyze_watch_times_for_query(query, top_videos):
    print(f"\nWatch Time Analysis for Query: {query}")

    watch_times = [convert_to_pst(video['time']).hour for video, score in top_videos]
    total_watch_time_minutes = sum(watch_times) * 60  # Convert hours to minutes
    avg_watch_time_minutes = total_watch_time_minutes / len(watch_times) if watch_times else 0

    avg_hours = int(avg_watch_time_minutes // 60)
    avg_minutes = int(avg_watch_time_minutes % 60)

    print(f"Average Watch Time for Query '{query}': {avg_hours}:{avg_minutes:02} (PST)")
    print(f"Most Common Watch Time for Query '{query}': {get_time_of_day(max(set(watch_times), key=watch_times.count))}")


def get_tfidf_scores(titles, vectorizer):
    tfidf_matrix = vectorizer.fit_transform(titles)
    return tfidf_matrix

def get_top_videos(query, watch_history, vectorizer, tfidf_matrix, top_n=10):
    query_vector = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
    top_video_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    top_videos = [(watch_history[i], cosine_similarities[i]) for i in top_video_indices]
    return top_videos


def display_top_videos(top_videos):
    table = PrettyTable()
    table.field_names = ["#", "Title", "URL", "Watched On", "Relevance Score"]

    for i, (video, score) in enumerate(top_videos, start=1):
        title = video.get('title', 'N/A')
        title_url = video.get('titleUrl', 'N/A')
        watched_on = video.get('time', 'N/A')
        relevance_score = f"{score:.4f}"

        table.add_row([i, title, title_url, watched_on, relevance_score])

    print("\nTop Videos:")
    print(table)




def analyze_engagement(watch_history):
    video_count = {}
    channel_count = {}

    for entry in watch_history:
        video_id = entry.get('titleUrl', 'Unknown Video ID')
        video_title = entry.get('title', 'Unknown Video Title')

        # Counting video views
        video_count[video_title] = video_count.get(video_title, 0) + 1

        # Counting channel views if 'subtitles' is present
        if 'subtitles' in entry:
            channel_id = entry['subtitles'][0].get('url', 'Unknown Channel ID')
            channel_name = entry['subtitles'][0].get('name', 'Unknown Channel Name')
            channel_count[channel_name] = channel_count.get(channel_name, 0) + 1

    return video_count, channel_count

def display_engagement_analysis(video_count, channel_count):
    video_table = PrettyTable()
    video_table.field_names = ["#", "Video ID", "Watch Count"]

    for i, (video_id, count) in enumerate(sorted(video_count.items(), key=lambda x: x[1], reverse=True)[:10], start=1):
        video_table.add_row([i, video_id, count])

    channel_table = PrettyTable()
    channel_table.field_names = ["#", "Channel ID", "Watch Count"]

    for i, (channel_id, count) in enumerate(sorted(channel_count.items(), key=lambda x: x[1], reverse=True)[:10], start=1):
        channel_table.add_row([i, channel_id, count])

    print("\nEngagement Analysis:")
    print("\nMost-watched Videos:")
    print(video_table)

    print("\nMost-watched Channels:")
    print(channel_table)




# def main():
#     watch_history_file = '/home/lrozinskas/CS128/Data_folder/history/watch-history.json'
#     watch_history = load_watch_history(watch_history_file)
#
#     titles = [entry['title'] for entry in watch_history]
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = get_tfidf_scores(titles, vectorizer)
#
#     while True:
#         user_query = input("Enter your query (or 'exit' to stop): ")
#         if user_query.lower() == 'exit':
#             break
#
#         top_videos = get_top_videos(user_query, watch_history, vectorizer, tfidf_matrix)
#         display_top_videos([(video, score) for video, score in top_videos])
#
#         analyze_watch_times_for_query(user_query, top_videos)
#
#     # Analyze engagement
#     video_count, channel_count = analyze_engagement(watch_history)
#     display_engagement_analysis(video_count, channel_count)
#
# if __name__ == "__main__":
#     main()


def main():
    watch_history_file = '/home/lrozinskas/CS128/Data_folder/history/watch-history.json'
    watch_history = load_watch_history(watch_history_file)

    titles = [entry['title'] for entry in watch_history]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = get_tfidf_scores(titles, vectorizer)

    while True:
        user_query = input("Enter your query (or 'exit' to stop): ")
        if user_query.lower() == 'exit':
            break

        user_time_input = input("Enter the time you'd like to watch the video (format HH:mm am/pm): ")
        try:
            user_time = datetime.strptime(user_time_input, "%I:%M %p").time()
        except ValueError:
            print("Invalid time format. Please enter the time in the format HH:mm am/pm.")
            continue

        top_videos = get_top_videos(user_query, watch_history, vectorizer, tfidf_matrix)
        filtered_top_videos = []

        for video, score in top_videos:
            video_time = convert_to_pst(video['time']).time()
            if video_time <= user_time:
                filtered_top_videos.append((video, score))

            if len(filtered_top_videos) == 10:
                break

        display_top_videos(filtered_top_videos)
        analyze_watch_times_for_query(user_query, filtered_top_videos)

    # Analyze engagement
    video_count, channel_count = analyze_engagement(watch_history)
    display_engagement_analysis(video_count, channel_count)

if __name__ == "__main__":
    main()

