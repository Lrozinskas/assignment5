# Assignment 5: Youtube Data Analysis
**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

*Assistant Professor* Mike Ryu (mryu@westmont.edu) 

## Autor Information
* **Name(s)**: Luke Rozinskas
* **Email(s)**: lrozinskas@westmont.edu

## Usage
1. Download user watch data from Google Takeout as JSON file
2. Clone the git hub repository and replace relevant links to the JSON data with links to your own data
3. Enjoy seeing some of what you've watched on Youtube!

## Project Goal
Implement a program to give a semi-comprehensive analysis of user's Youtube watch history data. Utilized different techniques such as Tf-idf matrices, frequency counters, and LDA models to generate insight and get relevant information for the user's activity.

## Implementation: The Methods
### General Analysis
Gives user, via a jupyter notebook file, some general data about their history. How many videos they've watched, time from the 1st to most recent video watched, their most watched video and creator, etc.

### Vectorspace
#### Querying
This file uses tf-idf analysis to look at title of videos you've watched and get a score for their relevance. This is via the methods get_tfidf_scores() and get_top_videos(), and then they are displayed and filtered through in the methods display_top_videos(), display_engagement_analysis().

We also take into account times the videos are watched and use the methods convert_to_pst(), get_time_of_day(), analyze_watch_times(), display_watch_time_analysis(), analyze_watch_times_for_query() to do so during the querying.

#### Prophet
This file take the matplotlib library and utilizes its functionalities produce some graphs/plots of the user's watching frequency over the time of their data. I use counts and average frequency of traffic to then project the watch traffic over the next two years. 

#### Topicanalysis
This file utilizes a jupyter notebook and the pyLDAvis.gensim_models to look over the data we take in (the terms from video titles) and creates an intertopic distance map with principal componenet axis to plot the distributions. 


