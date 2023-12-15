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


## Credits
#### General Analysis
Code taken and adapted from Bjoernpl. Project at https://github.com/bjoernpl/YoutubeHistoryVisualizer

#### Querying
Code written with and adapted from ChatGPT Ai from prompt: "From my Youtube Watch History from Google Takeout, help me to create a program that will take in a user query and output 10 most relevant videos from tf-idf scores given the titles of the videos. However, I want this to be combined with the watch times of the videos (that is, when the videos were watched) in order to tell the program what time I intend to watch a video and then give me back the most relevant videos that would be correlated with the time of the videos in the history"

#### Prophet
Code written with and adatped from ChatGPT Ai from prompt: "I want to take a look at my Youtube Watch History from Google Takeout and get two things: 1) I want to see a graphical distribution of my Youtube traffic from over the entirety of my data, and 2) I want to make a projection, based upon the history that I have, over the next two years at what my Youtube traffic might be"

#### TopicAnalysis
Code written with and adapted from ChatGPT Ai from prompt: "I am looking at my Youtube Watch History from Google Takeout and have a topic analysis, based on something like a cluster from the data. In other words, I want to create a program that will give me relevant information about the distribution of content topics I watch. Use whatever library that you would like that would help with feedback and analysis. Additionally, make sure to utilize the "title" key for this program as that is where the relevant information about the videos lay."

