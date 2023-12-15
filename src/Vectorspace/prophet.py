import matplotlib.pyplot as plt
from collections import Counter
from querying import convert_to_pst, load_watch_history
import calendar


__author__ = "Luke Rozinskas"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Luke Rozinskas", "ChatGPT Ai"]
__license__ = "MIT"
__email__ = "lrozinskas@westmont.edu"

def plot_youtube_traffic(watch_history):
    watch_dates = [convert_to_pst(entry['time']).date() for entry in watch_history]
    month_counts = Counter(date.month for date in watch_dates)

    months = range(1, 13)
    month_abbreviations = [calendar.month_abbr[month] for month in months]

    counts = [month_counts[month] for month in months]

    plt.figure(figsize=(10, 6))
    plt.bar(month_abbreviations, counts, tick_label=month_abbreviations)
    plt.title('YouTube Traffic Across the Year')
    plt.xlabel('Month')
    plt.ylabel('Number of Videos Watched')
    plt.show()

def project_youtube_traffic(watch_history, num_years=2):
    watch_dates = [convert_to_pst(entry['time']).date() for entry in watch_history]
    month_counts = Counter(date.month for date in watch_dates)

    current_year = max(watch_dates).year

    # Calculate average monthly counts from historical data
    average_counts = [month_counts[month] / len(set(date.year for date in watch_dates)) for month in range(1, 13)]

    projected_months = [(current_year + i, month) for i in range(1, num_years + 1) for month in range(1, 13)]
    projected_abbreviations = [calendar.month_abbr[month] for _, month in projected_months]

    # Use distinct colors for historical line and future projection
    historical_counts = [average_counts[month - 1] for _, month in projected_months]

    # Calculate projected counts based on historical averages and dynamic growth factors
    projected_counts = []

    for i, (year, month) in enumerate(projected_months):
        historical_avg = average_counts[month - 1]
        growth_factor = month_counts[month] / (average_counts[month - 1] * len(set(date.year for date in watch_dates)))

        # Apply a dynamic growth factor, but cap it at a reasonable maximum (e.g., 1.5)
        dynamic_growth_factor = min(growth_factor, 1.5)

        # Calculate the projected count based on historical average and dynamic growth factor
        projected_count = historical_avg * (1 + dynamic_growth_factor)

        projected_counts.append(projected_count)

    plt.figure(figsize=(12, 6))
    plt.plot(projected_abbreviations, historical_counts, marker='o', label='Historical Average Counts', linestyle='--', color='blue')
    plt.plot(projected_abbreviations, projected_counts, marker='o', label='Projected Counts', linestyle='-', color='orange')
    plt.title('Projected YouTube Traffic Over the Next 2 Years')
    plt.xlabel('Month')
    plt.ylabel('Number of Videos Watched')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    watch_history_file = '/home/lrozinskas/CS128/Data_folder/history/watch-history.json'
    watch_history = load_watch_history(watch_history_file)

    # Plot YouTube traffic across the year
    plot_youtube_traffic(watch_history)

    # Project YouTube traffic over the next 2 years
    project_youtube_traffic(watch_history)

if __name__ == "__main__":
    main()


