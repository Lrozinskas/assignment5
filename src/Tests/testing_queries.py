import unittest
from datetime import datetime
import io
import mock


# Import the functions and classes from the script
from Vectorspace.querying import load_watch_history, convert_to_pst, get_time_of_day, analyze_watch_times, \
    display_watch_time_analysis, analyze_watch_times_for_query, get_tfidf_scores, get_top_videos, display_top_videos

class TestYourScript(unittest.TestCase):

    def test_load_watch_history(self):
        # Assuming you have a sample watch history JSON file for testing
        file_path = 'path/to/sample/watch-history.json'
        result = load_watch_history(file_path)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(entry, dict) for entry in result))

    def test_convert_to_pst(self):
        # Assuming you have a sample timestamp string for testing
        time_str = "2023-01-01T12:00:00Z"
        result = convert_to_pst(time_str)
        self.assertIsInstance(result, datetime)

    # Add more tests for other functions similarly...

    def test_analyze_watch_times(self):
        # Assuming you have a sample watch history list for testing
        watch_history = [{'time': '2023-01-01T12:00:00Z'}, {'time': '2023-01-01T18:00:00Z'}]
        result = analyze_watch_times(watch_history)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(hour, int) for hour in result))

    # Add more tests for other functions similarly...

    def test_get_time_of_day(self):
        result_morning = get_time_of_day(8)
        result_afternoon = get_time_of_day(14)
        result_evening = get_time_of_day(20)
        self.assertEqual(result_morning, "Morning")
        self.assertEqual(result_afternoon, "Afternoon")
        self.assertEqual(result_evening, "Evening/Night")

    # Add more tests for other functions similarly...

    def test_display_watch_time_analysis(self):
        # Assuming you have a sample watch times list for testing
        watch_times = [8, 12, 16, 20]
        # Redirect stdout to capture the print output for testing
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            display_watch_time_analysis(watch_times)
            output = mock_stdout.getvalue().strip()

        # Assuming your display function prints a title in the output
        self.assertIn("Distribution of Watch Times", output)
        # Add more assertions based on your actual print statements

    # Add more tests for other functions similarly...

if __name__ == '__main__':
    unittest.main()
