import pandas as pd
from datetime import datetime, timedelta
import os
from typing import List, Optional
import os
import pickle
import copy

class WeatherData:
    def __init__(self, name, data_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Weather/weather_new.pkl") -> None:
        """
        Initializes the WeatherData system with a path to a CSV file containing weather data.
        Args:
            name: The name of the user.
            data_path: The path to the CSV file containing weather data.
        """
        self.name = name
        self.data_path = data_path
        # Load the weather data from a CSV file if a path is provided.
        if data_path:
            self.load_data(data_path)
        else:
            self.data_df = pd.DataFrame()

    def load_data(self, path: str) -> None:
        """
        Loads weather data from a specified CSV file into a DataFrame.
        Args:
            path: Path to the CSV file containing weather data.
        """
        with open(path, 'rb') as file:
            self.weather_data = pickle.load(file)

    def get_today_weather(self, location: str) -> dict:
        """
        Retrieves the today's weather data based on location.
        Args:
            location: the location for retieving the weather data.
        Returns:
            A dictionary containing the status and today's weather data.
        """
        at_time = os.environ.get("CURRENT_DATE", None)
        # current_time = datetime.strptime(at_time, '%Y-%m-%d %H:%M:%S')
        try:
            datetime.strptime(at_time, "%Y-%m-%d")
        except Exception as e:
            return {"status": "error", "message": f"the at_time format not right: {str(e)}"}
        # Check if the location exists in the weather data
        if location not in self.weather_data.keys():
            return {"status": "error", "message": "Location not found"}

        # Check if the at_time exists in the location's weather data
        if at_time not in self.weather_data[location].keys():
            return {"status": "error", "message": "Time not found for the given location"}
        
        data = copy.copy(self.weather_data[location][at_time]["forecast"]["forecastday"][0])
        h = data.pop("hour")
        return {"status": "success", "data": data}
        # return self._filter_by_time(current_time, current_time)

    

    def get_future_weather(self, location: str, start_time: str, end_time: str) -> dict:
        """
        Retrieves the location's weather for a future timestamp.
        Args:
            location: the location for retieving the weather data.
            start_time: The start timestamp for the future weather(format: 'YYYY-MM-DD', e.g. 2024-05-28).
            end_time: The end timestamp for the future weather(format: 'YYYY-MM-DD', e.g. 2024-05-29).
        Returns:
            A dictionary containing the status and the future weather data.
        """
        # Parse the at_time string into a datetime object
        current_time = datetime.strptime(start_time, '%Y-%m-%d')

        # Check if the location exists in the weather data
        if location not in self.weather_data:
            return {"status": "error", "message": "Location not found"}

        # Check if the at_time exists in the location's weather data
        if start_time not in self.weather_data[location]:
            return {"status": "error", "message": "start_Time not found for the given location"}
        all_data = []

        while current_time <= datetime.strptime(end_time, '%Y-%m-%d'):
            date_str = current_time.strftime('%Y-%m-%d')
            if date_str not in self.weather_data[location]:
                return {"status": "success", "data": all_data, "message": f"The Weather after date {date_str} is not accessible"}
            # Retrieve the weather data for the given location and time
            data = self.weather_data[location][date_str]["forecast"]["forecastday"][0]

            # Remove the 'hour' key as it's not needed in the output
            data.pop("hour", None)  # Use .pop() with a default value to avoid KeyError if 'hour' key is missing
            all_data.append(data)
            current_time += timedelta(days=1)
                
            return {"status": "success", "data": all_data}

        

    

if __name__ == "__main__":
    # Create a WeatherData object and load data from a CSV file
    weather_data = WeatherData("")
    os.environ['CURRENT_DATE'] = "2024-09-06"
    # Get the current weather
    print(weather_data.get_today_weather("Chicago"))
    print(weather_data.get_future_weather("New York", "2024-09-09", "2024-09-10"))
    