import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from typing import Union
import time
import random
import os
from PLA.toolkit.utils import generate_timestamp_random_id
import pickle

class Thermostat:
    def __init__(self, name: str,
                
                 is_heating: bool = False,
                 is_cooling: bool = False) -> None:
        """
        Initializes the Thermostat with default settings and loads an optional schedule and history from CSV files.
        Args:
            schedule_path: Optional path to the CSV file containing the heating/cooling schedule.
            history_path: Optional path to the CSV file containing the operation history.
        """
        self.name = name
        
        
        self.is_heating: bool = is_heating
        self.is_cooling: bool = is_cooling
        path = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Weather/weather_new.pkl"
        with open(path, 'rb') as file:
            self.weather_data = pickle.load(file)

        

    

    
    def set_temperature_and_humidity_in_home(self, temperature: Union[float, str] = None, humidity: Union[float, str] = None) -> dict:
        """
        Sets the current temperature and humidity of the thermostat.
        Args:
            temperature: Desired temperature in Celsius.
            humidity: Desired humidity in percentage.
        Returns:
            Status message indicating the temperature and humidity have been set.
        """
        
        if temperature != None and humidity is None:
            return {"status": "success", 'message': f"Current temperature set to {temperature}°C."}
        elif temperature == None and humidity is not None:
            {"status": "success", 'message': f"Current humidity set to {humidity}%."}
        elif temperature == None and humidity == None:
            {"status": "success", 'message': f"The temperature and humidity is not set."}
        return {"status": "success", 'message': f"Current temperature set to {temperature}°C and current humidity set to {humidity}%."}

        
    
    def get_current_temperature(self, at_time: str) -> float:
        """
        Retrieves the current temperature based on the specified time.
        Args:
            at_time: The specific time for which to retrieve the temperature.
        Returns:
            The current temperature at the specified time.
        """
        try:
            c_time = datetime.strptime(at_time, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            return {"status": "error", "message": f"the at_time format not right: {str(e)}"}
        
        data = self.weather_data["New York"][at_time.split(" ")[0]]["forecast"]["forecastday"][0]["hour"]
        # print(data)
        current_d = None
        for d in data:
            if datetime.strptime(d["time"], "%Y-%m-%d %H:%M") < c_time:
                current_d = d
            else:
                break
        
        temperature = current_d['temp_c'] + 2
        
        return temperature 

    def get_current_humidity(self, at_time: str) -> float:
        """
        Retrieves the current humidity based on the specified time.
        Args:
            at_time: The specific time for which to retrieve the humidity.
        Returns:
            The current humidity at the specified time.
        """
        try:
            c_time = datetime.strptime(at_time, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            return {"status": "error", "message": f"the at_time format not right: {str(e)}"}
        
        data = self.weather_data["New York"][at_time.split(" ")[0]]["forecast"]["forecastday"][0]["hour"]
        current_d = None
        for d in data:
            if datetime.strptime(d["time"], "%Y-%m-%d %H:%M") < c_time:
                current_d = d
            else:
                break
        humidity = current_d['humidity'] + 3
        
        return humidity 
    
    def get_home_temperature_and_humidity(self, at_time: str) -> dict:
        """
        Get and monitor the current temperature and humidity levels.
        Args:
            at_time: The specific time for which to retrieve the temperature.
        Returns:
            A string with the current temperature and humidity.
        """
        return {"status": "success", 'message': (f"Current Temperature: {self.get_current_temperature(at_time)}°C, "
                f"Current Humidity: {self.get_current_humidity(at_time)}%.")}
    
    # def save_schedule(self, path: str) -> None:
    #     """
    #     Saves the schedule to a CSV file.
    #     Args:
    #         path: Path to the CSV file where the schedule will be saved.
    #     """
    #     pd.DataFrame(self.schedule).to_csv(path, index=False)

    def save_history(self, path: str) -> None:
        """
        Saves the operation history to a CSV file.
        Args:
            path: Path to the CSV file where the history will be saved.
        """
        pd.DataFrame(self.history).to_csv(path, index=False)

    def save(self) -> None:
        # self.save_schedule(self.schedule_path)
        self.save_history(self.history_path)

if __name__ == '__main__':
    thermostat = Thermostat(name="Jane_Doe")
    
    # print(thermostat.set_temperature(22.0))  # Set temperature
    # print(thermostat.set_humidity(45.0))  # Set humidity
    print(thermostat.get_home_temperature_and_humidity("2024-09-02 14:00:01"))  # Monitor current status
    # print(thermostat.schedule_cycle("2024-09-02 14:00:01", 30, "heat"))  # Schedule heating
    # print(thermostat.get_home_temperature_and_humidity())  # Monitor current status

    # Print operation history
    # print("Operation History:")
    # for record in thermostat.get_history():
    #     print(record)

