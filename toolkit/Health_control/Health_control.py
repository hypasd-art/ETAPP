import pandas as pd
from typing import List, Union
from datetime import datetime, timedelta
import os

class HealthMonitoringApp:
    def __init__(self, name: str, health_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Health/Health_records_{}.csv",
                 workout_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Health/Workout_records_{}.csv",
                 health_summary_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Health/Health_summary_records_{}.csv",) -> None:
        """
        Initializes the Health Monitoring App with a user's name and a health data database.

        Args:
            name: The name of the user.
            database: List of dictionaries representing health records.
        """
        self.name = name
        self.health_path = health_path.format(self.name)
        self.workout_path = workout_path.format(self.name)
        self.health_summary_path = health_summary_path.format(self.name)

        # Initialize an empty DataFrame to store health records.
        self.health_df = pd.DataFrame()
        self.workout_df = pd.DataFrame()
        self.health_summary_df = pd.DataFrame()
        self.workout_summary_df = pd.DataFrame()
        # Load health records from a CSV file if a path is provided.
        if health_path:
            self.load_health_records(self.health_path, self.health_summary_path)
        if workout_path:
            self.load_workout_records(self.workout_path)

    def load_health_records(self, health_path: str, health_summary_path: str) -> None:
        """
        Loads health records from a specified CSV file into a DataFrame.
        Args:
            path: Path to the CSV file containing health records.
        """
        # Read the CSV file into the DataFrame.
        self.health_df = pd.read_csv(health_path)
        self.health_df['timestamp'] = pd.to_datetime(self.health_df['timestamp'])
        self.health_summary_df = pd.read_csv(health_summary_path)

    def load_workout_records(self, path: str) -> None:
        """
        Loads health records from a specified CSV file into a DataFrame.
        Args:
            path: Path to the CSV file containing health records.
        """
        # Read the CSV file into the DataFrame.
        self.workout_df = pd.read_csv(path)
        
        

    
    
    
    def get_current_health_and_mood_status(self) -> dict:
        """
        Retrieves the current health and mood status of the user.
        Returns:
            A dictionary containing the status and the most recent health and mood record before the specified timestamp.
        """
        # Convert the start and end times to datetime objects.
        at_time = os.environ.get("CURRENT_TIME")
        at_time = datetime.strptime(at_time, '%Y-%m-%d %H:%M:%S')
        self.health_df['timestamp'] = pd.to_datetime(self.health_df['timestamp'])
        mask = self.health_df['timestamp'] <= at_time
        closest_record = self.health_df.loc[mask].iloc[-1:] if not self.health_df.loc[mask].empty else pd.Series()
    
        if not closest_record.empty:
            closest_record['timestamp'] = closest_record['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            closest_record = closest_record.to_dict('records')
        else:
            closest_record = []  
        
        return {"status": "success", "data": closest_record}
    
    def get_recent_health_and_mood_summary(self, time: str) -> dict:
        """
        Retrieves the most recent health and mood records after the specified timestamp until now.
        Args:
            time: The timestamp after which to retrieve the health and mood summary record. The format should be '%Y-%m-%d %H:%M:%S'.
        Returns:
            A dictionary containing the status of the operation and the most recent health and mood records.
        """
        at_time = os.environ.get("CURRENT_TIME")
        at_time = datetime.strptime(at_time, '%Y-%m-%d %H:%M:%S')
        time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        if time >= at_time:
            return {"status": "failure", "message": "The input time should be before the current time"}
        self.health_summary_df['date'] = pd.to_datetime(self.health_summary_df['date'])
        mask = (self.health_summary_df['date'] <= at_time) & (self.health_summary_df['date'] >= time)
        closest_record = self.health_summary_df.loc[mask].copy() #  if not self.health_summary_df.loc[mask].empty else pd.Series()
        
        closest_record = closest_record.sort_values(by="date", ascending=False)

        if not closest_record.empty:
            closest_record['date'] = closest_record['date'].dt.strftime('%Y-%m-%d %H:%M:%S') # .to_dict('records')
            closest_record = closest_record.to_dict('records')
        else:
            closest_record = []  
        return {"status": "success", "data": closest_record}
    
    
    
    def get_user_recent_workout_records(self, time: str) -> dict:
        """
        Retrieves the most recent workout record after the specified timestamp until now.
        Args:
            time: The timestamp after which to retrieve the workout record. The format should be '%Y-%m-%d %H:%M:%S'.
        Returns:
            A dictionary containing the status of the operation and the most recent workout record.
        """
        at_time = os.environ.get("CURRENT_TIME")
        at_time = datetime.strptime(at_time, '%Y-%m-%d %H:%M:%S')
        time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

        # Filter the DataFrame for records within the specified time range.
        self.workout_df['Start Time'] = pd.to_datetime(self.workout_df['Start Time'])
        asof_record = self.workout_df[(self.workout_df['Start Time']>=time) & (self.workout_df['Start Time'] <= at_time)]
        if not asof_record.empty:
            asof_record['Start Time'] = asof_record['Start Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            

            closest_record = asof_record.to_dict('records')
            
        else:
            closest_record = []  
        return {"status": "success", "data": closest_record}


if __name__ == '__main__':
    health_app = HealthMonitoringApp(name="James_Harrington")
    os.environ["CURRENT_TIME"] = "2024-09-03 18:00:00"
    health_app.get_current_health_and_mood_status()
    print(health_app.get_user_recent_workout_records("2024-09-01 00:00:00"))
    print(health_app.get_recent_health_and_mood_summary("2024-09-01 00:00:00"))
    
