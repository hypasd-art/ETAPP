import pandas as pd
from typing import List, Union, Dict
from datetime import datetime, timedelta
import time
from PLA.toolkit.utils import generate_timestamp_random_id
import random
import os


class Calendar:
    def __init__(self, name: str, 
                    events_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Calendar/events/events_{}.csv", 
                    alarms_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Calendar/alarms/alarms_{}.csv") -> None:
        """
        Initializes the Calendar with a personal calendar database.

        Args:
            name: The user's name.
            events_path: Optional path to the CSV file containing calendar events.
            alarms_path: Optional path to the CSV file containing alarms.
        """
        self.name = name
        self.events_path = events_path.format(self.name)
        self.alarms_path = alarms_path.format(self.name)
        self.events_df = pd.DataFrame()
        self.alarms_df = pd.DataFrame()  # DataFrame to store alarms
        if events_path:
            self.load_events(events_path.format(self.name))
        if alarms_path:
            self.load_alarms(alarms_path.format(self.name))
            
    def load_events(self, path: str) -> None:
        """
        Loads events data from the specified CSV file.

        Args:
            path: Path to the events data file.
        """
        self.events_df = pd.read_csv(path)
        self.events_df['start_time'] = pd.to_datetime(self.events_df['start_time'])
        self.events_df['end_time'] = pd.to_datetime(self.events_df['end_time'])
        
    
    def load_alarms(self, path: str) -> None:
        """
        Loads alarms data from the specified CSV file and stores it in a DataFrame.

        Args:
            path: Path to the CSV file containing alarms data.
        """
        self.alarms_df = pd.read_csv(path)

    def add_event_in_calendar(self, title: str, description: str, start_time: str, 
                  end_time: str, reminder: str = None) -> dict:
        """
        Adds a new event to the calendar.
        
        Args:
            title: The title of the event.
            description: A brief description of the event.
            start_time: The start time of the event, should in the format: %Y-%m-%d %H:%M:%S, e.g. 2024-05-28 15:30:00.
            end_time: The end time of the event, should in the format: %Y-%m-%d %H:%M:%S, e.g. 2024-05-28 15:30:00.
            reminder: The time of the event when a reminder should be set(should in the format: %Y-%m-%d %H:%M:%S, e.g. 2024-05-28 18:30:00.), or None if no reminder is needed.
        
        Returns:
            A success message indicating the event has been added or error message.
        
        """
        date_format = '%Y-%m-%d %H:%M:%S'
        start_time = datetime.strptime(start_time, date_format)
        end_time = datetime.strptime(end_time, date_format)

        reminder = datetime.strptime(reminder, date_format) if reminder is not None else None

        if end_time <= start_time:
            raise ValueError('End time must be after start time.')
        new_event = pd.DataFrame({
            'id': [generate_timestamp_random_id()],
            'title': [title],
            'description': [description],
            'start_time': [start_time],
            'end_time': [end_time],
            'reminder': [reminder]
        })
        if self.is_conflicting(new_event):
            # raise Exception('The new event is conflicting with other events in Calendar')
            return {"status": "failure", 'error message': "The new event is conflicting with other events in Calendar."}
        self.events_df = pd.concat([self.events_df, new_event], ignore_index=True)
        return {"status": "success", 'message': "Event added successfully."}

    def view_today_events_in_calendar(self) -> dict:
        """
        Get all today's events in the calendar.

        Returns:
            A dictionary contains the status and a list of today's event in details.
        """
        date = os.environ.get("CURRENT_DATE", None)
        self.events_df['start_time'] = self.events_df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        self.events_df['end_time'] = self.events_df['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        event = self.events_df[self.events_df['start_time'].str.contains(date)].to_dict('records')
        self.events_df['start_time'] = pd.to_datetime(self.events_df['start_time'])
        self.events_df['end_time'] = pd.to_datetime(self.events_df['end_time'])
        return {"status": "success", "data": event}
    
    def view_events_in_calendar_by_providing_time_range(self, start_time: str, end_time: str) -> dict:
        """
        Retrieves events that start within a specified time range from the calendar.

        Args:
            start_time: The start of the time range in 'YYYY-MM-DD' format.
            end_time: The end of the time range in 'YYYY-MM-DD' format.

        Returns:
            A dictionary containing the status and a list of events that start within the provided time range.
        """
        # Convert the start and end times to datetime objects for easier comparison with the 'start_time' column in the DataFrame.
        start_time = datetime.strptime(start_time, '%Y-%m-%d')
        end_time = datetime.strptime(end_time, '%Y-%m-%d')

        if start_time == end_time:
            end_time = start_time + timedelta(days=1)
            self.events_df['start_time'] = pd.to_datetime(self.events_df['start_time'])
            self.events_df['end_time'] = pd.to_datetime(self.events_df['end_time'])

            # Filter the events DataFrame to include only events that have a 'start_time' within the specified range.
            # It's assumed that the 'start_time' column in the events DataFrame is in a format that can be compared directly with datetime objects.
            filtered_events = self.events_df[(self.events_df['start_time'] >= start_time) & (self.events_df['start_time'] < end_time)]
            filtered_events['start_time'] = filtered_events['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            filtered_events['end_time'] = filtered_events['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            # Convert the filtered events to a list of dictionaries for easier serialization and consumption.
            return {"status": "success", "data": filtered_events.to_dict('records')}

        self.events_df['start_time'] = pd.to_datetime(self.events_df['start_time'])
        self.events_df['end_time'] = pd.to_datetime(self.events_df['end_time'])

        # Filter the events DataFrame to include only events that have a 'start_time' within the specified range.
        # It's assumed that the 'start_time' column in the events DataFrame is in a format that can be compared directly with datetime objects.
        filtered_events = self.events_df[(self.events_df['start_time'] >= start_time) & (self.events_df['start_time'] <= end_time)]
        filtered_events['start_time'] = filtered_events['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        filtered_events['end_time'] = filtered_events['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # Convert the filtered events to a list of dictionaries for easier serialization and consumption.
        events_list = filtered_events.to_dict('records')
        return {"status": "success", "data": filtered_events.to_dict('records')}


    def delete_event_in_calendar(self, event_id: int) -> dict:
        """
        Deletes an event from the calendar by its ID. 

        Args:
            event_id: ID of the event to delete.

        Returns:
            A success message or an error message if the event is not found.
        """
        mask = self.events_df['id'] != event_id
        original_length = len(self.events_df)
        self.events_df = self.events_df[mask]
        if self.events_df.shape[0] < original_length:
            return {"status": "success", 'message': "Event deleted successfully."}
        else:
            return {"status": "failure", 'message': f"Event id {event_id} is not found"}
        
    def search_calendar_event_with_title(self, query: str) -> Dict[str, Union[str, datetime, timedelta]]:
        """
        Retrieves the details of a specific event by using query to match its title. 

        Args:
            query: The query to retrieve event.

        Returns:
            Details of the event.
        """
        event = self.events_df[self.events_df['title'].str.contains(query.lower(), case=False)]
        if event.empty:
            return {"status": "failure", "message": f'Event with query {query} not found.', "data": []}
        return {"status": "success", 'message': "", "data": event.to_dict('records')}
    
    def is_conflicting(self, new_event: pd.DataFrame) -> bool:
        """
        Checks if the new event conflicts with existing events in the calendar.
        """
        conflict = False
        date_format = '%Y-%m-%d %H:%M:%S'
        for index, event in self.events_df.iterrows():
            # start_time = datetime.strptime(event['start_time'], date_format)
            # end_time = datetime.strptime(event['end_time'], date_format)
            start_time = event['start_time']
            end_time = event['end_time']
            if (new_event['start_time'][0] < end_time and new_event['end_time'][0] > start_time):
                conflict = True
                break
        return conflict
    
    def add_alarm(self, alarm_time: datetime, message: str) -> dict:
        """
        Adds a new alarm to the calendar.

        Args:
            alarm_time: The time at which the alarm should go off, should in the format: %Y-%m-%d %H:%M:%S, e.g. 2024-05-28 15:30:00..
            message: The message to display when the alarm goes off.

        Returns:
            A success message or an error message if the alarm can't be add.
        """
        new_alarm = pd.DataFrame({
            'id': [generate_timestamp_random_id()],
            'alarm_time': [alarm_time],
            'message': [message]
        })
        self.alarms_df = pd.concat([self.alarms_df, new_alarm], ignore_index=True)
        return {"status": "success", "message": "Alarm added successfully"}

    
    def remove_alarm(self, alarm_index: int) -> dict:
        """
        Removes an alarm from the calendar by its index.

        Args:
            alarm_index: Index of the alarm to remove.

        Returns:
            A success message or an error message if the alarm index is not found.
        """
        try:
            self.alarms_df = self.alarms_df.drop(alarm_index)
            return {"status": "success", "message": "Alarm removed successfully."}
        except IndexError:
            return {"status": "failure", "message": f"Alarm index {alarm_index} not found."}

    def view_today_alarms(self) -> dict:
        """
        View the today's alarms in the calendar.

        Returns:
            A dictionary contains the status and a list containing alarm details.
        """
        date = os.environ.get("CURRENT_DATE", None)
        alarm = self.alarms_df[self.alarms_df['alarm_time'].str.contains(date, case=False)]
        return {"status": "success", "data": alarm.to_dict('records')}


    


if __name__ == '__main__':
    calendar = Calendar("James_Harrington")
    os.environ["CURRENT_DATE"] = '2024-09-03'
    print(calendar.view_today_events_in_calendar())
    print(calendar.view_events_in_calendar_by_providing_time_range("2024-09-07", "2024-09-07"))
    