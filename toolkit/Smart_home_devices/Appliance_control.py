import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import time
import random
import os
from PLA.toolkit.utils import generate_timestamp_random_id



# the Smart Appliance contains the 
class SmartAppliance:
    def __init__(self, name: str, database_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Smart_home_devices/Appliance/", 
                 curtains_open: bool = False, bathtub_open: bool = False, 
                 kettle_temperature: float = 20.0, is_boiling: bool = False,
                 bathtub_water_level: int = 0, bathtub_temperature: float = 20.0
                 ) -> None:
        """
        Initializes the Smart Appliance system with a user's name and an operation history.
        Args:
            name: The name of the user.
            database_path: Path to the CSV file containing past operations.
        """
        self.name = name
        self.database_path = database_path
        


        # Initialize the state of the smart appliance.
        self.curtains_open = curtains_open
        self.bathtub_open = bathtub_open
        self.kettle_temperature = kettle_temperature  # Initial water temperature in Celsius
        self.is_boiling = is_boiling
        self.bathtub_water_level = bathtub_water_level  # Initial water level in the bathtub
        self.bathtub_temperature = bathtub_temperature  # Initial water temperature in Celsius

    

    def control_curtains_in_home(self, open: str) -> str:
        """
        Controls the opening and closing of the curtains in home.
        Args:
            open: True to open the curtains, False to close.
        Returns:
            Status message.
        """
        self.curtains_open = open.capitalize()
        if open not in ['True', 'False']:
            raise Exception("The open must be 'True' or 'False'.")
        action = "opened" if open == "True" else "closed"
        # self.log_operation(f"Curtains {action}.", "curtains")
        return {"status": "success", "message": f"Curtains {action}."}

    

    def control_bathtub_in_home(self, fill: str, water_level: float = None, temperature: float = None, keep_temperature: str = "False") -> str:
        """
        Controls the filling and temperature of the bathtub in home.
        Args:
            fill: True to fill the bathtub, False to drain.
            temperature: Desired water temperature in Celsius.
        Returns:
            Status message.
        """
        operation = {"operation": None, "temperature": None}
        fill = fill.capitalize()
        if fill not in ['True', 'False']:
            raise Exception("The fill must be 'True' or 'False'.")
        keep_temperature = keep_temperature.capitalize()
        if keep_temperature not in ['True', 'False']:
            raise Exception("The keep_temperature must be 'True' or 'False'.")
        if fill == "True":
            self.bathtub_open = True
            self.bathtub_water_level = 10  # Increment water level by 10 liters
            operation["operation"] = "fill up the bathtub"
            if temperature is not None:
                self.bathtub_temperature = temperature
                operation["temperature"] = temperature # += f", and setting temperature to {temperature}°C."
        else:
            self.bathtub_open = False
            self.bathtub_water_level = 0  # Decrement water level by 10 liters
            operation["operation"] = "drain the bathtub"
        operation["water_level"] = water_level
        return {"status": "success", 'message': operation}

    

    def boil_water_in_home(self, temperature: float, keep_temperature: str = "False") -> dict:
        """
        Boiling water in the kettle in home.
        Args:
            temperature: Desired water temperature in Celsius(<100).
            keep_temperature: If True, keeps the water at the desired temperature ('True' or 'False').
        Returns:
            Status message.
        """
        if temperature > 100:
            return {"status": "failure", "message": "Error: Temperature exceeds boiling point of water."}
        keep_temperature = keep_temperature.capitalize()
        if keep_temperature not in ['True', 'False']:
            raise Exception("The keep_temperature must be 'True' or 'False'.")
        current_temperature = self.kettle_temperature
        self.kettle_temperature = temperature
        self.is_boiling = True
        

        self.is_boiling = False
        if keep_temperature == "True":
            return {"status": "success", "message": f"Water is boiling at {temperature}°C, and will maintain this temperature."}
        else:
            return {"status": "success", "message": f"Water has boiled at {temperature}°C."}
    
    
    
if __name__ == '__main__':
    user_appliance = SmartAppliance("Jane_Doe")

    # Control curtains
    print(user_appliance.control_curtains(True))
    print(user_appliance.control_curtains(False))

    # Drain bathroom water
    print(user_appliance.control_bathtub(True, 60))

    # Set kettle temperature and boil water
    print(user_appliance.boil_water(90))

    

    

    