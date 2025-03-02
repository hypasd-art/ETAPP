import pandas as pd
from datetime import datetime
from typing import Dict, List, Union
import time
import random
import os
from PLA.toolkit.utils import generate_timestamp_random_id


class Light:
    def __init__(self, name: str, 
                
                 state: bool = False, brightness: int = 0, color: str = "yellow") -> None:
        """
        Initializes a light with a name, state, brightness, color, and history.
        Args:
            name: The name of the light.
            history_path: Path to the CSV file containing operation history.
        """
        self.name = name
        # self.history_path = history_path.format(name)
        self.state: bool = state  # Light is initially off
        self.brightness: int = brightness  # Brightness level (0-100)
        self.color: str = color  # Default color
        

    

    def control_light_in_home(self, action: str, location: str, brightness: int = 1, color: str = "white") -> dict:
        """
        Controls the light in the home by turning it on or off with specified settings.
        Args:
            action: The action to perform ('on' or 'off').
            location: The location of the light in the home ('residence', 'kitchen', 'dining room', 'living room', 'bedroom', 'bathroom').
            brightness: Brightness level (1-3). Required if action is 'on'.
            color: Color of the light ('yellow' or 'white'). Required if action is 'on'.
        Returns:
            A dictionary containing the status and a message about the light's state.
        """
        if action not in ['on', 'off']:
            raise Exception("The action must be 'on' or 'off'.")
        
        if not location:
            raise Exception("Location must be specified.")
        
        location = location.lower()
        
        if location not in ["residence", "kitchen", "dining room", "living room", "bedroom", "bathroom"]:
            raise Exception(f"""Invalid location. Supported locations are: {', '.join(["residence", "kitchen", "dining room", "living room", "bedroom", "bathroom"])}.""")
        
        if action == 'on':
            if brightness is None or color is None:
                raise Exception("Both brightness and color must be provided when turning the light on.")
            
            return self.turn_on_light_in_home(brightness, color, location)
        
        elif action == "off":
            return self.turn_off_light_in_home(location)

    
    def turn_on_light_in_home(self, brightness: int = 1, color: str = "white", location: str = "") -> dict:
        """
        Turns the light on with specified brightness and color.
        Args:
            brightness: Brightness level (1-3).
            color: Color of the light ('yellow' or 'white').
        Returns:
            Status message.
        """
        if brightness not in [1, 2, 3]:
            raise Exception("Brightness level must be 1 (low), 2 (medium), or 3 (high).")
        
        if color not in ["yellow", "white", "white-yellow"]:
            raise Exception("Color must be 'yellow', 'white' or 'white-yellow'.")

        self.state = True
        self.brightness = brightness
        self.color = color
        # self._log_operation({"state":self.state, "brightness": brightness, "color": self.color})
        return {"status": "success", 'message': f"Light in {location} is now ON with brightness {self.brightness} and color {self.color}."}

    
    def turn_off_light_in_home(self, location: str) -> dict:
        """Turns the light off."""
        self.state = False
        # self._log_operation({"state":self.state, "brightness": None, "color": None})
        return {"status": "success", 'message': f"Light in {location} is now OFF."}

    
    
if __name__ == '__main__':
    # Example usage of the Light class
    living_room_light = Light("Jane_Doe")
    
    # Turn the light on with specific settings
    print(living_room_light.turn_on_light(brightness=2, color="yellow"))
    # Get current status
    print(living_room_light.get_lighting_status())
    # Turn the light off
    print(living_room_light.turn_off_light())
    # Get operation history
    print(living_room_light.get_lighting_control_history())

    # Save the updated history to the CSV file
    living_room_light.save_history()