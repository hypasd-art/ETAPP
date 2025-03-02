import pandas as pd
from pandas import DataFrame
from typing import Optional, Dict
import os
import pickle
import numpy as np

class Navigation_And_Map:
    def __init__(self, 
                 name: str,
                 city_path=os.path.dirname(os.path.abspath(__file__)) + "/../../database/Navigation/background/citySet_with_states.txt",
                 accommodation_path=os.path.dirname(os.path.abspath(__file__)) + "/../../database/Navigation/accommodations/clean_accommodations_2022_revise.csv",
                 attraction_path=os.path.dirname(os.path.abspath(__file__)) + "/../../database/Navigation/attractions/attractions.csv",
                 restaurant_path=os.path.dirname(os.path.abspath(__file__)) + "/../../database/Navigation/restaurants/clean_restaurant_2022_revise.csv",
                 distance_path=os.path.dirname(os.path.abspath(__file__)) + '/../../database/Navigation/googleDistanceMatrix/distance.csv',
                 flight_path=os.path.dirname(os.path.abspath(__file__)) + "/../../database/Navigation/flights/clean_Flights_revise.csv",
                 building_path=os.path.dirname(os.path.abspath(__file__)) + "/../../database/Navigation/city_building_information.pkl"):
        """
        Initializes the TravelAssistant with paths to various data files.
        
        Args:
            city_path: Path to the city data file.
            accommodation_path: Path to the accommodations data file.
            attraction_path: Path to the attractions data file.
            restaurant_path: Path to the restaurants data file.
            distance_path: Path to the distance matrix data file.
            flight_path: Path to the flights data file.
        """
        self.name = name
        with open(building_path, 'rb') as file:
            self.cities_building_layout = pickle.load(file)
        self.cities = self.load_cities(city_path)
        self.accommodations = self.load_accommodations(accommodation_path)
        self.attractions = self.load_attractions(attraction_path)
        self.restaurants = self.load_restaurants(restaurant_path)
        self.distance_matrix = self.load_distance_matrix(distance_path)
        self.flights = self.load_flights(flight_path)

    def load_cities(self, path: str) -> Dict[str, list]:
        """
        Loads city and state mappings from the specified file.
        
        Args:
            path: Path to the city-state mapping file.
        
        Returns:
            data: A dictionary mapping states to their cities.
        """
        city_state_mapping = open(path, "r").read().strip().split("\n")
        data = {}
        for unit in city_state_mapping:
            city, state = unit.split("\t")
            if state not in data:
                data[state] = [city]
            else:
                data[state].append(city)
        return data

    def load_accommodations(self, path: str) -> DataFrame:
        """
        Loads accommodation data from the specified CSV file.
        
        Args:
            path: Path to the accommodations data file.
        
        Returns:
            DataFrame: A DataFrame containing accommodation data.
        """
        return pd.read_csv(path).dropna()[['NAME', 'price', 'room type', 
                                              'house_rules', 'minimum nights', 
                                              'maximum occupancy', 'review rate number', 
                                              'city']]

    def load_attractions(self, path: str) -> DataFrame:
        """
        Loads attraction data from the specified CSV file.
        
        Args:
            path: Path to the attractions data file.
        
        Returns:
            DataFrame: A DataFrame containing attraction data.
        """
        return pd.read_csv(path).dropna()[['Name', 'Latitude', 'Longitude', 
                                             'Address', 'Phone', 'Website', "City"]]

    def load_restaurants(self, path: str) -> DataFrame:
        """
        Loads restaurant data from the specified CSV file.
        
        Args:
            path: Path to the restaurants data file.
        
        Returns:
            DataFrame: A DataFrame containing restaurant data.
        """
        return pd.read_csv(path).dropna()[['Name', 'Average Cost', 'Cuisines', 
                                             'Aggregate Rating', 'City']]

    def load_distance_matrix(self, path: str) -> DataFrame:
        """
        Loads distance data from the specified CSV file.
        
        Args:
            path: Path to the distance matrix data file.
        
        Returns:
            DataFrame: A DataFrame containing distance data.
        """
        return pd.read_csv(path)

    def load_flights(self, path: str) -> DataFrame:
        """
        Loads flight data from the specified CSV file.
        
        Args:
            path: Path to the flights data file.
        
        Returns:
            DataFrame: A DataFrame containing flight data.
        """
        return pd.read_csv(path).dropna()[['Flight Number', 'Price', 'DepTime', 
                                             'ArrTime', 'ActualElapsedTime', 
                                             'FlightDate', 'OriginCityName', 
                                             'DestCityName', 'Distance']]
    
    
        
    def find_accommodations(self, city: str) -> dict:
        """
        Searches for accommodations in a specified city.
        
        Args:
            city: The city in which to search for accommodations.
        
        Returns:
            A dictionary contains the status and a list of accommodations in the city.
        """
        results = self.accommodations[self.accommodations["city"] == city]
        return {"status": "success", "data": results.reset_index(drop=True).to_dict('records')} 

    def find_attractions(self, city: str) -> dict:
        """
        Searches for attractions in a specified city.
        
        Args:
            city: The city in which to search for attractions.
        
        Returns:
            A dictionary contains the status and a list of attractions in the city.
        """
        results = self.attractions[self.attractions["City"] == city]
        return {"status": "success", "data": results.reset_index(drop=True).to_dict('records')} 

    def find_restaurants(self, city: str) -> dict:
        """
        Searches for restaurants in a specified city.
        
        Args:
            city: The city in which to search for restaurants.
        
        Returns:
            A dictionary contains the status and a list of restaurants in the city.
        """
        results = self.restaurants[self.restaurants["City"] == city]
        return {"status": "success", "data": results.reset_index(drop=True).to_dict('records')} 
    
    

    def find_flight(self, origin_city: str, destination_city: str, departure_date: str) -> dict:
        """
        Searches for flights between two cities on a specified date.
        
        Args:
            origin_city: The origin city.
            destination_city: The destination city.
            departure_date: The date of departure.
        
        Returns:
            A dictionary contains the status and a list of filght between the origin_city and the destination_city.
        """
        results = self.flights[(self.flights["OriginCityName"] == origin_city) & 
                                (self.flights["DestCityName"] == destination_city) & 
                                (self.flights["FlightDate"] == departure_date)]
        return {"status": "success", "data": results.reset_index(drop=True).to_dict('records')} 

if __name__ == '__main__':
    travel_assistant = Navigation_And_Map()
    # Example usage
    print(travel_assistant.get_cities_by_state("California"))
    print(travel_assistant.find_accommodations("Los Angeles"))
    print(travel_assistant.find_attractions("San Francisco"))
    print(travel_assistant.find_restaurants("New York"))
    print(travel_assistant.get_distance_info("Los Angeles", "San Francisco"))
    print(travel_assistant.find_flights("Los Angeles", "New York", "2022-06-01"))
    print(travel_assistant.get_city_set())  # Get unique cities from flight data
