import pandas as pd
from datetime import datetime
from typing import List, Dict, Union
import requests
import os
import time
import random
import pickle
import json
from PLA.toolkit.utils import generate_timestamp_random_id
from rank_bm25 import BM25Okapi



class ECommerce:
    def __init__(self, name: str, database_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Shopping/product.csv",
                 cart_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Shopping/carts_{}.csv",
                
                 ) -> None:
        """
        Initializes the ECommerce system with the user's name and a database of operations.
        Args:
            name: The name of the user.
            database_path: Path to the CSV file containing the history of operations.
        """
        self.name = name
        # self.datbase_path = database_path
        self.cart_path = cart_path.format(name)
        # self.history_path = history_path.format(name)
        # self.bought_products_path = bought_products_path.format(self.name)
        self.cart = pd.DataFrame()
        # self.browsing_history = pd.DataFrame()
        # Load the operation history from a CSV file if a path is provided.
        if database_path:
            self.load_database(database_path)
        
        if cart_path:
            self.load_cart(self.cart_path)
        
        self.bm25_model = self._build_bm25_model()

    def load_database(self, path: str) -> None:
        """
        Loads all products from a specified CSV file into a DataFrame.
        Args:
            path: Path to the CSV file containing all products.
        """
        self.database_df = pd.read_csv(path)
        
    def load_cart(self, path: str) -> None:
        """
        Loads user's cart from a specified CSV file into a DataFrame.
        Args:
            path: Path to the CSV file containing cart.
        """
        self.cart = pd.read_csv(path)



    def _build_bm25_model(self) -> BM25Okapi:
        """
        Builds a BM25 model using the product titles in the database.
        Returns:
            A BM25Okapi model.
        """
        # Tokenize the product titles
        tokenized_corpus = [doc.split() for doc in self.database_df['product_title']]
        # Create the BM25 model
        bm25_model = BM25Okapi(tokenized_corpus)
        return bm25_model
    
    
    def add_product_to_cart(self, product_id: str, product_name: str, quantity: int) -> dict:
        """
        Adds a product to the cart.
        Args:
            product_id: The unique identifier of the product.
            product_name: The name of the product.
            quantity: The number of units of the product to add to the cart.
        Returns:
            A message confirming the addition.
        """
        # self.cart = pd.concat([self.cart, pd.DataFrame({'product_id': [product_id], 'product_name': [product_name], 'quantity': [quantity]})], ignore_index=True)
        return {"status": "success", "message": f"{quantity} of {product_id} added to cart."}
    
    def search_products_in_shopping_manager(self, query: Union[str, list]) -> Dict:
        """
        Searches for products in the browsing history.
        Args:
            query: The name of the product to search for or a list of product names to search for.
        Returns:
            A dictionary containing a status message and a list of matching products.
        """
        
        all_matches = []
        columns_to_keep = [
            'asin', 'product_title', 'product_price', 'product_original_price',
            'currency', 'product_star_rating', 'product_num_ratings', 'product_url',
            'sales_volume', 'delivery', 'has_variations'
        ]

        # Check if the query is a list
        if isinstance(query, list):
            # Iterate over each item in the list and search for it
            for q in query:
                tokenized_query = q.split()
                # Get the BM25 scores for the query
                scores = self.bm25_model.get_scores(tokenized_query)
                # Get the top matching products
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]  # Get top 10 matches
                matches = self.database_df.iloc[top_indices][columns_to_keep]
                # matches = self.database_df[self.database_df['product_title'].str.contains(q, case=False)][columns_to_keep]
                # Extend the all_matches list with the current query's matches
                all_matches.extend(matches.to_dict('records'))
        else:
            tokenized_query = query.split()
            # Get the BM25 scores for the query
            scores = self.bm25_model.get_scores(tokenized_query)
            # Get the top matching products
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]  # Get top 10 matches
            matches = self.database_df.iloc[top_indices][columns_to_keep]
            # matches = self.database_df[self.database_df['product_title'].str.contains(query, case=False)][columns_to_keep]
            # print(matches.columns)
            # print("\n"*10)
            if not matches.empty:
                products_list = matches
                return {"status": "success", "products": products_list.to_dict('records')}

        # Check if any matches were found
        if all_matches != []:
            return {"status": "success", "products": all_matches}
        else:
            return {"status": f"No products found for '{query}'", "products": []}

    def view_cart_in_shopping_manager(self) -> list:
        """
        Views the current items in the cart.
        Returns:
            A list containing items in the cart.
        """
        return self.cart.to_dict(orient='records')

    def get_status_information_of_purchased_products(self) -> dict:
        return
    
    

if __name__ == '__main__':
    ecommerce = ECommerce("John_Doe")
    
    print(ecommerce.search_products_in_shopping_manager(["Organic Cotton Bed Sheets", "Books by Matt Haig", "Leather Wallet", "Gourmet Olive Oil"]))
    print(ecommerce.search_products_in_shopping_manager(["orange", "common fruits"]))

    