from typing import List, Dict, Union
from datetime import datetime
import pandas as pd
from pandas import DataFrame
import requests
import os
import pickle
import json
from pyserini.search.lucene import LuceneSearcher
from typing import List



class Browser:
    def __init__(self, name: str, 
                 data_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Web_Browsing/news_data.pkl", 
                
                 ) -> None:
        """
        Initializes the web browser with history and bookmarks.
        """
        self.name = name
        

        self.app_id = None
        # Base URL for the WolframAlpha API
        self.url = "http://api.wolframalpha.com/v2/query"

        with open(data_path, 'rb') as file:
            self.news_data = pickle.load(file)

        
            
    def information_retrieval(self, query, topk, searcher):
        hits = searcher.search(query, topk)
        paragraphs = []
        titles = []
        for i in range(len(hits)):
            doc = searcher.doc(hits[i].docid)
            json_doc = json.loads(doc.raw())
            doc_text = json_doc['contents']
            title = doc_text.split('\n')[0]
            # print(title)
            paragraphs.append(json_doc['contents'])
            titles.append(title)
        return paragraphs, titles

    def search_from_wikipedia(self, query: str, topk: int = 5) -> dict:
        """
        Searches Wikipedia for content based on a given query and returns the top k results.

        Args:
            query: The search term to look for on Wikipedia.
            topk: The number of top results to return. Defaults to 5.

        Returns:
            A dictionary containing the search status and the processed search results. The 'data' field contains a list of processed paragraphs, each in the format "title\ntruncated content".

        """
        searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs') # sparse wikipedia-kilt-doc

        searcher = LuceneSearcher.from_prebuilt_index('wikipedia-kilt-doc')
        paragraphs, titles = self.information_retrieval(query, topk, searcher)
        processed_paragraphs = []
        for p in paragraphs:
            lines = p.split('\n')
            title = lines[0]
            content = ' '.join(lines[1:])
            
            truncated_p = ' '.join(content.split(' ')[:100])
            processed_paragraphs.append(title + '\n' + truncated_p)
            
        
        return {"status": "success", "data": processed_paragraphs}
        
    
    
    def search_news_by_category(self, category: str) -> dict:
        """
        Finds the news by category.
        
        Args:
            category: The category to search for news. should shoose from [entertainment, world, business, sport, health, science, technology, ]
        
        Returns:
            A dictionary contains the search status and the search result
        """
        query = category
        for data in self.news_data:
            if data['category'] == query:
                return {"status": "success", "data": data}
        return {"status": "error", "message": f"The query {query} is not in defined category"}
        

    def search_heat_news(self) -> dict:
        """
        Fetches the latest hot news articles.

        Returns:
            A dictionary contains the search status and search result.
        """
        for data in self.news_data:
            if data['category'] == 'hot':
                return {"status": "success", "data": data['news']}
        
    
    
    
    

if __name__ == '__main__':
    # Example usage of the Browser class
    browser = Browser("John_Deo")
    

    print(browser.search_from_wikipedia("hhh"))
    
    
