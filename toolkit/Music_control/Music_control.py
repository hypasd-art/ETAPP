import pandas as pd
import os
from PLA.toolkit.utils import generate_timestamp_random_id
import requests

class MusicStreamingApp:
    def __init__(self, name: str, 
                 favorites_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Music/favorites_{}.csv", 
                 database_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Music/music_all.csv"
                 ) -> None:
        """
        Initializes the Music Streaming App with a user's name and a database for music operations.
        Args:
            name: The name of the user.
            database_path: Optional path to the CSV file containing music operation records.
        """
        self.name = name
        self.favorites_path = favorites_path.format(name)
        # self.history_path = history_path.format(name)

        # Initialize empty DataFrames to store music operation records.
        self.database_df = pd.DataFrame()
        self.load_favorites(self.favorites_path)
        # self.load_history(self.history_path)
        self.current_music = None
        self.is_playing = False
        # Load music operation records from a CSV file if a path is provided.
        if database_path:
            self.load_database(database_path)

    def load_database(self, path: str) -> None:
        """
        Loads music operation records from a specified CSV file into a DataFrame.
        Args:
            path: Path to the CSV file containing music operation records.
        """
        # Read the CSV file into the DataFrame.
        self.database_df = pd.read_csv(path)

    def load_favorites(self, path: str) -> None:
        """
        Loads music operation records from a specified CSV file into a DataFrame.
        Args:
            path: Path to the CSV file containing music operation records.
        """
        # Read the CSV file into the DataFrame.
        self.favorites = pd.read_csv(path)

    
    def play_music(self, music_name: str, volume_level: int) -> dict:
        """
        Plays a music track and records the playback operation.
        Args:
            music_name: The name of the music track to play. excluding the artist's name.
            volume_level: The volume level for playback (0-100), ranging from 0 (silent) to 100 (loudest).
        Returns:
            Status message indicating the result of the operation.
        """
        
        result = self.search_music_by_name(music_name)
        
        if result['status'] == "success":
            music_name = result['data'][0]['title']
            return {"status": "success", "message": f"Playing: {music_name} at a volume level: {volume_level}"}
        else:
            return {"status": "failure", "message": f"Error: Music track '{music_name}' not found in the database."}
    
    def search_music_by_name(self, query: str) -> dict:
        """
        Searches for music in the database by matching music's name and sorts the results by annotation count.

        Args:
            query: The query to search for music by matching music's name in the music database. 

        Returns:
            dict: A dictionary containing the status of the search and a list of matching music records.
        """
        max_observation_length = int(os.environ.get("max_observation_length"))
        
        matches = self.database_df[self.database_df['title'].str.contains(query.lower(), case=False)]
        if not matches.empty:
            c_len = 2+ len('{"status": "success", "data": ') + 1
            matches = matches.sort_values(by='annotation_count', ascending=False)
            # matches = matches[['artist_names', 'full_title']]
            music_list = matches.to_dict(orient='records')
            
            return {"status": "success", "data": music_list}
            # return {"status": "success", "data": cut_music_list}
        else:
            return {"status": "failure", "data": []}
        
    
    def search_music(self, q: str, per_page: int=10, page: int=1, text_format: str=None, toolbench_rapidapi_key: str='c61161d43emshdf6dd68cccf3c7bp157507jsn01c27187f369'):
        """
        "The search capability covers all content hosted on Genius (all songs)."
        q: Search query
            per_page: Number of results to return per request
            page: Paginated offset, (e.g., per_page=5&page=3 returns results 11–15)
            text_format: Format for text bodies related to the document. One or more of `dom`, `plain`, `markdown`, and `html`, separated by commas (defaults to html).
            
        """
        url = f"https://genius-song-lyrics1.p.rapidapi.com/search/"
        querystring = {'q': q, }
        if per_page:
            querystring['per_page'] = per_page
        if page:
            querystring['page'] = page
        if text_format:
            querystring['text_format'] = text_format
        
        headers = {
                "X-RapidAPI-Key": toolbench_rapidapi_key,
                "X-RapidAPI-Host": "genius-song-lyrics1.p.rapidapi.com"
            }


        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except:
            observation = response.text
        return observation
    

    
    
    def get_music_list_in_favorites(self) -> list:
        """
        Returns the list of favorite tracks.
        Returns:
            List of favorite tracks.
        """
        return self.favorites.to_dict('records')
    
    
# os.environ['max_obversation_length'] = str(100)
if __name__ == '__main__':
    music_app = MusicStreamingApp(name="John_Doe")
    os.environ['max_observation_length'] = str(2000)
    result = music_app.search_music_by_name('Take Five')
    print(result)
    print("\n\n\n")
    # print(music_app.search_music_by_category("classical"))
    print(music_app.play_music("Take Five", 3))

    # # Save the updated database to the CSV file
    # music_app.save_database()