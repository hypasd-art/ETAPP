import re
import pandas as pd
from typing import List, Union
from datetime import datetime
import time
import random
import os
from PLA.toolkit.utils import generate_timestamp_random_id
from rank_bm25 import BM25Okapi

EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

class Email:
    def __init__(self, name: str, emails_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../../database/Email/emails_{}.csv", valid_address_path: str = None) -> None:
        """
        Initializes the Email system with a personal email database.

        Args:
            name: The user's name.
            emails_path: Optional path to the CSV file containing email records.
        """
        self.name = name
        self.email = self.name.replace(" ", "_") + "@mail.com"
        self.emails_path = emails_path.format(self.name)
        self.vaild_address = valid_address_path
        self.emails_df = pd.DataFrame()
        if emails_path:
            self.load_emails(self.emails_path)
        self.bm25 = self._initialize_bm25()

    def get_today_emails_until_now(self) -> dict:
        """
        Retrieves today's emails until now.

        Returns:
            A dictionary containing the status and a list of email records for the current day until now.
        """
        date = os.environ.get("CURRENT_DATE", None)
        current_time = os.environ.get("CURRENT_TIME", None)
        
        emails = self.emails_df[
            (self.emails_df['timestamp'].str.contains(date, case=False)) & 
            (pd.to_datetime(self.emails_df['timestamp']) < datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S'))
        ]
        return {"status": "success", "data": emails.to_dict('records')}


    def load_emails(self, path: str) -> None:
        """
        Loads emails data from the specified CSV file and stores it in a DataFrame.

        Args:
            path: Path to the CSV file containing emails data.
        """
        self.emails_df = pd.read_csv(path)

    def _initialize_bm25(self):
        
        corpus = self.emails_df['content'].tolist()
        
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        return BM25Okapi(tokenized_corpus)

    def is_valid_email(self, email: str) -> bool:
        """
        Validates if the given email is correctly formatted and in the list of valid addresses.

        Args:
            email: Email to validate.

        Returns:
            True if the email is valid, False otherwise.
        """
        return re.match(EMAIL_REGEX, email) is not None # and email in valid_address

    def send_email(self, receiver: Union[str, List[str]], subject: str, content: str, attachments: str = None) -> dict:
        """
        Sends an email.

        Args:
            receiver: the receiver(s) of the email.
            subject: the subject of the email.
            content: the content of the email.
            attachments: the path of the attachments.

        Returns:
            the status of the email.
        """
        subject = subject.strip()
        content = content.strip()

        # Ensure subject and content are not empty
        if not subject:
            return {"status": "failure", "message": 'Subject cannot be empty.'}
        if not content:
            return {"status": "failure", "message": 'Content cannot be empty.'}

        # Normalize receiver to a list
        if isinstance(receiver, str):
            receiver = [receiver.strip()]
        elif isinstance(receiver, list):
            receiver = [r.strip() for r in receiver]
        else:
            return {"status": "failure", "message": 'Email receiver should be a string or a list of strings.'}

        # Validate each receiver email
        for r in receiver:
            if not self.is_valid_email(r):
                return {"status": "failure", "message": f'Invalid email address: {r}'}

        

        return {"status": "success", "message": f"The mail to {receiver} is send"}

    def search_email_by_sender_and_receiver(self, address: str) -> dict:
        """
        Searches for emails by subject.

        Args:
            subject: Subject to search for.

        Returns:
            List of emails matching the subject.
        """
        address = address.strip().lower()
        current_time = os.environ.get("CURRENT_TIME", None)
        return {"status": "success", "data": self.emails_df[(self.emails_df['sender'].str.contains(address, case=False) | self.emails_df['receiver'].str.contains(address, case=False)) & 
                                                (pd.to_datetime(self.emails_df['timestamp']) < datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S'))].to_dict('records')}


    def search_email_by_content(self, query: str) -> dict:
        """
        Searches for emails by the similarity of email's content and query.

        Args:
            query: Query to search for.

        Returns:
            List of emails matching the query.
        """
        current_time = os.environ.get("CURRENT_TIME", None)
        tokenized_query = query.split(" ")
        
        
        scores = self.bm25.get_scores(tokenized_query)
        
        
        
        filtered_emails = self.emails_df[
            (pd.to_datetime(self.emails_df['timestamp']) < datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S'))
        ]
        
        filtered_indices = filtered_emails.index.tolist()
        
        
        filtered_scores = [scores[i] for i in filtered_indices]
        
        
        top_indices = sorted(range(len(filtered_scores)), key=lambda i: filtered_scores[i], reverse=True)[:5]
        
        
        top_3_emails = filtered_emails.iloc[top_indices].sort_values(by="timestamp", ascending=False)
        
        
        return {
            "status": "success",
            "data": top_3_emails.to_dict('records')
        }
        
    
    def save_emails(self, emails_path: str = None) -> None:
        self.emails_df.to_csv(emails_path if emails_path is not None else self.emails_path, index=False)

if __name__ == '__main__':
    email_system = Email("James_Harrington")
    os.environ['CURRENT_DATE'] = '2024-09-08'
    os.environ['CURRENT_TIME'] = '2024-09-09 09:00:00'
    t_email = email_system.get_today_emails_until_now()
    print(email_system.search_email_by_sender_and_receiver("sarah.lee@artenthusiasts.org"))
    print("\n")
    print(t_email)
    print("\n")
    print(email_system.send_email(receiver="Alice@email.com", subject="h", content="hello"))
    print("\n")
    print(email_system.search_email_by_content(query="speech presentation"))
    

