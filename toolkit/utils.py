import random
import os

def generate_timestamp_random_id():
    timestamp = os.environ.get("CURRENT_DATE")
    random_part = random.randint(10000, 99999)
    return f"{timestamp}{random_part}"