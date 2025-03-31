# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

import os  
import csv  
import sqlite3  
from util import get_resource_path  
# Path to the SQLite database file containing advertisement data
db_file = get_resource_path('advertisements.db')
# Path to the CSV file mapping demographics to video advertisements
mapping_file = 'mapping.csv'   
# Directory containing video files
video_folder = 'videos' 

# Establish connection to the SQLite database
connection = sqlite3.connect(db_file)
# Create a cursor object to execute SQL commands
cursor = connection.cursor()

# Open and read the mapping CSV file with UTF-8 encoding
with open(mapping_file, mode='r', newline='', encoding='utf-8') as file:
    # Create a DictReader to parse CSV rows into dictionaries
    csv_reader = csv.DictReader(file)
    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Extract and convert demographics_id to integer
        demographics_id = int(row['demographics_id'])
        # Get the video filename from the CSV
        video_name = row['video_name']
        # Get ad description, default to empty string if not provided
        ad_description = row.get('ad_description', '')  
        # Get product name, default to empty string if not provided
        product_name = row.get('product_name', '')
        # Construct the full video path by joining the folder and filename
        video_path = os.path.join(video_folder, video_name)  

        # SQL query to check if an ad with the given demographics_id and video path already exists
        check_query = """
            SELECT * FROM ads WHERE demographics_id = ? AND ad_content = ?
        """
        # Execute the check query with parameters
        cursor.execute(check_query, (demographics_id, video_path))
        
        # Fetch the result of the check query
        if cursor.fetchone() is not None:
            # SQL query to update the product name if the ad already exists
            update_query = """
                UPDATE ads
                SET product_name = ?
                WHERE demographics_id = ? AND ad_content = ?
            """
            # Execute the update query with parameters
            cursor.execute(update_query, (product_name, demographics_id, video_path))
            # Log the update action
            print(f"Updated product_name for: {video_name}")
    
        else:
            # SQL query to insert a new ad entry if it does not exist
            insert_query = """
                INSERT INTO ads (demographics_id, ad_content, ad_description, product_name)
                VALUES (?, ?, ?, ?)
            """
            # Execute the insert query with parameters
            cursor.execute(insert_query, (demographics_id, video_path, ad_description, product_name))
            # Log the insert action
            print(f"Inserted new entry for: {video_name}")

connection.commit()
cursor.close()
connection.close()

print("All advertisements processed successfully!")