# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

import sqlite3  # For SQLite database operations
from util import get_resource_path  # Utility function to resolve resource paths

# Database configuration
db_file = get_resource_path('advertisements.db')  # Path to the SQLite database file

def get_targeted_videos_with_ads(age_group, gender, ethnicity):
    """
    Retrieve a list of advertisement content, descriptions, weights, and product names based on demographic information.

    Args:
        age_group (str): The target age group (e.g., '18-24').
        gender (str): The target gender (e.g., 'M', 'F').
        ethnicity (str): The target ethnicity (e.g., 'Asian', 'Caucasian').

    Returns:
        list: A list of tuples, each containing (ad_content, ad_description, weight, product_name).

    Raises:
        sqlite3.Error: If a database error occurs during query execution.
    """
    # Initialize connection
    connection = sqlite3.connect(db_file)  # Establish connection to the database
    cursor = connection.cursor()  # Create cursor for executing SQL commands

    # SQL query
    query = """
        SELECT a.ad_content, a.ad_description, a.weight, a.product_name
        FROM demographics AS d
        INNER JOIN ads AS a
        ON d.demographics_id = a.demographics_id
        WHERE d.gender = ?
        AND d.age_group = ?
        AND d.ethnicity = ?
        ORDER BY a.ad_content ASC;
    """  # Query to fetch ads matching demographic criteria, sorted by ad_content

    # Execute query
    video_ads_list = []  # List to store the query results
    try:
        cursor.execute(query, (gender, age_group, ethnicity))  # Execute query with parameterized inputs
        results = cursor.fetchall()  # Retrieve all matching rows
        video_ads_list = [(row[0], row[1], row[2], row[3]) for row in results]  # Convert rows to list of tuples
    except sqlite3.Error as err:
        print(f"Error: {err}")  # Log database errors
    finally:
        # Close connection
        if 'cursor' in locals():
            cursor.close()  # Close cursor if it exists
        if 'connection' in locals():
            connection.close()  # Close database connection if it exists

    return video_ads_list  # Return the list of targeted ads

# Example usage
if __name__ == "__main__":
    # Single line input
    user_input = input("Enter age_group, gender, ethnicity: ").strip()  # Prompt user for demographic input

    # Parse the input
    try:
        # Remove parentheses and split by commas
        user_input = user_input.strip("()")  # Remove surrounding parentheses
        age_group, gender, ethnicity = [x.strip().strip("'\"") for x in user_input.split(',')]  # Parse and clean input
    except ValueError:
        print("Invalid input format. Please enter values in the format: ('age_group', 'gender', 'ethnicity')")  # Handle parsing errors
        exit()  # Exit script on invalid input

    # Fetch the list of targeted videos with ads
    targeted_videos_with_ads = get_targeted_videos_with_ads(age_group, gender, ethnicity)  # Call function with parsed inputs
    
    # Print the results in the desired format
    if targeted_videos_with_ads:
        print(targeted_videos_with_ads)  # Output list of ad tuples if results exist
    else:
        print("No ads found for the given demographic information.")  # Indicate no results found