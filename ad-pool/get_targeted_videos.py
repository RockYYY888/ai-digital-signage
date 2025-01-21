import os
import mysql.connector

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'zyy123456',  
    'database': 'advertisements'
}

def get_targeted_videos_with_ads(age_group, gender, ethnicity):
    """
    Retrieve a list of advertisement content and their descriptions based on demographic information.

    Parameters:
        age_group (str): Age group.
        gender (str): Gender.
        ethnicity (str): Ethnicity.

    Returns:
        list of tuples: Each tuple contains ad_content and ad_description.
    """
    # Initialize connection
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    # SQL query
    query = """
        SELECT a.ad_content, a.ad_description
        FROM demographics AS d
        INNER JOIN ads AS a
        ON d.demographics_id = a.demographics_id
        WHERE d.gender = %s
        AND d.age_group = %s
        AND d.ethnicity = %s
        ORDER BY a.ad_content ASC;
    """

    # Execute query
    video_ads_list = []
    try:
        cursor.execute(query, (gender, age_group, ethnicity))
        results = cursor.fetchall()
        video_ads_list = [(row[0], row[1]) for row in results]  # Extract ad content and description into tuples
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        # Close connection
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()

    return video_ads_list

# Example usage
if __name__ == "__main__":
    # Fetch the list of targeted videos with ads
    targeted_videos_with_ads = get_targeted_videos_with_ads('17-35', 'Male', 'Asian')
    
    # Print the results as a Python list
    print("Targeted videos with advertisements:")
    print(targeted_videos_with_ads)