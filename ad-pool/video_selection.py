import os
import sqlite3

# Database configuration
db_file = os.path.join(os.path.dirname(__file__), 'advertisements.db')

def get_targeted_videos_with_ads(age_group, gender, ethnicity):
    """
    Retrieve a list of advertisement content, descriptions, weights, and product names based on demographic information. 

    Parameters:
        age_group (str): Age group.
        gender (str): Gender.
        ethnicity (str): Ethnicity.

    Returns:
        list of tuples: Each tuple contains ad_content, ad_description, weight, and product_name.
    """
    # Initialize connection
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

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
    """

    # Execute query
    video_ads_list = []
    try:
        cursor.execute(query, (gender, age_group, ethnicity))
        results = cursor.fetchall()
        video_ads_list = [(row[0], row[1], row[2], row[3]) for row in results]  # Extract ad content, description, and weight
    except sqlite3.Error as err:
        print(f"Error: {err}")
    finally:
        # Close connection
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

    return video_ads_list

# Example usage
if __name__ == "__main__":
    # Single line input
    user_input = input("Enter age_group, gender, ethnicity: ").strip()

    # Parse the input
    try:
        # Remove parentheses and split by commas
        user_input = user_input.strip("()")
        age_group, gender, ethnicity = [x.strip().strip("'\"") for x in user_input.split(',')]
    except ValueError:
        print("Invalid input format. Please enter values in the format: ('age_group', 'gender', 'ethnicity')")
        exit()

    # Fetch the list of targeted videos with ads
    targeted_videos_with_ads = get_targeted_videos_with_ads(age_group, gender, ethnicity)
    
    # Print the results in the desired format
    if targeted_videos_with_ads:
        print(targeted_videos_with_ads)  # This will output the list of tuples as you requested
    else:
        print("No ads found for the given demographic information.")