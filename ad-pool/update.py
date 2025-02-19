import os
import csv
import sqlite3

db_file = 'advertisements.db' 
mapping_file = 'mapping.csv'   
video_folder = 'videos' 

connection = sqlite3.connect(db_file)
cursor = connection.cursor()

with open(mapping_file, mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        demographics_id = int(row['demographics_id'])
        video_name = row['video_name']
        ad_description = row.get('ad_description', '')  
        product_name = row.get('product_name', '')
        video_path = os.path.join(video_folder, video_name)  

        check_query = """
            SELECT * FROM ads WHERE demographics_id = ? AND ad_content = ?
        """
        cursor.execute(check_query, (demographics_id, video_path))
        
        if cursor.fetchone() is not None:
            update_query = """
                UPDATE ads
                SET product_name = ?
                WHERE demographics_id = ? AND ad_content = ?
            """
            cursor.execute(update_query, (product_name, demographics_id, video_path))
            print(f"Updated product_name for: {video_name}")
    
        else:
            insert_query = """
                INSERT INTO ads (demographics_id, ad_content, ad_description, product_name)
                VALUES (?, ?, ?, ?)
            """
            cursor.execute(insert_query, (demographics_id, video_path, ad_description, product_name))
            print(f"Inserted new entry for: {video_name}")

connection.commit()
cursor.close()
connection.close()

print("All advertisements processed successfully!")