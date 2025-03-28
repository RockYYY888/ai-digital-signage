import os
import random
import sqlite3
from ad_pool.video_selection import get_targeted_videos_with_ads
import threading
from util import get_resource_path

watching_lock = threading.Lock()
class AdPool:
    def __init__(self):
        self.current_ad = None
        self.current_ads_list = []  # Store the list of ads
        self.lock = threading.Lock()
        self.db_file = get_resource_path('advertisements.db')  # Ensure the path is correct
        # Load all ads on initialization
        self.load_all_ads()

    def load_all_ads(self):
        """Load all ads from the database"""
        connection = sqlite3.connect(self.db_file)
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT ad_content FROM ads;")
            results = cursor.fetchall()
            self.current_ads_list = []  # Clear existing list
            for result in results:
                ad_path = result[0]
                # Update to point to the ad_pool/videos folder
                absolute_ad_path = os.path.join(os.path.dirname(__file__), ad_path)  
                if os.path.exists(absolute_ad_path):
                    self.current_ads_list.append((absolute_ad_path, 1.0))
                    print(f"Added to ad pool: {absolute_ad_path}")
                else:
                    print(f"Ad file not found: {absolute_ad_path}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            cursor.close()
            connection.close()

    def update_ads_for_demographic(self, age_group, gender, ethnicity):
        """Update the ad pool based on demographic characteristics"""
        with self.lock:
            ads_data = get_targeted_videos_with_ads(age_group, gender, ethnicity)
            print(f"Ads data: {ads_data}")
            if ads_data:
                self.current_ads_list = []
                for ad in ads_data:
                    ad_path = ad[0]
                    absolute_ad_path = os.path.join(os.path.dirname(__file__), ad_path)
                    if os.path.exists(absolute_ad_path):
                        self.current_ads_list.append((absolute_ad_path, ad[2]))
                        print(f"Added to ad pool: {absolute_ad_path}")
                    else:
                        print(f"Ad file not found: {absolute_ad_path}")
                self.current_ad = None
                print(f"Updated ad pool: {self.current_ads_list}")
            else:
                print(f"No ads found for demographic: {age_group}, {gender}, {ethnicity}")
                self.load_all_ads()  # Fallback to loading all ads

    def get_random_ad(self):
        """Randomly select an ad from the ad pool"""
        with self.lock:
            if not self.current_ads_list:
                return None
            # Randomly select an ad
            self.current_ad = random.choice([path for path, _ in self.current_ads_list])
            print(f"Selected ad: {self.current_ad}")
            # Reset watch time
            global total_watch_time
            with watching_lock:
                total_watch_time = 0
            return self.current_ad
