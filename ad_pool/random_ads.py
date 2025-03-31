# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
import os
import random
import sqlite3
from ad_pool.video_selection import get_targeted_videos_with_ads
import threading
from util import get_resource_path

# Global lock to synchronize access to watch time across threads
watching_lock = threading.Lock()

class AdPool:
    """A class to manage an advertisement pool for targeted digital signage.

    This class handles loading, updating, and selecting advertisements based on demographic data.
    It interacts with a SQLite database and ensures thread-safe operations using locks.
    """

    def __init__(self):
        """Initialize the AdPool instance with an empty ad list and database connection."""
        self.current_ad = None  # The currently selected advertisement
        self.current_ads_list = []  # List of tuples: (ad_path, priority_score)
        self.lock = threading.Lock()  # Lock for thread-safe ad pool updates
        self.db_file = get_resource_path('advertisements.db')  # Path to the SQLite database file
        # Load all available ads during initialization
        self.load_all_ads()

    def load_all_ads(self):
        """Load all advertisements from the SQLite database into the ad pool.

        This method queries the 'ads' table, retrieves ad file paths, and constructs absolute paths.
        It updates `current_ads_list` with valid ad entries and logs missing files.
        """
        connection = sqlite3.connect(self.db_file)
        cursor = connection.cursor()
        try:
            # Fetch all ad content paths from the database
            cursor.execute("SELECT ad_content FROM ads;")
            results = cursor.fetchall()
            self.current_ads_list = []  # Reset the ad list
            for result in results:
                ad_path = result[0]  # Extract ad file path from query result
                # Construct absolute path relative to the current file's directory
                absolute_ad_path = os.path.join(os.path.dirname(__file__), ad_path)
                if os.path.exists(absolute_ad_path):
                    # Add valid ad with default priority score of 1.0
                    self.current_ads_list.append((absolute_ad_path, 1.0))
                    print(f"Added to ad pool: {absolute_ad_path}")
                else:
                    # Log warning for missing ad files
                    print(f"Ad file not found: {absolute_ad_path}")
        except sqlite3.Error as e:
            # Handle and log database-related errors
            print(f"Database error: {e}")
        finally:
            # Ensure resources are properly released
            cursor.close()
            connection.close()

    def update_ads_for_demographic(self, age_group, gender, ethnicity):
        """Update the ad pool based on specified demographic characteristics.

        Args:
            age_group (str): The target age group (e.g., '18-24').
            gender (str): The target gender (e.g., 'M', 'F', 'Other').
            ethnicity (str): The target ethnicity (e.g., 'Asian', 'Caucasian').

        This method retrieves targeted ads using demographic data, updates `current_ads_list`,
        and falls back to loading all ads if no targeted ads are found.
        """
        with self.lock:  # Ensure thread-safe update of the ad pool
            # Fetch targeted ads based on demographic criteria
            ads_data = get_targeted_videos_with_ads(age_group, gender, ethnicity)
            print(f"Ads data: {ads_data}")
            if ads_data:
                self.current_ads_list = []  # Reset the ad list
                for ad in ads_data:
                    ad_path = ad[0]  # Ad file path
                    absolute_ad_path = os.path.join(os.path.dirname(__file__), ad_path)
                    if os.path.exists(absolute_ad_path):
                        # Add ad with its associated priority score (ad[2])
                        self.current_ads_list.append((absolute_ad_path, ad[2]))
                        print(f"Added to ad pool: {absolute_ad_path}")
                    else:
                        print(f"Ad file not found: {absolute_ad_path}")
                self.current_ad = None  # Clear current ad selection
                print(f"Updated ad pool: {self.current_ads_list}")
            else:
                # Fallback to loading all ads if no targeted ads are available
                print(f"No ads found for demographic: {age_group}, {gender}, {ethnicity}")
                self.load_all_ads()

    def get_random_ad(self):
        """Select and return a random advertisement from the ad pool.

        Returns:
            str or None: The path to the selected ad, or None if the ad pool is empty.

        This method randomly chooses an ad from `current_ads_list`, updates `current_ad`,
        and resets the global watch time counter.
        """
        with self.lock:  # Ensure thread-safe access to the ad pool
            if not self.current_ads_list:
                return None  # Return None if no ads are available
            # Randomly select an ad path from the list, ignoring priority scores
            self.current_ad = random.choice([path for path, _ in self.current_ads_list])
            print(f"Selected ad: {self.current_ad}")
            # Reset global watch time using the shared lock
            global total_watch_time
            with watching_lock:
                total_watch_time = 0
            return self.current_ad