import os
import random
import sqlite3
from ad_pool.video_selection import get_targeted_videos_with_ads
import threading
import time
watching_lock = threading.Lock()
class AdPool:
    def __init__(self):
        self.current_ad = None
        self.current_ads_list = []  # 存储广告列表
        self.lock = threading.Lock()
        self.db_file = os.path.join(os.path.dirname(__file__), 'advertisements.db')  # 确保路径正确
        # 初始化时加载所有广告
        self.load_all_ads()

    def load_all_ads(self):
        """加载数据库中的所有广告"""
        connection = sqlite3.connect(self.db_file)
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT ad_content FROM ads;")
            results = cursor.fetchall()
            self.current_ads_list = []  # 清空现有列表
            for result in results:
                ad_path = result[0]
                # 更新为指向 ad_pool/videos 文件夹
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
        """根据人群特征更新广告池"""
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
                self.load_all_ads()  # 回退到加载所有广告

    def get_random_ad(self):
        """从广告池中随机选择一个广告"""
        with self.lock:
            if not self.current_ads_list:
                return None
            # 随机选择一个广告
            self.current_ad = random.choice([path for path, _ in self.current_ads_list])
            print(f"Selected ad: {self.current_ad}")
            # 重置观看时间
            global total_watch_time
            with watching_lock:
                total_watch_time = 0
            return self.current_ad