import os
import sys


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 运行时的临时目录
        print(f"Temporary directory: {sys._MEIPASS}")
        print(f"Files in temporary directory: {os.listdir(sys._MEIPASS)}")
        cv_dir = os.path.join(sys._MEIPASS, 'CV')
        if os.path.exists(cv_dir):
            print(f"Files in CV directory: {os.listdir(cv_dir)}")
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.path.abspath("."), relative_path)