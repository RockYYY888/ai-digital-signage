# test/dashboardTest.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from flask import Flask
from Dashboard.dashboard import get_fresh_data, create_bar_chart, create_pie_chart, init_dashboard
import sqlite3

# create database
def create_temp_db():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE demographics (
        demographics_id INTEGER PRIMARY KEY,
        gender TEXT,
        age_group TEXT,
        ethnicity TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE ads (
        ad_id INTEGER PRIMARY KEY,
        duration REAL
    )
    """)
    cursor.execute("""
    CREATE TABLE viewers (
        viewer_id INTEGER PRIMARY KEY,
        demographics_id INTEGER,
        ad_id INTEGER,
        view_time REAL,
        visit_date TEXT,
        FOREIGN KEY(demographics_id) REFERENCES demographics(demographics_id),
        FOREIGN KEY(ad_id) REFERENCES ads(ad_id)
    )
    """)
    cursor.execute("INSERT INTO demographics VALUES (1, 'Male', '20-30', 'Asian')")
    cursor.execute("INSERT INTO ads VALUES (1, 30.0)")
    cursor.execute("INSERT INTO viewers VALUES (1, 1, 1, 15.0, '2023-10-01')")
    conn.commit()
    return conn

# get_fresh_data 
def test_get_fresh_data():
    conn = create_temp_db()
    with patch('Dashboard.dashboard.sqlite3.connect', return_value=conn):
        data = get_fresh_data()
        assert not data.empty, "data not null"
        assert data['viewer_id'].iloc[0] == 1, "viewer_id is 1"
        assert data['gender'].iloc[0] == 'Male', "gender is 'Male'"
        assert data['completion_rate'].iloc[0] == 0.5, "completion_rate is 0.5"

#  create_bar_chart 
def test_create_bar_chart():
    data = pd.DataFrame({
        'gender': ['Male', 'Female'],
        'completion_level': ['0-20%', '20-40%'],
        'count': [10, 20]
    })
    counts = data.pivot(index='gender', columns='completion_level', values='count').fillna(0)
    fig = create_bar_chart(counts, "Test Bar Chart", "Gender", ["#FF6B6B", "#4D96FF"])
    assert fig.layout.title.text == "Test Bar Chart", " 'Test Bar Chart'"
    assert len(fig.data) == 2, "2 bar chart"

#  create_pie_chart 
def test_create_pie_chart():
    counts = pd.Series([10, 20, 30], index=['0-20%', '20-40%', '40-60%'])
    fig = create_pie_chart(counts)
    assert fig.layout.title.text == 'Overall Completion Rate Distribution', "'Overall Completion Rate Distribution'"
    assert len([d for d in fig.data if d.type == 'pie']) == 1, "pie chart"

#  init_dashboard 
def test_init_dashboard():
    server = Flask(__name__)
    with patch('Dashboard.dashboard.get_fresh_data') as mock_get_fresh_data:
        mock_data = pd.DataFrame({
            'viewer_id': [1],
            'visit_date': [pd.Timestamp('2023-10-01')],
            'completion_rate': [0.5],
            'ad_id': ['AD-1'],
            'view_time': [15.0],
            'duration': [30.0]
        })
        mock_get_fresh_data.return_value = mock_data
        dash_app = init_dashboard(server)
    assert dash_app.layout is not None, "not none"
    assert 'total-viewers-all' in str(dash_app.layout), "should contain total-viewers-all"
    assert 'date-picker' in str(dash_app.layout), "date-picker"


if __name__ == "__main__":
    pytest.main(["-v"])