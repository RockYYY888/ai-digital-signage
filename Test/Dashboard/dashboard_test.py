# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.
# test/dashboardTest.py

import pytest  
import pandas as pd  
from unittest.mock import patch, MagicMock 
from flask import Flask 
from Dashboard.dashboard import get_fresh_data, create_bar_chart, create_pie_chart, init_dashboard  
import sqlite3  

# create database
def create_temp_db():
    """Create a temporary in-memory SQLite database for testing.

    Returns:
        sqlite3.Connection: An in-memory database connection with sample data.
    """
    conn = sqlite3.connect(':memory:')  # Create an in-memory SQLite database
    cursor = conn.cursor()  # Create a cursor for executing SQL commands
    cursor.execute("""
    CREATE TABLE demographics (
        demographics_id INTEGER PRIMARY KEY,
        gender TEXT,
        age_group TEXT,
        ethnicity TEXT
    )
    """)  # Create demographics table
    cursor.execute("""
    CREATE TABLE ads (
        ad_id INTEGER PRIMARY KEY,
        duration REAL
    )
    """)  # Create ads table
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
    """)  # Create viewers table with foreign keys
    cursor.execute("INSERT INTO demographics VALUES (1, 'Male', '20-30', 'Asian')")  # Insert sample demographic data
    cursor.execute("INSERT INTO ads VALUES (1, 30.0)")  # Insert sample ad data
    cursor.execute("INSERT INTO viewers VALUES (1, 1, 1, 15.0, '2023-10-01')")  # Insert sample viewer data
    conn.commit()  # Commit changes to the database
    return conn  # Return the database connection

# get_fresh_data 
def test_get_fresh_data():
    """Test the get_fresh_data function with a mock database.

    Verifies that get_fresh_data returns a non-empty DataFrame with correct data values.
    """
    conn = create_temp_db()  # Create a temporary database
    with patch('Dashboard.dashboard.sqlite3.connect', return_value=conn):  # Mock sqlite3.connect to use in-memory DB
        data = get_fresh_data()  # Call get_fresh_data with mocked connection
        assert not data.empty, "data not null"  # Ensure DataFrame is not empty
        assert data['viewer_id'].iloc[0] == 1, "viewer_id is 1"  # Verify viewer_id matches inserted value
        assert data['gender'].iloc[0] == 'Male', "gender is 'Male'"  # Verify gender matches inserted value
        assert data['completion_rate'].iloc[0] == 0.5, "completion_rate is 0.5"  # Verify completion_rate is calculated correctly

#  create_bar_chart 
def test_create_bar_chart():
    """Test the create_bar_chart function with sample data.

    Verifies that create_bar_chart generates a bar chart with the correct title and number of traces.
    """
    data = pd.DataFrame({
        'gender': ['Male', 'Female'],
        'completion_level': ['0-20%', '20-40%'],
        'count': [10, 20]
    })  # Create sample data for testing
    counts = data.pivot(index='gender', columns='completion_level', values='count').fillna(0)  # Pivot data into a counts matrix
    fig = create_bar_chart(counts, "Test Bar Chart", "Gender", ["#FF6B6B", "#4D96FF"])  # Generate bar chart
    assert fig.layout.title.text == "Test Bar Chart", " 'Test Bar Chart'"  # Verify chart title
    assert len(fig.data) == 2, "2 bar chart"  # Verify two bar traces (one per gender)

#  create_pie_chart 
def test_create_pie_chart():
    """Test the create_pie_chart function with sample data.

    Verifies that create_pie_chart generates a pie chart with the correct title and a single pie trace.
    """
    counts = pd.Series([10, 20, 30], index=['0-20%', '20-40%', '40-60%'])  # Create sample completion level counts
    fig = create_pie_chart(counts)  # Generate pie chart
    assert fig.layout.title.text == 'Overall Completion Rate Distribution', "'Overall Completion Rate Distribution'"  # Verify chart title
    assert len([d for d in fig.data if d.type == 'pie']) == 1, "pie chart"  # Verify one pie trace exists

#  init_dashboard 
def test_init_dashboard():
    """Test the init_dashboard function with mocked data.

    Verifies that init_dashboard initializes a Dash app with a non-null layout containing expected components.
    """
    server = Flask(__name__)  # Create a Flask server instance
    with patch('Dashboard.dashboard.get_fresh_data') as mock_get_fresh_data:  # Mock get_fresh_data function
        mock_data = pd.DataFrame({
            'viewer_id': [1],
            'visit_date': [pd.Timestamp('2023-10-01')],
            'completion_rate': [0.5],
            'ad_id': ['AD-1'],
            'view_time': [15.0],
            'duration': [30.0]
        })  # Create mock data for dashboard
        mock_get_fresh_data.return_value = mock_data  # Configure mock to return sample data
        dash_app = init_dashboard(server)  # Initialize the dashboard
    assert dash_app.layout is not None, "not none"  # Ensure layout is initialized
    assert 'total-viewers-all' in str(dash_app.layout), "should contain total-viewers-all"  # Verify total-viewers-all component exists
    assert 'date-picker' in str(dash_app.layout), "date-picker"  # Verify date-picker component exists

if __name__ == "__main__":
    pytest.main(["-v"])  # Run pytest with verbose output when script is executed directly