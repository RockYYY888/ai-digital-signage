# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

import pandas as pd  
import plotly.graph_objects as go  
import dash 
from dash import dcc, html 
from dash.dependencies import Input, Output  
from util import get_resource_path  
import sqlite3  
from flask import Flask 

# Path to the SQLite database file
db_path = get_resource_path("advertisements.db")

def get_fresh_data():
    """Get the latest data from the database and preprocess it.

    Returns:
        pandas.DataFrame: Processed data with viewer, demographic, and ad information.
    """
    conn = sqlite3.connect(db_path)  # Establish connection to the database
    query = """
    SELECT 
        v.viewer_id,
        v.view_time,
        d.gender,
        d.age_group,
        d.ethnicity,
        v.visit_date,
        v.ad_id,
        a.duration
    FROM viewers v
    JOIN demographics d ON v.demographics_id = d.demographics_id
    JOIN ads a ON v.ad_id = a.ad_id;
    """  # SQL query to fetch viewer, demographic, and ad data
    data = pd.read_sql(query, conn)  # Load query results into a DataFrame
    conn.close()  # Close the database connection

    # Data preprocessing
    data['visit_date'] = pd.to_datetime(data['visit_date'])  # Convert visit_date to datetime
    data['completion_rate'] = data['view_time'] / data['duration']  # Calculate completion rate
    data.loc[data['completion_rate'] > 1, 'completion_rate'] = 0.5  # Cap completion rate at 0.5 if exceeds 1
    data['ad_id'] = 'AD-' + data['ad_id'].astype(str)  # Prefix ad_id with 'AD-'

    def completion_rate_level(completion_rate):
        # Categorize completion rate into discrete levels
        if 0 <= completion_rate <= 0.2:
            return '0-20%'
        elif 0.2 < completion_rate <= 0.4:
            return '20-40%'
        elif 0.4 < completion_rate <= 0.6:
            return '40-60%'
        elif 0.6 < completion_rate <= 0.8:
            return '60-80%'
        else:
            return '80-100%'

    data['completion_level'] = data['completion_rate'].apply(completion_rate_level)  # Add completion level column
    return data

# Color Palette
color_palette = ["#FF6B6B", "#FFD930", "#6BCB77", "#4D96FF", "#9955FF"]  # Colors for visualizations

# =============== Utility function: Building a bar chart ===============

def create_bar_chart(data_counts, title, legend_title, colors):
    # Create a bar chart from grouped data counts
    fig = go.Figure()
    
    for i, group in enumerate(data_counts.index):
        fig.add_trace(go.Bar(
            x=data_counts.columns,  # X-axis: completion rate ranges
            y=data_counts.loc[group],  # Y-axis: counts for each group
            name=group,  # Legend entry name
            marker_color=colors[i % len(colors)],  # Cycle through color palette
            legendgroup=legend_title,  # Group legend entries
            showlegend=True,  # Show in legend
            legendgrouptitle_text=legend_title if i == 0 else None  # Add title to legend group once
        ))
    
    fig.update_layout(
        title={
            'text': title,  # Chart title
            'x': 0.5, 'xanchor': 'center',  # Center the title
            'font': {'color': 'white'}  # White title text
        },
        plot_bgcolor='rgba(255, 255, 255, 0)',  # Transparent plot background
        paper_bgcolor='rgba(255, 255, 255, 0)',  # Transparent paper background
        xaxis={'title': 'Completion Rate Range', 'color': 'white'},  # X-axis configuration
        yaxis={'title': 'Count', 'color': 'white'},  # Y-axis configuration
        font={'color': 'white'},  # White text throughout
        dragmode=False,  # Disable dragging
        legend=dict(
            font=dict(color='white'),  # White legend text
            bgcolor='rgba(255, 255, 255, 0)',  # Transparent legend background
            groupclick="toggleitem"  # Toggle legend items individually
        )
    )
    return fig

def create_no_data_figure(title, legend_title, all_groups, colors):
    # Create a placeholder figure for when no data is available
    fig = go.Figure()

    for i, group in enumerate(all_groups):
        fig.add_trace(go.Bar(
            x=[''], y=[0],  # Empty bar for placeholder
            name=group,  # Legend entry name
            marker_color=colors[i % len(colors)],  # Cycle through colors
            showlegend=True,  # Show in legend
            legendgrouptitle_text=legend_title if i == 0 else None,  # Add legend title once
            legendgroup=legend_title,
            visible='legendonly'  # Hide bar, show only in legend
        ))

    fig.add_annotation(
        text="NO DATA",  # Display "NO DATA" message
        xref="paper", yref="paper", x=0.5, y=0.5,  # Center the annotation
        showarrow=False, font=dict(size=24, color="white")  # Large white text
    )

    fig.update_layout(
        title={
            'text': title, 'font': {'color': 'white'},
            'x': 0.5, 'xanchor': 'center'  # Center the title
        },  
        plot_bgcolor='rgba(255, 255, 255, 0)', paper_bgcolor='rgba(255, 255, 255, 0)',  # Transparent backgrounds
        xaxis={'visible': False}, yaxis={'visible': False},  # Hide axes
        legend=dict(
            title=None, font=dict(color='white'), bgcolor='rgba(255, 255, 255, 0)',
            groupclick="toggleitem"  # Toggle legend items
        )
    )
    return fig

def create_pie_chart(data_counts=None, show_no_data=False):
    # Create a pie chart for completion rate distribution or a no-data placeholder
    all_levels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']  # All possible completion levels
    colors = ["#FF6B6B", "#4D96FF", "#6BCB77", "#9955FF", "#FFD930"]  # Pie chart colors
    
    if show_no_data or data_counts is None or data_counts.sum() == 0:
        fig = go.Figure()
        
        fig.add_annotation(
            text="NO DATA", xref="paper", yref="paper", x=0.5, y=0.5,  # Center "NO DATA" message
            showarrow=False, font=dict(size=24, color="white")
        )
        
        for i, level in enumerate(all_levels):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',  # Placeholder for legend
                marker=dict(color=colors[i], size=10), name=level, showlegend=True
            ))
    else:
        filtered_data = {k: v for k, v in data_counts.items() if v > 0}  # Filter out zero counts
        
        if not filtered_data:  
            fig = go.Figure()
            fig.add_annotation(
                text="NO DATA", xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=24, color="white")
            )
            
            for i, level in enumerate(all_levels):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(color=colors[i], size=10), name=level, showlegend=True
                ))
        else:
            fig = go.Figure(go.Pie(
                labels=list(filtered_data.keys()), values=list(filtered_data.values()),  # Pie chart data
                hole=0.3,  # Donut chart style
                marker=dict(colors=[colors[all_levels.index(k)] for k in filtered_data.keys()]),  # Match colors
                textinfo='percent', textfont={'size': 16}, textposition='auto', showlegend=True
            ))
            
            missing_levels = [level for level in all_levels if level not in filtered_data]  # Add missing levels to legend
            for level in missing_levels:
                i = all_levels.index(level)
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(color=colors[i], size=10), name=level, showlegend=True
                ))
    
    fig.update_layout(
        title={
            'text': 'Overall Completion Rate Distribution', 'x': 0.5, 'xanchor': 'center',
            'font': {'color': 'white'}
        },
        plot_bgcolor='rgba(255, 255, 255, 0)', paper_bgcolor='rgba(255, 255, 255, 0)',  # Transparent backgrounds
        font={'color': 'white'}, margin=dict(l=0, r=0, t=50, b=120),  # White text, custom margins
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.45, xanchor="center", x=0.5,  # Horizontal legend below chart
            font=dict(color='white'), bgcolor='rgba(255, 255, 255, 0)', itemsizing='constant'
        ),
        autosize=True, showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False)  # Hide axes
    )
    
    return fig

# =============== Initialize the dashboard ===============

def init_dashboard(server: Flask):
    """Create and return a Dash application that supports dynamic data refresh.

    Args:
        server (Flask): The Flask server instance to integrate with Dash.

    Returns:
        dash.Dash: The initialized Dash application instance.
    """
    dash_app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname='/dashboard/',  # Base URL path for the dashboard
    )

    # Custom HTML templates
    dash_app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link rel="stylesheet" href="style.css">
            <style>
                body {
                    background-color: #192444;
                    margin: 0;
                    padding: 20px;
                }
                .SingleDatePicker_picker { position: absolute !important; top: 100px !important; left: 20px !important; z-index: 1000 !important; }
                .DayPicker { width: auto !important; min-width: 220px !important; font-family: Verdana !important; }
                .CalendarMonth { padding: 0 10px !important; font-family: Verdana !important; }
                .CalendarDay { font-size: 14px !important; padding: 0 !important; height: 40px !important; width: 40px !important; line-height: 40px !important; font-family: Verdana !important; }
                .DateInput { width: 100% !important; }
                .DateInput_input { background-color: #1f2c56 !important; color: white !important; font-size: 16px !important; width: 100% !important; font-family: Verdana !important; cursor: pointer !important; }
                .SingleDatePickerInput { background-color: #1f2c56 !important; border: 1px solid white !important; width: 100% !important; font-family: Verdana !important; }
                .CalendarMonth_caption { padding-top: 22px !important; padding-bottom: 37px !important; font-family: Verdana !important; }
                .DayPicker_weekHeader { padding: 0 !important; top: 62px !important; font-family: Verdana !important; }
                .DayPicker_weekHeader_li { text-align: center !important; width: 40px !important; font-family: Verdana !important; }
                .DayPicker_weekHeader_ul { display: flex !important; justify-content: center !important; padding: 0 8px !important; font-family: Verdana !important; }
                .CalendarMonth_table { margin-top: 10px !important; }
                .DayPicker_transitionContainer { min-height: 320px !important; }
                button:hover {
                    background-color: #3a5f9f !important;
                    transition: background-color 0.3s ease;
                }
                button:active {
                    background-color: #1f2c56 !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''  # Custom HTML template with CSS styling for the dashboard

    # Get initial data for layout initialization
    data = get_fresh_data()  # Fetch initial dataset
    total_visits = data.shape[0]  # Total number of viewer records
    today_date = data['visit_date'].max()  # Most recent date in the dataset
    today_data = data.loc[data['visit_date'] == today_date].copy()  # Filter data for today
    today_visits = today_data.shape[0]  # Number of visits today
    today_avg_completion_rate = round(today_data['completion_rate'].mean() * 100, 2) if today_visits > 0 else 0  # Average completion rate today

    # Initial weekly and monthly data
    data['year'] = data['visit_date'].dt.isocalendar().year  # Extract year for weekly grouping
    data['week'] = data['visit_date'].dt.isocalendar().week  # Extract week for weekly grouping
    weekly_ad_avg_completion = data.groupby(['ad_id', 'year', 'week'])['completion_rate'].mean().reset_index()  # Weekly average completion rates
    weekly_ad_avg_completion['year_week'] = weekly_ad_avg_completion['year'].astype(str) + '-W' + weekly_ad_avg_completion['week'].astype(str).str.zfill(2)  # Format year-week
    data['month'] = data['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')  # Extract month for monthly grouping
    monthly_ad_avg_completion = data.groupby(['ad_id', 'month'])['completion_rate'].mean().reset_index()  # Monthly average completion rates

    # ========== Dash Layout ==========
    dash_app.layout = html.Div([
        # Header
        html.Div([
            html.H3("Advertisement Analytics Dashboard",
                    style={"margin-bottom": "0px", 'color': '#00ffcc', 'textAlign': 'center', 'width': '100%', 'font-size': '3.5rem', 'letter-spacing': '0.1rem'}),
            html.Img(src=dash_app.get_asset_url('refresh.PNG'), id="refresh-button", n_clicks=0,
                     style={
                         'backgroundColor': '#2a3f6f', 'color': 'white', 'border': '1px solid #4a6faf',
                         'borderRadius': '15px', 'padding': '8px 16px', 'cursor': 'pointer', 'outline': 'none',
                         'height': '20px', 'lineHeight': '24px', 'fontFamily': 'Verdana', 'display': 'inline-block',
                         'margin-left': '20px', 'position': 'absolute', 'right': '20px', 'top': '20px'
                     })  # Refresh button image
        ], style={"margin-bottom": "15px", "display": "flex", "justify-content": "center", "align-items": "center", "position": "relative"}),

        # Date picker and basic info
        html.Div([
            html.Div([
                html.Div([
                    html.H6('Total Viewers', style={'textAlign': 'center', 'color': 'white', 'font-family': 'Verdana', 'font-size': '16px'}),
                    html.P(id='total-viewers-all', children=f"{total_visits:,.0f}",
                           style={'textAlign': 'center', 'color': 'orange', 'fontSize': 32})  # Total viewers display
                ], className="info-card", style={
                    'backgroundColor': '#1f2c56', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px',
                }),

                html.Div([
                    html.P('Select Date:', className='fix_label', style={'color': 'white', 'textAlign': 'center'}),
                    dcc.DatePickerSingle(
                        id='date-picker', date=today_date.strftime('%Y-%m-%d'),  # Date picker for selecting a date
                        style={'backgroundColor': '#1f2c56', 'color': 'white', 'width': '100%', 'borderRadius': '5px',
                               'border': '1px solid #white', 'fontSize': '16px', 'zIndex': '100'}
                    ),
                    html.H6('Viewers of Selected Day', style={'textAlign': 'center', 'color': 'white', 'font-family': 'Verdana', 'marginTop': '20px', 'font-size': '16px'}),
                    html.P(id='total-viewers-selected', children=f"{today_visits:,.0f}",
                           style={'textAlign': 'center', 'color': 'orange', 'fontSize': 32}),  # Viewers for selected date
                    html.H6('Avg Completion Rate', style={'textAlign': 'center', 'color': 'white', 'font-family': 'Verdana', 'marginTop': '20px', 'font-size': '16px'}),
                    html.P(id='avg-completion-rate', children=f"{today_avg_completion_rate}%",
                           style={'textAlign': 'center', 'color': 'orange', 'fontSize': 32}),  # Average completion rate
                ], className="info-card", style={
                    'backgroundColor': '#1f2c56', 'padding': '20px', 'borderRadius': '10px',
                })
            ], className="three columns", id="cross-filter-options"),

            # Bar charts
            html.Div([
                dcc.Tabs(id='tabs', value='gender-tab', children=[
                    dcc.Tab(label='Gender Completion', value='gender-tab', children=[
                        dcc.Graph(id='gender-completion-chart', config={'scrollZoom': False, 'displayModeBar': False})  # Gender bar chart
                    ]),
                    dcc.Tab(label='Age Completion', value='age-tab', children=[
                        dcc.Graph(id='age-completion-chart', config={'scrollZoom': False, 'displayModeBar': False})  # Age bar chart
                    ]),
                    dcc.Tab(label='Ethnicity Completion', value='ethnicity-tab', children=[
                        dcc.Graph(id='ethnicity-completion-chart', config={'scrollZoom': False, 'displayModeBar': False})  # Ethnicity bar chart
                    ]),
                ], style={'backgroundColor': '#1f2c56', 'color': 'white', 'fontFamily': 'Verdana', 'border': 'none'},
                   colors={"border": "#1f2c56", "primary": "white", "background": "#1f2c56", "selected": "#192444"}),
            ], className="create_container four columns"),

            # Pie chart
            html.Div([
                dcc.Graph(id='pie-chart', config={'displayModeBar': False}),  # Pie chart for completion rate distribution
            ], className="create_container five columns"),
        ], className="row flex-display"),

        # Line chart with custom buttons
        html.Div([
            html.Div([
                html.P('Select Ad:', className='fix_label', style={'color': 'white'}),
                dcc.Dropdown(id='ad-dropdown',  # Initially empty, dynamically updated
                             style={'background-color': '#1f2c56', 'color': 'white', 'optionHeight': 30}),  # Ad selection dropdown
                html.Div([
                    html.P('Select Time Granularity:', className='fix_label',
                           style={'color': 'white', 'textAlign': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Button('Daily', id='daily-button', n_clicks=0, style={
                            'backgroundColor': '#2a3f6f', 'color': 'white', 'border': '1px solid #4a6faf',
                            'borderRadius': '15px', 'padding': '8px 16px', 'marginRight': '10px', 'fontFamily': 'Verdana',
                            'cursor': 'pointer', 'outline': 'none', 'height': '40px', 'lineHeight': '24px',
                            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
                        }),  # Daily granularity button
                        html.Button('Weekly', id='weekly-button', n_clicks=0, style={
                            'backgroundColor': '#2a3f6f', 'color': 'white', 'border': '1px solid #4a6faf',
                            'borderRadius': '15px', 'padding': '8px 16px', 'marginRight': '10px', 'fontFamily': 'Verdana',
                            'cursor': 'pointer', 'outline': 'none', 'height': '40px', 'lineHeight': '24px',
                            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
                        }),  # Weekly granularity button
                        html.Button('Monthly', id='monthly-button', n_clicks=0, style={
                            'backgroundColor': '#2a3f6f', 'color': 'white', 'border': '1px solid #4a6faf',
                            'borderRadius': '15px', 'padding': '8px 16px', 'fontFamily': 'Verdana',
                            'cursor': 'pointer', 'outline': 'none', 'height': '40px', 'lineHeight': '24px',
                            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
                        }),  # Monthly granularity button
                    ], style={'display': 'flex', 'justifyContent': 'center'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                dcc.Store(id='time-granularity-store', data='daily'),  # Store for time granularity state
                dcc.Graph(id='line-chart', config={'scrollZoom': False, 'displayModeBar': True,
                                                  'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']}),  # Line chart
                html.Div(id='slider-container', style={'marginTop': '20px'}),  # Container for time slider
                dcc.Store(id='time-axis-store'),  # Store for time axis data
                dcc.Store(id='granularity-for-chart', data='daily'),  # Store for chart granularity
                html.P(id='granularity-display', style={'color': 'white', 'textAlign': 'center'})  # Display current granularity
            ], className="create_container1 twelve columns"),
        ], className="row flex-display"),
    ], id="mainContainer", style={"display": "flex", "flex-direction": "column"})  # Main dashboard container

    # ============== Callback Function ==============
    @dash_app.callback(
        [
            Output('total-viewers-selected', 'children'),
            Output('avg-completion-rate', 'children'),
            Output('gender-completion-chart', 'figure'),
            Output('age-completion-chart', 'figure'),
            Output('ethnicity-completion-chart', 'figure'),
            Output('pie-chart', 'figure'),
            Output('total-viewers-all', 'children'),
            Output('ad-dropdown', 'options'),
            Output('ad-dropdown', 'value'),
            Output('date-picker', 'min_date_allowed'),
            Output('date-picker', 'max_date_allowed')
        ],
        [
            Input('date-picker', 'date'),
            Input('refresh-button', 'n_clicks')
        ]
    )
    def update_all(selected_date, refresh_clicks):
        # Update dashboard components based on selected date or refresh
        fresh = get_fresh_data()  # Fetch latest data
        total_visits = fresh.shape[0]  # Total number of visits

        min_date = fresh['visit_date'].min().strftime('%Y-%m-%d')  # Earliest date in dataset
        max_date = fresh['visit_date'].max().strftime('%Y-%m-%d')  # Latest date in dataset

        selected_date = pd.to_datetime(selected_date).date()  # Convert selected date to date object
        daily_data = fresh.loc[fresh['visit_date'].dt.date == selected_date].copy()  # Filter data for selected date

        all_genders = ['Male', 'Female']  # All possible gender categories
        all_age_groups = ['17-35', '35-50', '50+']  # All possible age groups
        all_ethnicities = ['White', 'Black', 'Asian', 'Indian', 'Other']  # All possible ethnicities
        all_completion_levels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']  # All completion rate levels

        if daily_data.empty:
            print(f"No data for selected date: {selected_date}")  # Log no-data condition
            no_data_gender = create_no_data_figure(
                "Gender Completion Rate Distribution", "Gender", all_genders, color_palette
            )
            no_data_age = create_no_data_figure(
                "Age Group Completion Rate Distribution", "Age Group", all_age_groups, color_palette
            )
            no_data_eth = create_no_data_figure(
                "Ethnicity Completion Rate Distribution", "Ethnicity", all_ethnicities, color_palette
            )
            
            pie_chart = create_pie_chart(show_no_data=True)  # Create no-data pie chart

            ad_options = [{'label': ad_id, 'value': ad_id} for ad_id in sorted(fresh['ad_id'].unique(), key=lambda x: int(x.split('-')[1]))]  # Ad dropdown options
            ad_value = ad_options[0]['value'] if ad_options else None  # Default ad selection

            return (
                "0", "0%", no_data_gender, no_data_age, no_data_eth, pie_chart,
                f"{total_visits:,.0f}", ad_options, ad_value, min_date, max_date
            )

        daily_visits = daily_data.shape[0]  # Number of visits on selected date
        daily_avg_rate = round(daily_data['completion_rate'].mean() * 100, 2) if daily_visits > 0 else 0  # Average completion rate

        daily_data['completion_level'] = daily_data['completion_rate'].apply(
            lambda cr: '0-20%' if cr <= 0.2 else '20-40%' if cr <= 0.4 else '40-60%' if cr <= 0.6 else '60-80%' if cr <= 0.8 else '80-100%'
        )  # Assign completion levels

        gender_counts = daily_data.groupby(['gender', 'completion_level']).size().unstack(fill_value=0)  # Gender completion counts
        gender_counts = gender_counts.reindex(index=all_genders, columns=all_completion_levels, fill_value=0)  # Reindex with all categories
        age_counts = daily_data.groupby(['age_group', 'completion_level']).size().unstack(fill_value=0)  # Age completion counts
        age_counts = age_counts.reindex(index=all_age_groups, columns=all_completion_levels, fill_value=0)  # Reindex with all categories
        eth_counts = daily_data.groupby(['ethnicity', 'completion_level']).size().unstack(fill_value=0)  # Ethnicity completion counts
        eth_counts = eth_counts.reindex(index=all_ethnicities, columns=all_completion_levels, fill_value=0)  # Reindex with all categories
        
        overall_counts = daily_data['completion_level'].value_counts().reindex(all_completion_levels, fill_value=0)  # Overall completion counts

        gender_chart = create_bar_chart(gender_counts, "Gender Completion Rate Distribution", "Gender", color_palette)  # Gender bar chart
        age_chart = create_bar_chart(age_counts, "Age Group Completion Rate Distribution", "Age Group", color_palette)  # Age bar chart
        eth_chart = create_bar_chart(eth_counts, "Ethnicity Completion Rate Distribution", "Ethnicity", color_palette)  # Ethnicity bar chart
        
        pie_chart = create_pie_chart(overall_counts)  # Pie chart with actual data

        ad_options = [{'label': ad_id, 'value': ad_id} for ad_id in sorted(fresh['ad_id'].unique(), key=lambda x: int(x.split('-')[1]))]  # Ad dropdown options
        ad_value = ad_options[0]['value'] if ad_options else None  # Default ad selection

        return (
            f"{daily_visits:,.0f}", f"{daily_avg_rate}%", gender_chart, age_chart, eth_chart, pie_chart,
            f"{total_visits:,.0f}", ad_options, ad_value, min_date, max_date
        )

    @dash_app.callback(
        Output('time-granularity-store', 'data'),
        [
            Input('daily-button', 'n_clicks'),
            Input('weekly-button', 'n_clicks'),
            Input('monthly-button', 'n_clicks')
        ],
        prevent_initial_call=True
    )
    def update_time_granularity(daily_clicks, weekly_clicks, monthly_clicks):
        # Update time granularity based on button clicks
        ctx = dash.callback_context
        if not ctx.triggered:
            return 'daily'  # Default to daily if no trigger
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'daily-button':
            return 'daily'
        elif button_id == 'weekly-button':
            return 'weekly'
        elif button_id == 'monthly-button':
            return 'monthly'
        return 'daily'  # Fallback to daily

    @dash_app.callback(
        [Output('time-axis-store', 'data'), Output('slider-container', 'children'), Output('granularity-for-chart', 'data')],
        [Input('ad-dropdown', 'value'), Input('time-granularity-store', 'data')]
    )
    def generate_time_axis(ad_id, time_granularity):
        """Generate timeline and slider, support refresh."""
        fresh = get_fresh_data()  # Fetch latest data
        filtered_data = fresh[fresh['ad_id'] == ad_id]  # Filter by selected ad
        time_points = []

        if time_granularity == 'daily':
            min_date = filtered_data['visit_date'].min()  # Earliest date for ad
            max_date = filtered_data['visit_date'].max()  # Latest date for ad
            if pd.isnull(min_date) or pd.isnull(max_date):
                return [], html.Div("NO DATA", style={'color': 'white'}), time_granularity  # No data case
            all_dates = pd.date_range(min_date, max_date).strftime('%Y-%m-%d')  # All dates in range
            time_points = sorted(all_dates)
        elif time_granularity == 'weekly':
            fresh['year'] = fresh['visit_date'].dt.isocalendar().year  # Extract year
            fresh['week'] = fresh['visit_date'].dt.isocalendar().week  # Extract week
            weekly_data = fresh.groupby(['ad_id', 'year', 'week'])['completion_rate'].mean().reset_index()  # Weekly averages
            weekly_data['year_week'] = weekly_data['year'].astype(str) + '-W' + weekly_data['week'].astype(str).str.zfill(2)  # Format year-week
            weekly_data_filtered = weekly_data[weekly_data['ad_id'] == ad_id]  # Filter by ad
            if weekly_data_filtered.empty:
                return [], html.Div("NO DATA", style={'color': 'white'}), time_granularity  # No data case
            min_week = weekly_data_filtered['year_week'].min()  # Earliest week
            max_week = weekly_data_filtered['year_week'].max()  # Latest week
            all_weeks = []
            current_week = min_week
            while current_week <= max_week:
                all_weeks.append(current_week)
                year, week_num = map(int, current_week.split('-W'))
                new_date = pd.Timestamp(f'{year}-01-01') + pd.Timedelta(weeks=week_num)
                next_year = new_date.isocalendar().year
                next_week = new_date.isocalendar().week
                current_week = f'{next_year}-W{str(next_week).zfill(2)}'
            time_points = sorted(all_weeks)
        elif time_granularity == 'monthly':
            fresh['month'] = fresh['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')  # Extract month
            monthly_data = fresh.groupby(['ad_id', 'month'])['completion_rate'].mean().reset_index()  # Monthly averages
            monthly_data_filtered = monthly_data[monthly_data['ad_id'] == ad_id]  # Filter by ad
            if monthly_data_filtered.empty:
                return [], html.Div("NO DATA", style={'color': 'white'}), time_granularity  # No data case
            min_month = monthly_data_filtered['month'].min()  # Earliest month
            max_month = monthly_data_filtered['month'].max()  # Latest month
            all_months = pd.date_range(min_month, max_month, freq='MS').strftime('%Y-%m')  # All months in range
            time_points = sorted(all_months)

        if len(time_points) == 0:
            return [], html.Div("NO DATA", style={'color': 'white'}), time_granularity  # No data case

        slider = dcc.Slider(
            id='time-slider', min=0, max=len(time_points) - 1, step=1, value=len(time_points) - 1,  # Time slider
            marks={i: time_points[i] for i in range(0, len(time_points), max(1, len(time_points) // 5))},  # Mark every 5th point
            tooltip={"placement": "bottom", "always_visible": True}
        )

        return time_points, slider, time_granularity

    @dash_app.callback(
        Output('line-chart', 'figure'),
        [
            Input('ad-dropdown', 'value'),
            Input('granularity-for-chart', 'data'),
            Input('time-slider', 'value'),
            Input('time-axis-store', 'data')
        ]
    )
    def update_line_chart(ad_id, time_granularity, slider_value, time_points):
        """Update line chart to support refresh and keep old version logic."""
        if not time_points or slider_value is None:
            return go.Figure()  # Return empty figure if no data

        fresh = get_fresh_data()  # Fetch latest data
        window_size = 14  # Size of the sliding window
        start_idx = max(0, slider_value - window_size + 1)  # Start index of window
        end_idx = slider_value + 1  # End index of window
        display_time_points = time_points[start_idx:end_idx]  # Time points to display

        fig = go.Figure()

        if time_granularity == 'daily':
            min_date = pd.to_datetime(display_time_points[0])  # Start date
            max_date = pd.to_datetime(display_time_points[-1])  # End date
            all_dates = pd.date_range(min_date, max_date)  # All dates in range
            filtered_data_period = fresh[(fresh['ad_id'] == ad_id) & (fresh['visit_date'].isin(all_dates))]  # Filter data
            agg_data = filtered_data_period.groupby('visit_date')['completion_rate'].mean().reindex(all_dates).fillna(0).reset_index()  # Aggregate daily data
            agg_data.columns = ['visit_date', 'completion_rate']
            x_data = agg_data['visit_date']
            y_data = agg_data['completion_rate'] * 100  # Convert to percentage
            title_suffix = 'Daily Completion Rate Trend (In Targeted Displaying Mode)'
        elif time_granularity == 'weekly':
            fresh['year'] = fresh['visit_date'].dt.isocalendar().year
            fresh['week'] = fresh['visit_date'].dt.isocalendar().week
            weekly_data = fresh.groupby(['ad_id', 'year', 'week'])['completion_rate'].mean().reset_index()  # Weekly averages
            weekly_data['year_week'] = weekly_data['year'].astype(str) + '-W' + weekly_data['week'].astype(str).str.zfill(2)
            weekly_data_filtered = weekly_data[(weekly_data['ad_id'] == ad_id) &
                                              (weekly_data['year_week'].isin(display_time_points))]  # Filter weekly data
            all_weeks_series = pd.Series(display_time_points)
            agg_data = weekly_data_filtered.groupby('year_week')['completion_rate'].mean().reindex(all_weeks_series).fillna(0).reset_index()
            agg_data.columns = ['year_week', 'completion_rate']
            x_data = agg_data['year_week']
            y_data = agg_data['completion_rate'] * 100  # Convert to percentage
            title_suffix = 'Weekly Completion Rate Trend (In Targeted Displaying Mode)'
        else:  # monthly
            fresh['month'] = fresh['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')
            monthly_data = fresh.groupby(['ad_id', 'month'])['completion_rate'].mean().reset_index()  # Monthly averages
            monthly_data_filtered = monthly_data[(monthly_data['ad_id'] == ad_id) &
                                                (monthly_data['month'].isin(display_time_points))]  # Filter monthly data
            all_months_series = pd.Series(display_time_points)
            agg_data = monthly_data_filtered.groupby('month')['completion_rate'].mean().reindex(all_months_series).fillna(0).reset_index()
            agg_data.columns = ['month', 'completion_rate']
            x_data = agg_data['month']
            y_data = agg_data['completion_rate'] * 100  # Convert to percentage
            title_suffix = 'Monthly Completion Rate Trend (In Targeted Displaying Mode)'

        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, mode='lines+markers',  # Line chart with markers
            line={'color': '#FFA500', 'width': 3}, marker={'size': 10, 'color': 'orange'}, name='Completion Rate'
        ))
        fig.update_layout(
            title={'text': f'{ad_id} {title_suffix}', 'y': 0.9, 'x': 0.5},  # Centered title
            xaxis={'title': 'Time', 'gridcolor': '#2a3f6f', 'color': 'white'},  # X-axis configuration
            yaxis={'title': 'Completion Rate (%)', 'gridcolor': '#2a3f6f', 'color': 'white'},  # Y-axis configuration
            plot_bgcolor='#1f2c56', paper_bgcolor='#1f2c56', font={'color': 'white', 'size': 14},  # Styling
            margin={'t': 60, 'b': 100, 'l': 80, 'r': 40}, hovermode='x unified', dragmode=False
        )
        return fig

    @dash_app.callback(
        Output('granularity-display', 'children'),
        Input('time-granularity-store', 'data')
    )
    def display_granularity(granularity):
        """Display the current time granularity."""
        return f"Current granularity: {granularity}"

    return dash_app  # Correctly return Dash instance