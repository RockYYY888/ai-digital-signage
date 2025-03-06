import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import sqlite3
import plotly.express as px

conn = sqlite3.connect('advertisements.db')

query = """
SELECT 
    v.viewer_id,
    v.view_time,
    d.gender,
    d.age_group,
    d.ethnicity,
    v.visit_date,
    v.ad_id
FROM viewers v
JOIN demographics d ON v.demographics_id = d.demographics_id
JOIN ads a ON v.ad_id = a.ad_id;
"""

data = pd.read_sql(query, conn)

conn.close()

data['visit_date'] = pd.to_datetime(data['visit_date'])


def satisfaction_level(view_time):
    if 0 <= view_time <= 2:
        return 'Very Unsatisfied'
    elif 2 < view_time <= 4:
        return 'Unsatisfied'
    elif 4 < view_time <= 6:
        return 'Neutral'
    elif 6 < view_time <= 8:
        return 'Satisfied'
    else:
        return 'Very Satisfied'


data['satisfaction'] = data['view_time'].apply(satisfaction_level)

total_visits = data.shape[0]

today_date = data['visit_date'].max()
today_data = data[data['visit_date'] == today_date]
today_visits = today_data.shape[0]
today_avg_time = round(today_data['view_time'].mean(), 2)

overall_satisfaction_counts = data['satisfaction'].value_counts().reindex(
    ['Very Unsatisfied', 'Unsatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'],
    fill_value=0
)

data['ad_id'] = 'AD-' + data['ad_id'].astype(str)

ad_avg_time = data.groupby('ad_id')['view_time'].mean().reset_index()

ad_avg_time['ad_id_num'] = ad_avg_time['ad_id'].astype(str).str.extract('(\d+)').astype(int)
ad_avg_time = ad_avg_time.sort_values('ad_id_num')
ad_avg_time['ad_id'] = 'AD-' + ad_avg_time['ad_id_num'].astype(str)

gender_satisfaction_counts = data.groupby(['gender', 'satisfaction']).size().unstack().fillna(0)
age_satisfaction_counts = data.groupby(['age_group', 'satisfaction']).size().unstack().fillna(0)
ethnicity_satisfaction_counts = data.groupby(['ethnicity', 'satisfaction']).size().unstack().fillna(0)

data['year'] = data['visit_date'].dt.isocalendar().year
data['week'] = data['visit_date'].dt.isocalendar().week
weekly_ad_avg_time = data.groupby(['ad_id', 'year', 'week'])['view_time'].mean().reset_index()
weekly_ad_avg_time['year_week'] = weekly_ad_avg_time['year'].astype(str) + '-W' + weekly_ad_avg_time['week'].astype(str).str.zfill(2)

weekly_ad_avg_time['ad_id_num'] = weekly_ad_avg_time['ad_id'].astype(str).str.extract('(\d+)').astype(int)
weekly_ad_avg_time = weekly_ad_avg_time.sort_values('ad_id_num')
weekly_ad_avg_time['ad_id'] = 'AD-' + weekly_ad_avg_time['ad_id_num'].astype(str)

data['month'] = data['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')
monthly_ad_avg_time = data.groupby(['ad_id', 'month'])['view_time'].mean().reset_index()

monthly_ad_avg_time['ad_id_num'] = monthly_ad_avg_time['ad_id'].astype(str).str.extract('(\d+)').astype(int)
monthly_ad_avg_time = monthly_ad_avg_time.sort_values(['ad_id_num', 'month'])
monthly_ad_avg_time['ad_id'] = 'AD-' + monthly_ad_avg_time['ad_id_num'].astype(str)

color_palette = ["#FF6B6B", "#FFD930", "#6BCB77", "#4D96FF", "#9955FF"]

app = dash.Dash(__name__)


def create_bar_chart(data_counts, title, legend_title, colors):
    fig = go.Figure()
    for i, group in enumerate(data_counts.index):
        fig.add_trace(go.Bar(
            x=data_counts.columns,
            y=data_counts.loc[group],
            name=group,
            marker_color=colors[i],
            legendgroup=legend_title,
            showlegend=True,
            legendgrouptitle_text=legend_title
        ))
    fig.update_layout(
        title=title,
        paper_bgcolor='#1f2c56',
        plot_bgcolor='#1f2c56',
        xaxis={'color': 'white'},
        yaxis={'title': 'Count', 'color': 'white'},
        font={'color': 'white'}
    )
    return fig


app.layout = html.Div([
    # First line：title
    html.Div([
        html.Div([
            html.H3("Advertisement Analytics Dashboard",
                    style={
                        "margin-bottom": "0px",
                        'color': '#00ffcc',
                        'textAlign': 'center',
                        'width': '100%',
                        'font-size': '3.5rem',
                        'letter-spacing': '0.1rem',
                    }),
        ], className="twelve columns", id="title", style={'width': '100%'}),
    ], id="header", className="row flex-display", style={
        "margin-bottom": "25px",
        "display": "flex",
        "justify-content": "center",
        "align-items": "center",
        "width": "100%",
        'padding': '20px 0'
    }),
    # the second line：calendar selector and basic information
    html.Div([
        html.Div([
            html.Div([
                html.H6('Total Viewers', style={'textAlign': 'center', 'color': 'white'}),
            ]),
            html.Div([
                html.P(id='total-viewers-all', children=f"{total_visits:,.0f}", style={'textAlign': 'center', 'color': 'orange', 'fontSize': 40})
            ]),
            html.P('Select Date:', className='fix_label', style={'color': 'white'}),
            dcc.DatePickerSingle(
                id='date-picker',
                date=today_date.strftime('%Y-%m-%d'),
                min_date_allowed=data['visit_date'].min().strftime('%Y-%m-%d'),
                max_date_allowed=data['visit_date'].max().strftime('%Y-%m-%d'),
                className='custom-date-picker',
                style={
                    'background-color': '#1f2c56',
                    'color': 'white',
                    'width': '100%',
                    'border-radius': '5px',
                    'border': '1px solid #white',
                    'font-size': '16px',
                    'z-index': '100'
                }
            ),
            html.Div([
                html.H6('Viewers of Selected Day', style={'textAlign': 'center', 'color': 'white'}),
            ]),
            html.Div([
                html.P(id='total-viewers-selected', children=f"{today_visits:,.0f}", style={'textAlign': 'center', 'color': 'orange', 'fontSize': 40})
            ]),
            html.Div([
                html.H6('Avg Watching Time', style={'textAlign': 'center', 'color': 'white'}),
            ]),
            html.Div([
                html.P(id='avg-watch-time', children=f"{today_avg_time} s", style={'textAlign': 'center', 'color': 'orange', 'fontSize': 40})
            ]),

        ], className="create_container three columns", id="cross-filter-options"),
        html.Div([
            dcc.Tabs(
                id='tabs',
                value='gender-tab',
                children=[
                    dcc.Tab(label='Gender Satisfaction', value='gender-tab', children=[
                        dcc.Graph(id='gender-satisfaction-chart', figure=create_bar_chart(gender_satisfaction_counts, "Gender Satisfaction", "Gender", color_palette[:len(gender_satisfaction_counts.index)]))
                    ]),
                    dcc.Tab(label='Age Satisfaction', value='age-tab', children=[
                        dcc.Graph(id='age-satisfaction-chart', figure=create_bar_chart(age_satisfaction_counts, "Age Group Satisfaction", "Age Group", color_palette[:len(age_satisfaction_counts.index)]))
                    ]),
                    dcc.Tab(label='Ethnicity Satisfaction', value='ethnicity-tab', children=[
                        dcc.Graph(id='ethnicity-satisfaction-chart', figure=create_bar_chart(ethnicity_satisfaction_counts, "Ethnicity Satisfaction", "Ethnicity", color_palette[:len(ethnicity_satisfaction_counts.index)]))
                    ]),
                ],
                style={
                    'background-color': '#1f2c56',
                    'color': 'white',
                    'font-family': 'Open Sans, HelveticaNeue, Helvetica Neue, Helvetica, Arial, sans-serif',
                    'border': 'none'
                },
                colors={
                    "border": "#1f2c56",
                    "primary": "white",
                    "background": "#1f2c56",
                    "selected": "#192444"

                }
            ),
        ], className="create_container four columns"),
        html.Div([
            dcc.Graph(
                id='pie-chart',
                figure={
                    'data': [go.Pie(
                        labels=overall_satisfaction_counts.index,
                        values=overall_satisfaction_counts.values,
                        hole=0.5,
                        marker={'colors': ["#FF6B6B", "#9955FF", "#6BCB77", "#4D96FF", "#FFD930"]},
                        textinfo='percent',
                        textfont={'size': 16},
                        textposition='auto'
                    )],
                    'layout': go.Layout(
                        title='Overall Satisfaction',
                        paper_bgcolor='#1f2c56',
                        plot_bgcolor='#1f2c56',
                        font={'color': 'white', 'size': 18},
                        height=500,
                        width=500,
                        margin=dict(l=0, r=0, t=50, b=100),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.3,
                            xanchor="center",
                            x=0.5
                        ),
                        autosize=True
                    )
                }
            ),
        ], className="create_container five columns"),
    ], className="row flex-display"),
    # The third line：line chart
    html.Div([
        html.Div([
            html.P('Select Ad:', className='fix_label', style={'color': 'white'}),
            dcc.Dropdown(
                id='ad-dropdown',
                options=[{'label': ad_id, 'value': ad_id} for ad_id in weekly_ad_avg_time['ad_id'].unique()],
                value=weekly_ad_avg_time['ad_id'].unique()[0],
                style={
                    'background-color': '#1f2c56',
                    'color': 'white',
                    'optionHeight': 30,
                    '--dropdown-option-background-color': 'grey',
                    '--dropdown-option-color': 'white'
                }
            ),
            html.P('Select Time Granularity:', className='fix_label', style={'color': 'white'}),
            dcc.RadioItems(
                id='time-granularity',
                options=[
                    {'label': 'Daily', 'value': 'daily'},
                    {'label': 'Weekly', 'value': 'weekly'},
                    {'label': 'Monthly', 'value': 'monthly'}
                ],
                value='daily',
                labelStyle={'display': 'inline-block', 'color': 'white', 'margin-right': '10px'}
            ),
            dcc.Graph(
                id='line-chart',
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                }
            ),
            html.Div(id='slider-container', style={'marginTop': '20px'}),
            dcc.Store(id='time-axis-store')
        ], className="create_container1 twelve columns"),
    ], className="row flex-display"),
], id="mainContainer", style={"display": "flex", "flex-direction": "column"})


# callback function: Update metrics and charts based on selected dates
@app.callback(
    [
        Output('total-viewers-selected', 'children'),
        Output('avg-watch-time', 'children'),
        Output('gender-satisfaction-chart', 'figure'),
        Output('age-satisfaction-chart', 'figure'),
        Output('ethnicity-satisfaction-chart', 'figure'),
        Output('pie-chart', 'figure')
    ],
    Input('date-picker', 'date')
)
def update_all(selected_date):
    selected_date = pd.to_datetime(selected_date)
    daily_data = data[data['visit_date'] == selected_date]
    daily_visits = daily_data.shape[0]
    daily_avg_time = round(daily_data['view_time'].mean(), 2) if daily_visits > 0 else 0

    # The satisfaction distribution is recalculated
    daily_data['satisfaction'] = daily_data['view_time'].apply(satisfaction_level)
    daily_overall_satisfaction_counts = daily_data['satisfaction'].value_counts().reindex(
        ['Very Unsatisfied', 'Unsatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'],
        fill_value=0
    )
    daily_gender_satisfaction_counts = daily_data.groupby(['gender', 'satisfaction']).size().unstack().fillna(0)
    daily_age_satisfaction_counts = daily_data.groupby(['age_group', 'satisfaction']).size().unstack().fillna(0)
    daily_ethnicity_satisfaction_counts = daily_data.groupby(['ethnicity', 'satisfaction']).size().unstack().fillna(0)

    # Generate a new chart
    gender_chart = create_bar_chart(daily_gender_satisfaction_counts, "Gender Satisfaction", "Gender", color_palette[:len(daily_gender_satisfaction_counts.index)])
    age_chart = create_bar_chart(daily_age_satisfaction_counts, "Age Group Satisfaction", "Age Group", color_palette[:len(daily_age_satisfaction_counts.index)])
    ethnicity_chart = create_bar_chart(daily_ethnicity_satisfaction_counts, "Ethnicity Satisfaction", "Ethnicity", color_palette[:len(daily_ethnicity_satisfaction_counts.index)])
    pie_chart = {
        'data': [go.Pie(
            labels=daily_overall_satisfaction_counts.index,
            values=daily_overall_satisfaction_counts.values,
            hole=0.4,
            marker={'colors': ["#FF6B6B", "#9955FF", "#6BCB77", "#4D96FF", "#FFD930"]},
            textinfo='percent',
            textfont={'size': 16},
            textposition='auto'
        )],
        'layout': go.Layout(
            title='Overall Satisfaction',
            paper_bgcolor='#1f2c56',
            plot_bgcolor='#1f2c56',
            font={'color': 'white'},
            margin=dict(l=0, r=0, t=50, b=100),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            autosize=True
        )
    }

    return (
        f"{daily_visits:,.0f}",
        f"{daily_avg_time} s",
        gender_chart,
        age_chart,
        ethnicity_chart,
        pie_chart
    )


# ...前面的代码保持不变...

@app.callback(
    [Output('time-axis-store', 'data'),
     Output('slider-container', 'children')],
    [Input('ad-dropdown', 'value'),
     Input('time-granularity', 'value')]
)
def generate_time_axis(ad_id, time_granularity):
    # Get data on the corresponding ads
    filtered_data = data[data['ad_id'] == ad_id]

    time_points = []
    if time_granularity == 'daily':
        min_date = filtered_data['visit_date'].min()
        max_date = filtered_data['visit_date'].max()
        all_dates = pd.date_range(min_date, max_date).strftime('%Y-%m-%d')
        time_points = sorted(all_dates)
    elif time_granularity == 'weekly':
        weekly_data = weekly_ad_avg_time[weekly_ad_avg_time['ad_id'] == ad_id]
        min_week = weekly_data['year_week'].min()
        max_week = weekly_data['year_week'].max()
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
        monthly_data = monthly_ad_avg_time[monthly_ad_avg_time['ad_id'] == ad_id]
        min_month = monthly_data['month'].min()
        max_month = monthly_data['month'].max()
        all_months = pd.date_range(min_month, max_month, freq='MS').strftime('%Y-%m')
        time_points = sorted(all_months)

    # If there is no data, return an empty timeline and prompt
    if len(time_points) == 0:
        return [], html.Div("NO DATA", style={'color': 'white'})

    # Create a single - handle slider
    slider = dcc.Slider(
        id='time-slider',
        min=0,
        max=len(time_points) - 1,
        step=1,
        value=len(time_points) - 1,
        marks={i: time_points[i] for i in range(0, len(time_points), max(1, len(time_points) // 5))},
        tooltip={"placement": "bottom", "always_visible": True}
    )

    return time_points, slider


# callback function of line chart
@app.callback(
    Output('line-chart', 'figure'),
    [Input('ad-dropdown', 'value'),
     Input('time-granularity', 'value'),
     Input('time-slider', 'value'),
     Input('time-axis-store', 'data')]
)
def update_line_chart(ad_id, time_granularity, slider_value, time_points):
    if not time_points or slider_value is None:
        return go.Figure()

    # Define the time window size
    window_size = 7
    start_idx = max(0, slider_value - window_size + 1)
    end_idx = slider_value + 1

    # Extract the point in time of the current window
    display_time_points = time_points[start_idx:end_idx]

    fig = go.Figure()

    if time_granularity == 'daily':
        min_date = pd.to_datetime(display_time_points[0])
        max_date = pd.to_datetime(display_time_points[-1])
        all_dates = pd.date_range(min_date, max_date)
        filtered_data = data[(data['ad_id'] == ad_id) &
                             (data['visit_date'].isin(all_dates))]
        agg_data = filtered_data.groupby('visit_date')['view_time'].mean().reindex(all_dates).fillna(0).reset_index()
        agg_data.columns = ['visit_date', 'view_time']
        x_data = agg_data['visit_date']
        y_data = agg_data['view_time']
        title_suffix = 'Daily Average View Time'

    elif time_granularity == 'weekly':
        weekly_data = weekly_ad_avg_time[(weekly_ad_avg_time['ad_id'] == ad_id) &
                                         (weekly_ad_avg_time['year_week'].isin(display_time_points))]
        all_weeks = pd.Series(display_time_points)
        agg_data = weekly_data.groupby('year_week')['view_time'].mean().reindex(all_weeks).fillna(0).reset_index()
        agg_data.columns = ['year_week', 'view_time']
        x_data = agg_data['year_week']
        y_data = agg_data['view_time']
        title_suffix = 'Weekly Average View Time'

    elif time_granularity == 'monthly':
        monthly_data = monthly_ad_avg_time[(monthly_ad_avg_time['ad_id'] == ad_id) &
                                           (monthly_ad_avg_time['month'].isin(display_time_points))]
        all_months = pd.Series(display_time_points)
        agg_data = monthly_data.groupby('month')['view_time'].mean().reindex(all_months).fillna(0).reset_index()
        agg_data.columns = ['month', 'view_time']
        x_data = agg_data['month']
        y_data = agg_data['view_time']
        title_suffix = 'Monthly Average View Time'

    # Add a line chart track
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines+markers',
        line={'color': '#FFA500', 'width': 3},
        marker={'size': 10, 'color': 'orange'},
        name='View Time'
    ))

    # update the layout
    fig.update_layout(
        title={'text': f'{ad_id} {title_suffix}', 'y': 0.9, 'x': 0.5},
        xaxis={'title': 'Time', 'gridcolor': '#2a3f6f', 'color': 'white'},
        yaxis={'title': 'View Time (seconds)', 'gridcolor': '#2a3f6f', 'color': 'white'},
        plot_bgcolor='#1f2c56',
        paper_bgcolor='#1f2c56',
        font={'color': 'white', 'size': 14},
        margin={'t': 60, 'b': 100, 'l': 80, 'r': 40},
        hovermode='x unified'
    )

    return fig


if __name__ == '__main__':
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link rel="stylesheet" href="/assets/your_css_filename.css"> 
            <style>
                body {
                    background-color: #192444;
                    margin: 0;
                    padding: 20px;
                }

                .SingleDatePicker_picker {
                    position: absolute !important;
                    top: 100px !important;
                    left: 20px !important;
                    z-index: 1000 !important;
                }

                .DayPicker {
                    width: auto !important;
                    min-width: 220px !important;
                    font-family: 'HelveticaNeue', Helvetica !important;
                }

                .CalendarMonth {
                    padding: 0 10px !important;
                    font-family: 'HelveticaNeue', Helvetica !important;
                }
                
                .CalendarDay {
                    font-size: 14px !important;
                    padding: 0 !important;
                    height: 40px !important;
                    width: 40px !important;
                    line-height: 40px !important;
                    font-family: 'HelveticaNeue', Helvetica !important;
                }
                
                .DateInput {
                    width: 100% !important;
                }
                
                .DateInput_input {
                    background-color: #1f2c56 !important;
                    color: white !important;
                    font-size: 16px !important;
                    width: 100% !important;
                    font-family: 'HelveticaNeue', Helvetica !important;
                }
                
                .SingleDatePickerInput {
                    background-color: #1f2c56 !important;
                    border: 1px solid white !important;
                    width: 100% !important;
                    font-family: 'HelveticaNeue', Helvetica !important;
                }
                
                .CalendarMonth_caption {
                    padding-top: 22px !important;
                    padding-bottom: 37px !important;
                    font-family: 'HelveticaNeue', Helvetica !important;
                }
                
                .DayPicker_weekHeader {
                    padding: 0 !important;
                    top: 62px !important;
                    font-family: 'HelveticaNeue', Helvetica !important;
                }

                .DayPicker_weekHeader_li {
                    text-align: center !important;
                    width: 40px !important;
                    font-family: 'HelveticaNeue', Helvetica !important;
                }

                .DayPicker_weekHeader_ul {
                    display: flex !important;
                    justify-content: center !important;
                    padding: 0 8px !important;
                    font-family: 'HelveticaNeue', Helvetica !important;
                }
                
                .CalendarMonth_table {
                    margin-top: 10px !important;
                }
                
                .DayPicker_transitionContainer {
                    min-height: 320px !important;
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
    '''
    app.run_server(debug=True, dev_tools_ui=False, dev_tools_props_check=False,port=5002)