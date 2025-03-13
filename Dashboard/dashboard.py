import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np

import sqlite3
from flask import Flask  # 仅用于类型注释或参考

# ================ 数据加载与预处理 ====================

db_path = "advertisements.db"
conn = sqlite3.connect(db_path)

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
"""
data = pd.read_sql(query, conn)
conn.close()

# Data preprocessing
data['visit_date'] = pd.to_datetime(data['visit_date'])
data['completion_rate'] = data['view_time'] / data['duration']
data['ad_id'] = 'AD-' + data['ad_id'].astype(str)

def completion_rate_level(completion_rate):
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

data['completion_level'] = data['completion_rate'].apply(completion_rate_level)

total_visits = data.shape[0]

today_date = data['visit_date'].max()
today_data = data.loc[data['visit_date'] == today_date].copy()
today_visits = today_data.shape[0]
today_avg_completion_rate = round(today_data['completion_rate'].mean() * 100, 2)

ad_avg_completion_rate = data.groupby('ad_id')['completion_rate'].mean().reset_index()
ad_avg_completion_rate['ad_id_num'] = ad_avg_completion_rate['ad_id'].str.extract('(\d+)').astype(int)
ad_avg_completion_rate = ad_avg_completion_rate.sort_values('ad_id_num')
ad_avg_completion_rate['ad_id'] = 'AD-' + ad_avg_completion_rate['ad_id_num'].astype(str)

gender_completion_counts = data.groupby(['gender', 'completion_level']).size().unstack().fillna(0)
age_completion_counts = data.groupby(['age_group', 'completion_level']).size().unstack().fillna(0)
ethnicity_completion_counts = data.groupby(['ethnicity', 'completion_level']).size().unstack().fillna(0)

overall_completion_counts = data['completion_level'].value_counts().reindex(['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], fill_value=0)

data['year'] = data['visit_date'].dt.isocalendar().year
data['week'] = data['visit_date'].dt.isocalendar().week
weekly_ad_avg_completion = data.groupby(['ad_id', 'year', 'week'])['completion_rate'].mean().reset_index()
weekly_ad_avg_completion['year_week'] = weekly_ad_avg_completion['year'].astype(str) + '-W' + weekly_ad_avg_completion['week'].astype(str).str.zfill(2)
weekly_ad_avg_completion['ad_id_num'] = weekly_ad_avg_completion['ad_id'].str.extract('(\d+)').astype(int)
weekly_ad_avg_completion = weekly_ad_avg_completion.sort_values('ad_id_num')
weekly_ad_avg_completion['ad_id'] = 'AD-' + weekly_ad_avg_completion['ad_id_num'].astype(str)

data['month'] = data['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')
monthly_ad_avg_completion = data.groupby(['ad_id', 'month'])['completion_rate'].mean().reset_index()
monthly_ad_avg_completion['ad_id_num'] = monthly_ad_avg_completion['ad_id'].str.extract('(\d+)').astype(int)
monthly_ad_avg_completion = monthly_ad_avg_completion.sort_values(['ad_id_num', 'month'])
monthly_ad_avg_completion['ad_id'] = 'AD-' + monthly_ad_avg_completion['ad_id_num'].astype(str)

color_palette = ["#FF6B6B", "#FFD930", "#6BCB77", "#4D96FF", "#9955FF"]

# =============== 工具函数：构建条形图 ===============

def create_bar_chart(data_counts, title, legend_title, colors):
    fig = go.Figure()
    for i, group in enumerate(data_counts.index):
        fig.add_trace(go.Bar(
            x=data_counts.columns,
            y=data_counts.loc[group],
            name=group,
            marker_color=colors[i % len(colors)],
            legendgroup=legend_title,
            showlegend=True,
            legendgrouptitle_text=legend_title
        ))
    fig.update_layout(
        title=title,
        plot_bgcolor='rgba(255, 255, 255, 0)',  # 设置图表背景为透明
        paper_bgcolor='rgba(255, 255, 255, 0)',
        xaxis={'title': 'Completion Rate Range', 'color': 'white'},
        yaxis={'title': 'Count', 'color': 'white'},
        font={'color': 'white'},
        dragmode=False  # Disable drag interaction (e.g., box selection for zoom)
    )
    return fig

def init_dashboard(server: Flask):
    """
    创建 Dash 实例，绑定到传入的 Flask `server`，并返回这个 Dash app。
    这样可以在同一个端口运行 Flask + Dash，通过 /dashboard/ 访问此页面。
    """
    dash_app = dash.Dash(
        __name__,
        server=server,      # 绑定到主 Flask 应用
        url_base_pathname='/dashboard/'
    )

    # 自定义 HTML 模板（可选）
    dash_app.index_string = '''
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
    '''

    # ========== Dash Layout ==========
    dash_app.layout = html.Div([
        # Header
        html.Div([
            html.H3("Advertisement Analytics Dashboard",
                    style={"margin-bottom": "0px", 'color': '#00ffcc', 'textAlign': 'center', 'width': '100%', 'font-size': '3.5rem', 'letter-spacing': '0.1rem'}),
        ], id="header", className="row flex-display",
            style={"margin-bottom": "25px", "display": "flex", "justify-content": "center", "align-items": "center", "width": "100%", 'padding': '20px 0'}),

        # Date picker and basic info
        html.Div([
            html.Div([
                html.Div([
                    html.H6('Total Viewers', style={'textAlign': 'center', 'color': 'white', 'font-family': 'Verdana', 'font-size': '16px'}),
                    html.P(id='total-viewers-all', children=f"{total_visits:,.0f}",
                           style={'textAlign': 'center', 'color': 'orange', 'fontSize': 32})
                ], className="info-card", style={
                    'backgroundColor': '#1f2c56',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'marginBottom': '20px',
                }),

                html.Div([
                    html.P('Select Date:', className='fix_label', style={'color': 'white', 'textAlign': 'center'}),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        date=today_date.strftime('%Y-%m-%d'),
                        min_date_allowed=data['visit_date'].min().strftime('%Y-%m-%d'),
                        max_date_allowed=data['visit_date'].max().strftime('%Y-%m-%d'),
                        style={
                            'backgroundColor': '#1f2c56',
                            'color': 'white',
                            'width': '100%',
                            'borderRadius': '5px',
                            'border': '1px solid #white',
                            'fontSize': '16px',
                            'zIndex': '100'
                        }
                    ),
                    html.H6('Viewers of Selected Day', style={'textAlign': 'center', 'color': 'white', 'font-family': 'Verdana', 'marginTop': '20px', 'font-size': '16px'}),
                    html.P(id='total-viewers-selected', children=f"{today_visits:,.0f}",
                           style={'textAlign': 'center', 'color': 'orange', 'fontSize': 32}),
                    html.H6('Avg Completion Rate', style={'textAlign': 'center', 'color': 'white', 'font-family': 'Verdana', 'marginTop': '20px', 'font-size': '16px'}),
                    html.P(id='avg-completion-rate', children=f"{today_avg_completion_rate}%",
                           style={'textAlign': 'center', 'color': 'orange', 'fontSize': 32}),
                ], className="info-card", style={
                    'backgroundColor': '#1f2c56',
                    'padding': '20px',
                    'borderRadius': '10px',
                })
            ], className="three columns", id="cross-filter-options"),

            # Bar charts with fixed configuration
            html.Div([
                dcc.Tabs(id='tabs', value='gender-tab', children=[
                    dcc.Tab(label='Gender Completion', value='gender-tab', children=[
                        dcc.Graph(
                            id='gender-completion-chart',
                            figure=create_bar_chart(gender_completion_counts, "Gender Completion Rate Distribution", "Gender", color_palette),
                            config={
                                'scrollZoom': False,
                                'displayModeBar': False,  # Hide mode bar to prevent interaction
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
                            }
                        )
                    ]),
                    dcc.Tab(label='Age Completion', value='age-tab', children=[
                        dcc.Graph(
                            id='age-completion-chart',
                            figure=create_bar_chart(age_completion_counts, "Age Group Completion Rate Distribution", "Age Group", color_palette),
                            config={
                                'scrollZoom': False,
                                'displayModeBar': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
                            }
                        )
                    ]),
                    dcc.Tab(label='Ethnicity Completion', value='ethnicity-tab', children=[
                        dcc.Graph(
                            id='ethnicity-completion-chart',
                            figure=create_bar_chart(ethnicity_completion_counts, "Ethnicity Completion Rate Distribution", "Ethnicity", color_palette),
                            config={
                                'scrollZoom': False,
                                'displayModeBar': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
                            }
                        )
                    ]),
                ], style={'backgroundColor': '#1f2c56', 'color': 'white', 'fontFamily': 'Verdana', 'border': 'none'},
                         colors={"border": "#1f2c56", "primary": "white", "background": "#1f2c56", "selected": "#192444"}),
            ], className="create_container four columns"),

            # Pie chart
            html.Div([
                dcc.Graph(id='pie-chart', figure={
                    'data': [go.Pie(labels=overall_completion_counts.index, values=overall_completion_counts.values, hole=0.3,
                                    marker={'colors': color_palette}, textinfo='percent', textfont={'size': 16}, textposition='auto')],
                    'layout': go.Layout(
                        title='Overall Completion Rate Distribution',
                        plot_bgcolor='rgba(255, 255, 255, 0)',  # 设置图表背景为透明
                        paper_bgcolor='rgba(255, 255, 255, 0)',
                        font={'color': 'white', 'size': 18},
                        height=500,
                        width=500,
                        margin=dict(l=0, r=0, t=50, b=100),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                        autosize=True
                    )
                }),
            ], className="create_container five columns"),
        ], className="row flex-display"),

        # Line chart with custom buttons
        html.Div([
            html.Div([
                html.P('Select Ad:', className='fix_label', style={'color': 'white'}),
                dcc.Dropdown(id='ad-dropdown',
                             options=[{'label': ad_id, 'value': ad_id} for ad_id in weekly_ad_avg_completion['ad_id'].unique()],
                             value=weekly_ad_avg_completion['ad_id'].unique()[0],
                             style={'background-color': '#1f2c56', 'color': 'white', 'optionHeight': 30}),
                html.Div([
                    html.P('Select Time Granularity:', className='fix_label',
                           style={'color': 'white', 'textAlign': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Button('Daily', id='daily-button', n_clicks=0, style={
                            'backgroundColor': '#2a3f6f',
                            'color': 'white',
                            'border': '1px solid #4a6faf',
                            'borderRadius': '15px',
                            'padding': '8px 16px',
                            'marginRight': '10px',
                            'fontFamily': 'Verdana',
                            'cursor': 'pointer',
                            'outline': 'none',
                            'height': '40px',
                            'lineHeight': '24px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center'
                        }),
                        html.Button('Weekly', id='weekly-button', n_clicks=0, style={
                            'backgroundColor': '#2a3f6f',
                            'color': 'white',
                            'border': '1px solid #4a6faf',
                            'borderRadius': '15px',
                            'padding': '8px 16px',
                            'marginRight': '10px',
                            'fontFamily': 'Verdana',
                            'cursor': 'pointer',
                            'outline': 'none',
                            'height': '40px',
                            'lineHeight': '24px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center'
                        }),
                        html.Button('Monthly', id='monthly-button', n_clicks=0, style={
                            'backgroundColor': '#2a3f6f',
                            'color': 'white',
                            'border': '1px solid #4a6faf',
                            'borderRadius': '15px',
                            'padding': '8px 16px',
                            'fontFamily': 'Verdana',
                            'cursor': 'pointer',
                            'outline': 'none',
                            'height': '40px',
                            'lineHeight': '24px',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center'
                        }),
                    ], style={'display': 'flex', 'justifyContent': 'center'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                dcc.Store(id='time-granularity-store', data='daily'),
                dcc.Graph(
                    id='line-chart',
                    config={
                        'scrollZoom': False,
                        'displayModeBar': True,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
                    }
                ),
                html.Div(id='slider-container', style={'marginTop': '20px'}),
                dcc.Store(id='time-axis-store')
            ], className="create_container1 twelve columns"),
        ], className="row flex-display"),
    ], id="mainContainer", style={"display": "flex", "flex-direction": "column"})

    # ============== 回调函数 ==============

    @dash_app.callback(
        [
            Output('total-viewers-selected', 'children'),
            Output('avg-completion-rate', 'children'),
            Output('gender-completion-chart', 'figure'),
            Output('age-completion-chart', 'figure'),
            Output('ethnicity-completion-chart', 'figure'),
            Output('pie-chart', 'figure')
        ],
        Input('date-picker', 'date')
    )
    def update_all(selected_date):
        selected_date = pd.to_datetime(selected_date)
        daily_data = data.loc[data['visit_date'] == selected_date].copy()

        daily_visits = daily_data.shape[0]
        daily_avg_completion_rate = round(daily_data['completion_rate'].mean() * 100, 2) if daily_visits > 0 else 0

        daily_data['completion_level'] = daily_data['completion_rate'].apply(completion_rate_level)
        daily_gender_completion_counts = daily_data.groupby(['gender', 'completion_level']).size().unstack().fillna(0)
        daily_age_completion_counts = daily_data.groupby(['age_group', 'completion_level']).size().unstack().fillna(0)
        daily_ethnicity_completion_counts = daily_data.groupby(['ethnicity', 'completion_level']).size().unstack().fillna(0)
        daily_overall_completion_counts = daily_data['completion_level'].value_counts().reindex(
            ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], fill_value=0
        )

        # Create charts with dragmode='none'
        gender_chart = create_bar_chart(daily_gender_completion_counts, "Gender Completion Rate Distribution", "Gender", color_palette)
        age_chart = create_bar_chart(daily_age_completion_counts, "Age Group Completion Rate Distribution", "Age Group", color_palette)
        ethnicity_chart = create_bar_chart(daily_ethnicity_completion_counts, "Ethnicity Completion Rate Distribution", "Ethnicity", color_palette)

        pie_chart = {
            'data': [
                go.Pie(
                    labels=daily_overall_completion_counts.index,
                    values=daily_overall_completion_counts.values,
                    hole=0.3,
                    marker={'colors': color_palette},
                    textinfo='percent',
                    textfont={'size': 16},
                    textposition='auto'
                )
            ],
            'layout': go.Layout(
                title='Overall Completion Rate Distribution',
                plot_bgcolor='rgba(255, 255, 255, 0)',  # 设置图表背景为透明
                paper_bgcolor='rgba(255, 255, 255, 0)',
                font={'color': 'white'},
                margin=dict(l=0, r=0, t=50, b=100),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                autosize=True
            )
        }

        return (
            f"{daily_visits:,.0f}",
            f"{daily_avg_completion_rate}%",
            gender_chart,
            age_chart,
            ethnicity_chart,
            pie_chart
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
        ctx = dash.callback_context
        if not ctx.triggered:
            return 'daily'
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'daily-button':
            return 'daily'
        elif button_id == 'weekly-button':
            return 'weekly'
        elif button_id == 'monthly-button':
            return 'monthly'
        return 'daily'  # Default fallback

    @dash_app.callback(
        [Output('time-axis-store', 'data'), Output('slider-container', 'children')],
        [Input('ad-dropdown', 'value'), Input('time-granularity-store', 'data')]
    )
    def generate_time_axis(ad_id, time_granularity):
        filtered_data = data[data['ad_id'] == ad_id]
        time_points = []

        if time_granularity == 'daily':
            min_date = filtered_data['visit_date'].min()
            max_date = filtered_data['visit_date'].max()
            all_dates = pd.date_range(min_date, max_date).strftime('%Y-%m-%d')
            time_points = sorted(all_dates)
        elif time_granularity == 'weekly':
            weekly_data_filtered = weekly_ad_avg_completion[weekly_ad_avg_completion['ad_id'] == ad_id]
            min_week = weekly_data_filtered['year_week'].min()
            max_week = weekly_data_filtered['year_week'].max()
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
            monthly_data_filtered = monthly_ad_avg_completion[monthly_ad_avg_completion['ad_id'] == ad_id]
            min_month = monthly_data_filtered['month'].min()
            max_month = monthly_data_filtered['month'].max()
            all_months = pd.date_range(min_month, max_month, freq='MS').strftime('%Y-%m')
            time_points = sorted(all_months)

        if len(time_points) == 0:
            return [], html.Div("NO DATA", style={'color': 'white'})

        slider = dcc.Slider(
            id='time-slider',
            min=0,
            max=len(time_points) - 1,
            step=1,
            value=len(time_points) - 1,
            marks={i: time_points[i] for i in range(0, len(time_points), max(1, len(time_points)//5))},
            tooltip={"placement": "bottom", "always_visible": True}
        )

        return time_points, slider

    @dash_app.callback(
        Output('line-chart', 'figure'),
        [
            Input('ad-dropdown', 'value'),
            Input('time-granularity-store', 'data'),
            Input('time-slider', 'value'),
            Input('time-axis-store', 'data')
        ]
    )
    def update_line_chart(ad_id, time_granularity, slider_value, time_points):
        if not time_points or slider_value is None:
            return go.Figure()

        window_size = 14
        start_idx = max(0, slider_value - window_size + 1)
        end_idx = slider_value + 1
        display_time_points = time_points[start_idx:end_idx]

        fig = go.Figure()

        if time_granularity == 'daily':
            min_date = pd.to_datetime(display_time_points[0])
            max_date = pd.to_datetime(display_time_points[-1])
            all_dates = pd.date_range(min_date, max_date)
            filtered_data_period = data[(data['ad_id'] == ad_id) & (data['visit_date'].isin(all_dates))]
            agg_data = filtered_data_period.groupby('visit_date')['completion_rate'].mean().reindex(all_dates).fillna(0).reset_index()
            agg_data.columns = ['visit_date', 'completion_rate']
            x_data = agg_data['visit_date']
            y_data = agg_data['completion_rate'] * 100
            title_suffix = 'Daily Completion Rate Trend (In Targeted Displaying Mode)'
        elif time_granularity == 'weekly':
            weekly_data_filtered = weekly_ad_avg_completion[
                (weekly_ad_avg_completion['ad_id'] == ad_id) &
                (weekly_ad_avg_completion['year_week'].isin(display_time_points))
                ]
            all_weeks_series = pd.Series(display_time_points)
            agg_data = weekly_data_filtered.groupby('year_week')['completion_rate'].mean().reindex(all_weeks_series).fillna(0).reset_index()
            agg_data.columns = ['year_week', 'completion_rate']
            x_data = agg_data['year_week']
            y_data = agg_data['completion_rate'] * 100
            title_suffix = 'Weekly Completion Rate Trend (In Targeted Displaying Mode)'
        else:  # monthly
            monthly_data_filtered = monthly_ad_avg_completion[
                (monthly_ad_avg_completion['ad_id'] == ad_id) &
                (monthly_ad_avg_completion['month'].isin(display_time_points))
                ]
            all_months_series = pd.Series(display_time_points)
            agg_data = monthly_data_filtered.groupby('month')['completion_rate'].mean().reindex(all_months_series).fillna(0).reset_index()
            agg_data.columns = ['month', 'completion_rate']
            x_data = agg_data['month']
            y_data = agg_data['completion_rate'] * 100
            title_suffix = 'Monthly Completion Rate Trend (In Targeted Displaying Mode)'

        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            mode='lines+markers',
            line={'color': '#FFA500', 'width': 3},
            marker={'size': 10, 'color': 'orange'},
            name='Completion Rate'
        ))
        fig.update_layout(
            title={'text': f'{ad_id} {title_suffix}', 'y': 0.9, 'x': 0.5},
            xaxis={'title': 'Time', 'gridcolor': '#2a3f6f', 'color': 'white'},
            yaxis={'title': 'Completion Rate (%)', 'gridcolor': '#2a3f6f', 'color': 'white'},
            plot_bgcolor='#1f2c56',
            paper_bgcolor='#1f2c56',
            font={'color': 'white', 'size': 14},
            margin={'t': 60, 'b': 100, 'l': 80, 'r': 40},
            hovermode='x unified',
            dragmode=False
        )
        return fig

    # 返回 Dash 实例
    return dash_app
