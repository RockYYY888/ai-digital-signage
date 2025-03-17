import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import sqlite3
from flask import Flask  # 仅用于类型注释或参考


db_path = "advertisements.db"
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

# =============== 工具函数：重新读数据 + 预处理 ===============
def get_fresh_data():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()

    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df['completion_rate'] = df['view_time'] / df['duration']
    df['ad_id'] = 'AD-' + df['ad_id'].astype(str)

    # 自定义分类函数
    def completion_rate_level(cr):
        if 0 <= cr <= 0.2:
            return '0-20%'
        elif 0.2 < cr <= 0.4:
            return '20-40%'
        elif 0.4 < cr <= 0.6:
            return '40-60%'
        elif 0.6 < cr <= 0.8:
            return '60-80%'
        else:
            return '80-100%'

    df['completion_level'] = df['completion_rate'].apply(completion_rate_level)

    return df


color_palette = ["#FF6B6B", "#FFD930", "#6BCB77", "#4D96FF", "#9955FF"]

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
        plot_bgcolor='rgba(255, 255, 255, 0)',
        paper_bgcolor='rgba(255, 255, 255, 0)',
        xaxis={'title': 'Completion Rate Range', 'color': 'white'},
        yaxis={'title': 'Count', 'color': 'white'},
        font={'color': 'white'},
        dragmode=False
    )
    return fig


def init_dashboard(server: Flask):
    dash_app = dash.Dash(
        __name__,
        server=server,
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
            <link rel="stylesheet" href="/dashboard/assets/style.css">
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

    # 顶层先获取一次数据（也可以删掉直接依赖回调，每次都去拿）
    all_data = get_fresh_data()

    # 拿到当前最新日期、一些初始指标
    today_date = all_data['visit_date'].max()
    total_visits = all_data.shape[0]
    today_data = all_data.loc[all_data['visit_date'] == today_date].copy()
    today_visits = today_data.shape[0]
    today_avg_completion_rate = round(today_data['completion_rate'].mean() * 100, 2) if today_visits > 0 else 0

    # 下面几行辅助做图用
    # 注：为了防止第一次加载空出现错误，这里和原来一样简单处理
    def completion_rate_level(cr):
        if 0 <= cr <= 0.2:
            return '0-20%'
        elif 0.2 < cr <= 0.4:
            return '20-40%'
        elif 0.4 < cr <= 0.6:
            return '40-60%'
        elif 0.6 < cr <= 0.8:
            return '60-80%'
        else:
            return '80-100%'

    # 这里展示初始分类分布
    gender_completion_counts = all_data.groupby(['gender','completion_level']).size().unstack().fillna(0)
    age_completion_counts = all_data.groupby(['age_group','completion_level']).size().unstack().fillna(0)
    ethnicity_completion_counts = all_data.groupby(['ethnicity','completion_level']).size().unstack().fillna(0)
    overall_completion_counts = all_data['completion_level'].value_counts().reindex(
        ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], fill_value=0
    )

    # 做周、月聚合
    all_data['year'] = all_data['visit_date'].dt.isocalendar().year
    all_data['week'] = all_data['visit_date'].dt.isocalendar().week
    weekly_data = all_data.groupby(['ad_id','year','week'])['completion_rate'].mean().reset_index()
    weekly_data['year_week'] = weekly_data['year'].astype(str) + '-W' + weekly_data['week'].astype(str).str.zfill(2)

    all_data['month'] = all_data['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')
    monthly_data = all_data.groupby(['ad_id','month'])['completion_rate'].mean().reset_index()

    # 获取所有可选的 Ad 列表做下拉选项
    unique_ads = sorted(all_data['ad_id'].unique())

    # ====== Dash Layout ======
    dash_app.layout = html.Div([
        # 顶部大标题
        html.Div([
            html.H3("Advertisement Analytics Dashboard",
                    style={"margin-bottom": "0px", 'color': '#00ffcc',
                           'textAlign': 'center', 'width': '100%',
                           'font-size': '3.5rem', 'letter-spacing': '0.1rem'}),
        ], style={"margin-bottom": "15px", "display": "flex",
                  "justify-content": "center", "align-items": "center"}),

        # 中间加一个刷新按钮 + 预留icon的占位
        html.Div([
            html.Span(id='icon-space', children='[Icon Placeholder]',
                      style={'margin-right': '20px', 'color':'white',
                             'font-family':'Verdana','font-size':'1.1rem'}),
            html.Button("刷新", id="refresh-button", n_clicks=0,
                        style={
                            'backgroundColor': '#2a3f6f',
                            'color': 'white',
                            'border': '1px solid #4a6faf',
                            'borderRadius': '15px',
                            'padding': '8px 16px',
                            'cursor': 'pointer',
                            'outline': 'none',
                            'height': '40px',
                            'lineHeight': '24px',
                            'fontFamily': 'Verdana',
                            'display': 'inline-block'
                        })
        ], style={'textAlign': 'center', 'marginBottom':'25px'}),

        # 第一行：左侧信息卡 + 中间条形图Tab + 右侧饼图
        html.Div([
            # 左侧
            html.Div([
                html.Div([
                    html.H6('Total Viewers',
                            style={'textAlign': 'center', 'color': 'white',
                                   'font-family': 'Verdana',
                                   'font-size': '16px'}),
                    html.P(id='total-viewers-all',
                           children=f"{total_visits:,.0f}",
                           style={'textAlign': 'center', 'color': 'orange',
                                  'fontSize': 32})
                ], style={
                    'backgroundColor': '#1f2c56',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'marginBottom': '20px',
                }),

                html.Div([
                    html.P('Select Date:', className='fix_label',
                           style={'color': 'white', 'textAlign': 'center'}),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        date=today_date.strftime('%Y-%m-%d') if not pd.isnull(today_date) else None,
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
                    html.H6('Viewers of Selected Day',
                            style={'textAlign': 'center', 'color': 'white',
                                   'font-family': 'Verdana', 'marginTop': '20px',
                                   'font-size': '16px'}),
                    html.P(id='total-viewers-selected',
                           children=f"{today_visits:,.0f}",
                           style={'textAlign': 'center', 'color': 'orange',
                                  'fontSize': 32}),
                    html.H6('Avg Completion Rate',
                            style={'textAlign': 'center', 'color': 'white',
                                   'font-family': 'Verdana', 'marginTop': '20px',
                                   'font-size': '16px'}),
                    html.P(id='avg-completion-rate',
                           children=f"{today_avg_completion_rate}%",
                           style={'textAlign': 'center', 'color': 'orange',
                                  'fontSize': 32}),
                ], style={
                    'backgroundColor': '#1f2c56',
                    'padding': '20px',
                    'borderRadius': '10px',
                })
            ], className="three columns"),

            # 中间Tab
            html.Div([
                dcc.Tabs(id='tabs', value='gender-tab', children=[
                    dcc.Tab(label='Gender Completion', value='gender-tab', children=[
                        dcc.Graph(
                            id='gender-completion-chart',
                            figure=create_bar_chart(gender_completion_counts,
                                                    "Gender Completion Rate Distribution",
                                                    "Gender",
                                                    color_palette),
                            config={
                                'scrollZoom': False,
                                'displayModeBar': False,
                                'modeBarButtonsToRemove': [
                                    'lasso2d', 'select2d', 'pan2d', 'zoom2d',
                                    'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'
                                ]
                            }
                        )
                    ]),
                    dcc.Tab(label='Age Completion', value='age-tab', children=[
                        dcc.Graph(
                            id='age-completion-chart',
                            figure=create_bar_chart(age_completion_counts,
                                                    "Age Group Completion Rate Distribution",
                                                    "Age Group",
                                                    color_palette),
                            config={
                                'scrollZoom': False,
                                'displayModeBar': False,
                                'modeBarButtonsToRemove': [
                                    'lasso2d','select2d','pan2d','zoom2d','zoomIn2d',
                                    'zoomOut2d','autoScale2d','resetScale2d'
                                ]
                            }
                        )
                    ]),
                    dcc.Tab(label='Ethnicity Completion', value='ethnicity-tab', children=[
                        dcc.Graph(
                            id='ethnicity-completion-chart',
                            figure=create_bar_chart(ethnicity_completion_counts,
                                                    "Ethnicity Completion Rate Distribution",
                                                    "Ethnicity",
                                                    color_palette),
                            config={
                                'scrollZoom': False,
                                'displayModeBar': False,
                                'modeBarButtonsToRemove': [
                                    'lasso2d','select2d','pan2d','zoom2d','zoomIn2d',
                                    'zoomOut2d','autoScale2d','resetScale2d'
                                ]
                            }
                        )
                    ]),
                ], style={'backgroundColor': '#1f2c56', 'color': 'white',
                          'fontFamily': 'Verdana', 'border': 'none'},
                         colors={"border": "#1f2c56", "primary": "white",
                                 "background": "#1f2c56", "selected": "#192444"}),
            ], className="four columns"),

            # 右侧 饼图
            html.Div([
                dcc.Graph(id='pie-chart', figure={
                    'data': [
                        go.Pie(labels=overall_completion_counts.index,
                               values=overall_completion_counts.values,
                               hole=0.3,
                               marker={'colors': color_palette},
                               textinfo='percent',
                               textfont={'size': 16},
                               textposition='auto')
                    ],
                    'layout': go.Layout(
                        title='Overall Completion Rate Distribution',
                        plot_bgcolor='rgba(255, 255, 255, 0)',
                        paper_bgcolor='rgba(255, 255, 255, 0)',
                        font={'color': 'white', 'size': 18},
                        height=500,
                        width=500,
                        margin=dict(l=0, r=0, t=50, b=100),
                        legend=dict(orientation="h",
                                    yanchor="bottom", y=-0.3,
                                    xanchor="center", x=0.5),
                        autosize=True
                    )
                }),
            ], className="five columns"),
        ], className="row flex-display"),

        # 第二行：时间粒度 + 折线图
        html.Div([
            html.Div([
                html.P('Select Ad:', className='fix_label', style={'color': 'white'}),
                dcc.Dropdown(id='ad-dropdown',
                             options=[{'label': ad, 'value': ad} for ad in unique_ads],
                             value=unique_ads[0] if unique_ads else None,
                             style={'background-color': '#1f2c56',
                                    'color': 'white', 'optionHeight': 30}),
                html.Div([
                    html.P('Select Time Granularity:', className='fix_label',
                           style={'color': 'white', 'textAlign': 'center','marginBottom': '10px'}),
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
                            'lineHeight': '24px'
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
                            'lineHeight': '24px'
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
                            'lineHeight': '24px'
                        }),
                    ], style={'display': 'flex', 'justifyContent': 'center'}),
                ], style={'display': 'flex','flexDirection': 'column','alignItems': 'center'}),

                dcc.Store(id='time-granularity-store', data='daily'),
                dcc.Graph(
                    id='line-chart',
                    config={
                        'scrollZoom': False,
                        'displayModeBar': True,
                        'modeBarButtonsToRemove': [
                            'lasso2d','select2d','pan2d','zoom2d','zoomIn2d',
                            'zoomOut2d','autoScale2d','resetScale2d'
                        ]
                    }
                ),
                html.Div(id='slider-container', style={'marginTop': '20px'}),
                dcc.Store(id='time-axis-store')
            ], className="create_container1 twelve columns"),
        ], className="row flex-display"),

    ], id="mainContainer",
        style={"display": "flex","flex-direction": "column"})


    # ============== 回调函数部分 ==============

    @dash_app.callback(
        [
            Output('total-viewers-selected', 'children'),
            Output('avg-completion-rate', 'children'),
            Output('gender-completion-chart', 'figure'),
            Output('age-completion-chart', 'figure'),
            Output('ethnicity-completion-chart', 'figure'),
            Output('pie-chart', 'figure'),
            Output('total-viewers-all','children')  # 补充：顺便更新总观众数
        ],
        [
            Input('date-picker', 'date'),
            Input('refresh-button','n_clicks')  # 点击刷新也会触发
        ]
    )
    def update_all(selected_date, refresh_clicks):
        """
        每次选择日期 or 点击刷新按钮，就重新读取数据库并更新。
        """
        # ---- 1) 重新查询数据库 ----
        fresh = get_fresh_data()

        # ---- 2) 处理一下全局指标 ----
        total_visitors = fresh.shape[0]

        if selected_date is None:
            # 如果 date-picker 还没选，默认今天
            selected_date = fresh['visit_date'].max()
        else:
            selected_date = pd.to_datetime(selected_date)

        daily_data = fresh.loc[fresh['visit_date'] == selected_date].copy()

        daily_visits = daily_data.shape[0]
        daily_avg_rate = round(daily_data['completion_rate'].mean() * 100, 2) if daily_visits>0 else 0

        # 分布
        daily_data['completion_level'] = daily_data['completion_rate'].apply(completion_rate_level)
        gender_counts = daily_data.groupby(['gender','completion_level']).size().unstack().fillna(0)
        age_counts = daily_data.groupby(['age_group','completion_level']).size().unstack().fillna(0)
        eth_counts = daily_data.groupby(['ethnicity','completion_level']).size().unstack().fillna(0)
        overall_counts = daily_data['completion_level'].value_counts().reindex(
            ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], fill_value=0
        )

        # ---- 3) 生成对应图表 ----
        g_chart = create_bar_chart(gender_counts, "Gender Completion Rate Distribution", "Gender", color_palette)
        a_chart = create_bar_chart(age_counts, "Age Group Completion Rate Distribution", "Age Group", color_palette)
        e_chart = create_bar_chart(eth_counts, "Ethnicity Completion Rate Distribution", "Ethnicity", color_palette)

        p_chart = {
            'data': [
                go.Pie(
                    labels=overall_counts.index,
                    values=overall_counts.values,
                    hole=0.3,
                    marker={'colors': color_palette},
                    textinfo='percent',
                    textfont={'size': 16},
                    textposition='auto'
                )
            ],
            'layout': go.Layout(
                title='Overall Completion Rate Distribution',
                plot_bgcolor='rgba(255, 255, 255, 0)',
                paper_bgcolor='rgba(255, 255, 255, 0)',
                font={'color': 'white'},
                margin=dict(l=0, r=0, t=50, b=100),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                            xanchor="center", x=0.5),
                autosize=True
            )
        }

        return (
            f"{daily_visits:,.0f}",
            f"{daily_avg_rate}%",
            g_chart,
            a_chart,
            e_chart,
            p_chart,
            f"{total_visitors:,.0f}"
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
        return 'daily'

    @dash_app.callback(
        [Output('time-axis-store', 'data'), Output('slider-container', 'children')],
        [Input('ad-dropdown', 'value'), Input('time-granularity-store', 'data')]
    )
    def generate_time_axis(ad_id, time_granularity):
        # 重新拿最新数据
        fresh = get_fresh_data()

        # 做周、月聚合
        fresh['year'] = fresh['visit_date'].dt.isocalendar().year
        fresh['week'] = fresh['visit_date'].dt.isocalendar().week
        weekly_df = fresh.groupby(['ad_id','year','week'])['completion_rate'].mean().reset_index()
        weekly_df['year_week'] = weekly_df['year'].astype(str) + '-W' + weekly_df['week'].astype(str).str.zfill(2)

        fresh['month'] = fresh['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')
        monthly_df = fresh.groupby(['ad_id','month'])['completion_rate'].mean().reset_index()

        filtered_data = fresh[fresh['ad_id'] == ad_id]
        time_points = []

        if time_granularity == 'daily':
            min_date = filtered_data['visit_date'].min()
            max_date = filtered_data['visit_date'].max()
            if pd.isnull(min_date) or pd.isnull(max_date):
                return [], html.Div("NO DATA", style={'color': 'white'})
            all_dates = pd.date_range(min_date, max_date).strftime('%Y-%m-%d')
            time_points = sorted(all_dates)
        elif time_granularity == 'weekly':
            weekly_data_filtered = weekly_df[weekly_df['ad_id'] == ad_id]
            if weekly_data_filtered.empty:
                return [], html.Div("NO DATA", style={'color': 'white'})
            min_week = weekly_data_filtered['year_week'].min()
            max_week = weekly_data_filtered['year_week'].max()
            # 简单方式：把所有 unique week 都取出并排序
            time_points = sorted(weekly_data_filtered['year_week'].unique())
        elif time_granularity == 'monthly':
            monthly_data_filtered = monthly_df[monthly_df['ad_id'] == ad_id]
            if monthly_data_filtered.empty:
                return [], html.Div("NO DATA", style={'color': 'white'})
            min_month = monthly_data_filtered['month'].min()
            max_month = monthly_data_filtered['month'].max()
            # 生成从 min_month 到 max_month 的所有月
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

        # 重新拿最新数据
        fresh = get_fresh_data()

        fresh['year'] = fresh['visit_date'].dt.isocalendar().year
        fresh['week'] = fresh['visit_date'].dt.isocalendar().week
        weekly_df = fresh.groupby(['ad_id','year','week'])['completion_rate'].mean().reset_index()
        weekly_df['year_week'] = weekly_df['year'].astype(str)+'-W'+weekly_df['week'].astype(str).str.zfill(2)

        fresh['month'] = fresh['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')
        monthly_df = fresh.groupby(['ad_id','month'])['completion_rate'].mean().reset_index()

        window_size = 14
        start_idx = max(0, slider_value - window_size + 1)
        end_idx = slider_value + 1
        display_time_points = time_points[start_idx:end_idx]

        fig = go.Figure()

        if time_granularity == 'daily':
            min_date = pd.to_datetime(display_time_points[0])
            max_date = pd.to_datetime(display_time_points[-1])
            all_dates = pd.date_range(min_date, max_date)
            filtered_period = fresh[(fresh['ad_id'] == ad_id) & (fresh['visit_date'].isin(all_dates))]
            agg_data = filtered_period.groupby('visit_date')['completion_rate'].mean().reindex(all_dates, fill_value=0).reset_index()
            agg_data.columns = ['visit_date','completion_rate']
            x_data = agg_data['visit_date']
            y_data = agg_data['completion_rate'] * 100
            title_suffix = 'Daily Completion Rate Trend'
        elif time_granularity == 'weekly':
            filtered_weekly = weekly_df[(weekly_df['ad_id'] == ad_id) &
                                        (weekly_df['year_week'].isin(display_time_points))]
            # 用 pandas 的 Categorical 来确保顺序
            cats = pd.Categorical(filtered_weekly['year_week'], categories=display_time_points, ordered=True)
            aggregated = filtered_weekly.groupby(cats)['completion_rate'].mean().fillna(0).reset_index()
            x_data = aggregated['year_week']
            y_data = aggregated['completion_rate']*100
            title_suffix = 'Weekly Completion Rate Trend'
        else:
            filtered_monthly = monthly_df[(monthly_df['ad_id'] == ad_id) &
                                          (monthly_df['month'].isin(display_time_points))]
            cats = pd.Categorical(filtered_monthly['month'], categories=display_time_points, ordered=True)
            aggregated = filtered_monthly.groupby(cats)['completion_rate'].mean().fillna(0).reset_index()
            x_data = aggregated['month']
            y_data = aggregated['completion_rate']*100
            title_suffix = 'Monthly Completion Rate Trend'

        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            mode='lines+markers',
            line={'color': '#FFA500', 'width': 3},
            marker={'size': 10, 'color': 'orange'},
            name='Completion Rate'
        ))
        fig.update_layout(
            title={'text': f'{ad_id} - {title_suffix}', 'y': 0.9, 'x': 0.5},
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

    return dash_app
