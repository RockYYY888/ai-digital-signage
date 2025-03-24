import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import sqlite3
from flask import Flask

# ================ 数据加载与预处理函数 ====================

db_path = "advertisements.db"

def get_fresh_data():
    """从数据库获取最新数据并进行预处理"""
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

    # 数据预处理
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
    return data


# 颜色调色板
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
        plot_bgcolor='rgba(255, 255, 255, 0)',
        paper_bgcolor='rgba(255, 255, 255, 0)',
        xaxis={'title': 'Completion Rate Range', 'color': 'white'},
        yaxis={'title': 'Count', 'color': 'white'},
        font={'color': 'white'},
        dragmode=False
    )
    return fig

def create_no_data_figure(title, legend_title, all_groups, colors):
    fig = go.Figure()

    # 添加虚拟痕迹以显示图例
    for i, group in enumerate(all_groups):
        fig.add_trace(go.Bar(
            x=[],  # 空X轴数据
            y=[],  # 空Y轴数据
            name=group,  # 图例名称
            marker_color=colors[i % len(colors)],  # 颜色
            showlegend=True,  # 显示图例
            legendgrouptitle_text=legend_title  # 图例标题
        ))

    # 添加“NO DATA”提示
    fig.add_annotation(
        text="NO DATA",
        xref="paper", yref="paper",
        x=0.5, y=0.5,  # 居中显示
        showarrow=False,
        font=dict(size=24, color="white")
    )

    # 更新图表布局
    fig.update_layout(
        title={'text': title, 'font': {'color': 'white'}},  # 标题颜色为白色
        plot_bgcolor='rgba(255, 255, 255, 0)',  # 透明背景
        paper_bgcolor='rgba(255, 255, 255, 0)',  # 透明背景
        xaxis={'visible': False},  # 隐藏X轴
        yaxis={'visible': False},  # 隐藏Y轴
        legend=dict(
            title=legend_title,
            font=dict(color='white'),  # 图例字体颜色为白色
            bgcolor='rgba(255, 255, 255, 0)'  # 图例背景透明
        )
    )
    return fig
# =============== 初始化仪表盘 ===============

def init_dashboard(server: Flask):
    """创建并返回 Dash 应用，支持动态数据刷新"""
    dash_app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname='/dashboard/'
    )

    # 自定义 HTML 模板（保持不变）
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

    # 获取初始数据用于布局初始化
    data = get_fresh_data()
    total_visits = data.shape[0]
    today_date = data['visit_date'].max()
    today_data = data.loc[data['visit_date'] == today_date].copy()
    today_visits = today_data.shape[0]
    today_avg_completion_rate = round(today_data['completion_rate'].mean() * 100, 2) if today_visits > 0 else 0

    # 初始周度和月度数据
    data['year'] = data['visit_date'].dt.isocalendar().year
    data['week'] = data['visit_date'].dt.isocalendar().week
    weekly_ad_avg_completion = data.groupby(['ad_id', 'year', 'week'])['completion_rate'].mean().reset_index()
    weekly_ad_avg_completion['year_week'] = weekly_ad_avg_completion['year'].astype(str) + '-W' + weekly_ad_avg_completion['week'].astype(str).str.zfill(2)
    data['month'] = data['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')
    monthly_ad_avg_completion = data.groupby(['ad_id', 'month'])['completion_rate'].mean().reset_index()

    # ========== Dash Layout ==========
    dash_app.layout = html.Div([
        # Header
        html.Div([
            html.H3("Advertisement Analytics Dashboard",
                    style={"margin-bottom": "0px", 'color': '#00ffcc', 'textAlign': 'center', 'width': '100%', 'font-size': '3.5rem', 'letter-spacing': '0.1rem'}),
            html.Img(src=dash_app.get_asset_url('refresh.PNG'), id="refresh-button", n_clicks=0,
                     style={
                         'backgroundColor': '#2a3f6f',
                         'color': 'white',
                         'border': '1px solid #4a6faf',
                         'borderRadius': '15px',
                         'padding': '8px 16px',
                         'cursor': 'pointer',
                         'outline': 'none',
                         'height': '20px',
                         'lineHeight': '24px',
                         'fontFamily': 'Verdana',
                         'display': 'inline-block',
                         'margin-left': '20px'
                     })
        ], style={"margin-bottom": "15px", "display": "flex", "justify-content": "center", "align-items": "center"}),

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

            # Bar charts
            html.Div([
                dcc.Tabs(id='tabs', value='gender-tab', children=[
                    dcc.Tab(label='Gender Completion', value='gender-tab', children=[
                        dcc.Graph(id='gender-completion-chart', config={'scrollZoom': False, 'displayModeBar': False})
                    ]),
                    dcc.Tab(label='Age Completion', value='age-tab', children=[
                        dcc.Graph(id='age-completion-chart', config={'scrollZoom': False, 'displayModeBar': False})
                    ]),
                    dcc.Tab(label='Ethnicity Completion', value='ethnicity-tab', children=[
                        dcc.Graph(id='ethnicity-completion-chart', config={'scrollZoom': False, 'displayModeBar': False})
                    ]),
                ], style={'backgroundColor': '#1f2c56', 'color': 'white', 'fontFamily': 'Verdana', 'border': 'none'},
                   colors={"border": "#1f2c56", "primary": "white", "background": "#1f2c56", "selected": "#192444"}),
            ], className="create_container four columns"),

            # Pie chart
            html.Div([
                dcc.Graph(id='pie-chart'),
            ], className="create_container five columns"),
        ], className="row flex-display"),

        # Line chart with custom buttons
        html.Div([
            html.Div([
                html.P('Select Ad:', className='fix_label', style={'color': 'white'}),
                dcc.Dropdown(id='ad-dropdown',  # 初始为空，动态更新
                             style={'background-color': '#1f2c56', 'color': 'white', 'optionHeight': 30}),
                html.Div([
                    html.P('Select Time Granularity:', className='fix_label',
                           style={'color': 'white', 'textAlign': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Button('Daily', id='daily-button', n_clicks=0, style={
                            'backgroundColor': '#2a3f6f', 'color': 'white', 'border': '1px solid #4a6faf',
                            'borderRadius': '15px', 'padding': '8px 16px', 'marginRight': '10px', 'fontFamily': 'Verdana',
                            'cursor': 'pointer', 'outline': 'none', 'height': '40px', 'lineHeight': '24px',
                            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
                        }),
                        html.Button('Weekly', id='weekly-button', n_clicks=0, style={
                            'backgroundColor': '#2a3f6f', 'color': 'white', 'border': '1px solid #4a6faf',
                            'borderRadius': '15px', 'padding': '8px 16px', 'marginRight': '10px', 'fontFamily': 'Verdana',
                            'cursor': 'pointer', 'outline': 'none', 'height': '40px', 'lineHeight': '24px',
                            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
                        }),
                        html.Button('Monthly', id='monthly-button', n_clicks=0, style={
                            'backgroundColor': '#2a3f6f', 'color': 'white', 'border': '1px solid #4a6faf',
                            'borderRadius': '15px', 'padding': '8px 16px', 'fontFamily': 'Verdana',
                            'cursor': 'pointer', 'outline': 'none', 'height': '40px', 'lineHeight': '24px',
                            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
                        }),
                    ], style={'display': 'flex', 'justifyContent': 'center'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                dcc.Store(id='time-granularity-store', data='daily'),
                dcc.Graph(id='line-chart', config={'scrollZoom': False, 'displayModeBar': True,
                                                  'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']}),
                html.Div(id='slider-container', style={'marginTop': '20px'}),
                dcc.Store(id='time-axis-store'),
                dcc.Store(id='granularity-for-chart', data='daily'),
                html.P(id='granularity-display', style={'color': 'white', 'textAlign': 'center'})
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
            Output('pie-chart', 'figure'),
            Output('total-viewers-all', 'children'),
            Output('ad-dropdown', 'options'),  # 新增输出
            Output('ad-dropdown', 'value')     # 新增输出
        ],
        [
            Input('date-picker', 'date'),
            Input('refresh-button', 'n_clicks')
        ]
    )
    def update_all(selected_date, refresh_clicks):
        """更新所有图表和指标，支持刷新，并动态更新 ad-dropdown"""
        fresh = get_fresh_data()
        total_visits = fresh.shape[0]

        selected_date = pd.to_datetime(selected_date)
        daily_data = fresh.loc[fresh['visit_date'] == selected_date].copy()

        daily_visits = daily_data.shape[0]
        daily_avg_rate = round(daily_data['completion_rate'].mean() * 100, 2) if daily_visits > 0 else 0

        daily_data['completion_level'] = daily_data['completion_rate'].apply(
            lambda cr: '0-20%' if cr <= 0.2 else '20-40%' if cr <= 0.4 else '40-60%' if cr <= 0.6 else '60-80%' if cr <= 0.8 else '80-100%'
        )
        gender_counts = daily_data.groupby(['gender', 'completion_level']).size().unstack().fillna(0)
        age_counts = daily_data.groupby(['age_group', 'completion_level']).size().unstack().fillna(0)
        eth_counts = daily_data.groupby(['ethnicity', 'completion_level']).size().unstack().fillna(0)
        overall_counts = daily_data['completion_level'].value_counts().reindex(
            ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], fill_value=0
        )

        gender_chart = create_bar_chart(gender_counts, "Gender Completion Rate Distribution", "Gender", color_palette)
        age_chart = create_bar_chart(age_counts, "Age Group Completion Rate Distribution", "Age Group", color_palette)
        eth_chart = create_bar_chart(eth_counts, "Ethnicity Completion Rate Distribution", "Ethnicity", color_palette)

        # 修改饼状图逻辑：只显示非零值的完成率区间
        nonzero_counts = overall_counts[overall_counts > 0]  # 过滤掉值为0的区间
        if nonzero_counts.empty:
            # 如果所有值都为0，显示“No Data”
            pie_chart = {
                'data': [],
                'layout': go.Layout(
                    title='Overall Completion Rate Distribution',
                    annotations=[dict(text='NO DATA', x=0.5, y=0.5, font_size=20, showarrow=False, font_color='white')],
                    plot_bgcolor='rgba(255, 255, 255, 0)',
                    paper_bgcolor='rgba(255, 255, 255, 0)',
                    font={'color': 'white'},
                    margin=dict(l=0, r=0, t=50, b=100),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    autosize=True
                )
            }
        else:
            # 只显示非零值的数据
            pie_chart = {
                'data': [go.Pie(
                    labels=nonzero_counts.index,
                    values=nonzero_counts.values,
                    hole=0.3,
                    marker={'colors': color_palette[:len(nonzero_counts)]},  # 动态调整颜色长度
                    textinfo='percent',
                    textfont={'size': 16},
                    textposition='auto'
                )],
                'layout': go.Layout(
                    title='Overall Completion Rate Distribution',
                    plot_bgcolor='rgba(255, 255, 255, 0)',
                    paper_bgcolor='rgba(255, 255, 255, 0)',
                    font={'color': 'white'},
                    margin=dict(l=0, r=0, t=50, b=100),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    autosize=True
                )
            }

        # 动态更新 ad-dropdown 选项，按数字顺序排序
        ad_options = [{'label': ad_id, 'value': ad_id} for ad_id in sorted(fresh['ad_id'].unique(), key=lambda x: int(x.split('-')[1]))]
        ad_value = sorted(fresh['ad_id'].unique(), key=lambda x: int(x.split('-')[1]))[0]  # 默认选择第一个

        return (
            f"{daily_visits:,.0f}",
            f"{daily_avg_rate}%",
            gender_chart,
            age_chart,
            eth_chart,
            pie_chart,
            f"{total_visits:,.0f}",
            ad_options,
            ad_value
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
        [Output('time-axis-store', 'data'), Output('slider-container', 'children'), Output('granularity-for-chart', 'data')],
        [Input('ad-dropdown', 'value'), Input('time-granularity-store', 'data')]
    )
    def generate_time_axis(ad_id, time_granularity):
        """生成时间轴和滑块，支持刷新"""
        fresh = get_fresh_data()
        filtered_data = fresh[fresh['ad_id'] == ad_id]
        time_points = []

        if time_granularity == 'daily':
            min_date = filtered_data['visit_date'].min()
            max_date = filtered_data['visit_date'].max()
            if pd.isnull(min_date) or pd.isnull(max_date):
                return [], html.Div("NO DATA", style={'color': 'white'}), time_granularity
            all_dates = pd.date_range(min_date, max_date).strftime('%Y-%m-%d')
            time_points = sorted(all_dates)
        elif time_granularity == 'weekly':
            fresh['year'] = fresh['visit_date'].dt.isocalendar().year
            fresh['week'] = fresh['visit_date'].dt.isocalendar().week
            weekly_data = fresh.groupby(['ad_id', 'year', 'week'])['completion_rate'].mean().reset_index()
            weekly_data['year_week'] = weekly_data['year'].astype(str) + '-W' + weekly_data['week'].astype(str).str.zfill(2)
            weekly_data_filtered = weekly_data[weekly_data['ad_id'] == ad_id]
            if weekly_data_filtered.empty:
                return [], html.Div("NO DATA", style={'color': 'white'}), time_granularity
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
            fresh['month'] = fresh['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')
            monthly_data = fresh.groupby(['ad_id', 'month'])['completion_rate'].mean().reset_index()
            monthly_data_filtered = monthly_data[monthly_data['ad_id'] == ad_id]
            if monthly_data_filtered.empty:
                return [], html.Div("NO DATA", style={'color': 'white'}), time_granularity
            min_month = monthly_data_filtered['month'].min()
            max_month = monthly_data_filtered['month'].max()
            all_months = pd.date_range(min_month, max_month, freq='MS').strftime('%Y-%m')
            time_points = sorted(all_months)

        if len(time_points) == 0:
            return [], html.Div("NO DATA", style={'color': 'white'}), time_granularity

        slider = dcc.Slider(
            id='time-slider',
            min=0,
            max=len(time_points) - 1,
            step=1,
            value=len(time_points) - 1,
            marks={i: time_points[i] for i in range(0, len(time_points), max(1, len(time_points) // 5))},
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
        """更新折线图，支持刷新并保持旧版逻辑"""
        if not time_points or slider_value is None:
            return go.Figure()

        fresh = get_fresh_data()
        window_size = 14
        start_idx = max(0, slider_value - window_size + 1)
        end_idx = slider_value + 1
        display_time_points = time_points[start_idx:end_idx]

        fig = go.Figure()

        if time_granularity == 'daily':
            min_date = pd.to_datetime(display_time_points[0])
            max_date = pd.to_datetime(display_time_points[-1])
            all_dates = pd.date_range(min_date, max_date)
            filtered_data_period = fresh[(fresh['ad_id'] == ad_id) & (fresh['visit_date'].isin(all_dates))]
            agg_data = filtered_data_period.groupby('visit_date')['completion_rate'].mean().reindex(all_dates).fillna(0).reset_index()
            agg_data.columns = ['visit_date', 'completion_rate']
            x_data = agg_data['visit_date']
            y_data = agg_data['completion_rate'] * 100
            title_suffix = 'Daily Completion Rate Trend (In Targeted Displaying Mode)'
        elif time_granularity == 'weekly':
            fresh['year'] = fresh['visit_date'].dt.isocalendar().year
            fresh['week'] = fresh['visit_date'].dt.isocalendar().week
            weekly_data = fresh.groupby(['ad_id', 'year', 'week'])['completion_rate'].mean().reset_index()
            weekly_data['year_week'] = weekly_data['year'].astype(str) + '-W' + weekly_data['week'].astype(str).str.zfill(2)
            weekly_data_filtered = weekly_data[(weekly_data['ad_id'] == ad_id) &
                                              (weekly_data['year_week'].isin(display_time_points))]
            all_weeks_series = pd.Series(display_time_points)
            agg_data = weekly_data_filtered.groupby('year_week')['completion_rate'].mean().reindex(all_weeks_series).fillna(0).reset_index()
            agg_data.columns = ['year_week', 'completion_rate']
            x_data = agg_data['year_week']
            y_data = agg_data['completion_rate'] * 100
            title_suffix = 'Weekly Completion Rate Trend (In Targeted Displaying Mode)'
        else:  # monthly
            fresh['month'] = fresh['visit_date'].dt.to_period('M').dt.strftime('%Y-%m')
            monthly_data = fresh.groupby(['ad_id', 'month'])['completion_rate'].mean().reset_index()
            monthly_data_filtered = monthly_data[(monthly_data['ad_id'] == ad_id) &
                                                (monthly_data['month'].isin(display_time_points))]
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

    @dash_app.callback(
        Output('granularity-display', 'children'),
        Input('time-granularity-store', 'data')
    )
    def display_granularity(granularity):
        """显示当前时间粒度"""
        return f"Current granularity: {granularity}"

    return dash_app  # 正确返回 Dash 实例

