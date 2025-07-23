import numpy as np
# import pandas as pd
import plotly.graph_objects as go 
import plotly.express as px 
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d

def generate_sample_data():
    """
    10000-20000 사이의 예측값과 0-1 사이의 초기 스코어를 생성합니다.
    (Original 예측값은 정수입니다.)
    """
    months = np.arange(1, 13)
    predictions = np.random.randint(10000, 20001, 12) 
    scores = np.random.uniform(0.1, 0.9, 12) 
    
    true_values = predictions + (scores - 0.5) * 2000 + np.random.normal(0, 1000, 12)
    true_values = np.clip(true_values, 9500, 20500)

    global_mean_prediction = np.mean(predictions)
    
    return months, predictions, scores, true_values, global_mean_prediction


# --- 알고리즘 1: 가중 평균 (Weighted Averaging) ---
# 가장 직관적이고 널리 사용되는 앙상블 기법 중 하나입니다.
# 각 개별 예측 결과에 해당 예측의 스코어를 직접적인 가중치로 부여하여 결합합니다.
# ----------------------------------------------------
def adjust_by_weighted_averaging(predictions, scores, fallback_value):
    adjusted_results = predictions * scores + fallback_value * (1 - scores)
    return adjusted_results

# --- 알고리즘 2: 스태킹 (Stacking) / 메타-러닝 (Meta-Learning) ---
# 기본 모델(Base Models)의 예측 결과를 새로운 '특징(feature)'으로 간주하고, 새로운 특징들을 입력으로 받아 최종 결과를 예측하는 기법입니다.
# ------------------------------------------------------------------------------------------------------------
def adjust_by_stacking_model(predictions, scores, true_values):
    X = np.column_stack((predictions, scores))
    y = true_values
    model = LinearRegression()
    model.fit(X, y)
    adjusted_results = model.predict(X)
    return adjusted_results

# --- 알고리즘 3: 베이지안 추론 기반 융합 (Simplified Bayesian-like Fusion) ---
# 베이지안 추론의 아이디어에서 영감을 받아, 스코어가 높을수록 예측에 대한 **확신(신뢰도)**이 강하고, 해당 예측에 더 큰 가중치를 부여해야 한다는 원칙을 적용합니다.
# ----------------------------------------------------------------------------------------------------------------------
def adjust_by_bayesian_like_fusion(predictions, scores, fallback_value):
    effective_scores = np.clip(scores + (scores**2 * 0.5), 0, 1)
    adjusted_results = predictions * effective_scores + fallback_value * (1 - effective_scores)
    return adjusted_results

# --- 알고리즘 4: 시간 기반 가중 평균 (Time-Series Weighted Averaging) ---
# 시계열 데이터에서 최근의 정보가 가장 예측력이 높거나 가장 관련성이 높다는 가정을 기반으로 합니다.
# 이 알고리즘은 각 월의 원래 스코어에 해당 월의 순서(시간적 근접성)에 비례하는 추가적인 가중치를 부여합니다.
# 최근 월(12월에 가까울수록)의 예측과 스코어에 더 큰 영향력을 행사하도록 만듭니다.
# --------------------------------------------------------------------------------------------------------------------------
def adjust_by_time_series_weighted(predictions, scores, fallback_value, months):
    normalized_month_weights = months / np.max(months)
    effective_scores = np.clip(scores * normalized_month_weights, 0, 1)
    adjusted_results = predictions * effective_scores + fallback_value * (1 - effective_scores)
    return adjusted_results


# selected_method_name 대신 line_visibility_dict를 받도록 변경
def plot_results(months, predictions, scores, adjusted_wa, adjusted_stacking, adjusted_bayesian, adjusted_ts, line_visibility_dict=None):
    """
    조정된 결과를 Plotly로 시각화합니다.
    (체크박스 선택에 따라 라인 가시성을 제어합니다.)
    """
    if line_visibility_dict is None:
        # 기본값: 모든 라인 보임
        line_visibility_dict = {
            'Original': True, 'Weighted Avg': True, 'Stacking': True,
            'Bayesian': True, 'Time-Series': True
        }

    x_new = np.linspace(months.min(), months.max(), 300)

    def smooth_curve(x_old, y_old, x_new):
        f = interp1d(x_old, y_old, kind='cubic', fill_value="extrapolate")
        return f(x_new)

    smoothed_predictions = smooth_curve(months, predictions, x_new)
    smoothed_adjusted_wa = smooth_curve(months, adjusted_wa, x_new)
    smoothed_adjusted_stacking = smooth_curve(months, adjusted_stacking, x_new)
    smoothed_adjusted_bayesian = smooth_curve(months, adjusted_bayesian, x_new)
    smoothed_adjusted_ts = smooth_curve(months, adjusted_ts, x_new)

    fig = go.Figure()

    line_colors = px.colors.qualitative.D3 
    
    # Original 라인
    fig.add_trace(go.Scatter(
        x=x_new, y=smoothed_predictions,
        mode='lines',
        name='Original',
        line=dict(color='black', dash='dot', width=2.5),
        hoverinfo='x+y',
        showlegend=True,
        visible=line_visibility_dict.get('Original', True) # 딕셔너리에서 가시성 가져옴
    ))

    # 다른 조정 라인들
    line_data = [
        (smoothed_adjusted_wa, 'Weighted Avg', line_colors[0]),
        (smoothed_adjusted_stacking, 'Stacking', line_colors[1]),
        (smoothed_adjusted_bayesian, 'Bayesian', line_colors[2]),
        (smoothed_adjusted_ts, 'Time-Series', line_colors[3]),
    ]

    for y_data, name, color in line_data:
        fig.add_trace(go.Scatter(
            x=x_new, y=y_data,
            mode='lines',
            name=name,
            line=dict(color=color, width=2.5),
            hoverinfo='x+y',
            showlegend=True,
            visible=line_visibility_dict.get(name, True) # 딕셔너리에서 가시성 가져옴
        ))

    # 스코어 막대 그래프 트레이스 (항상 보임)
    fig.add_trace(go.Bar(
        x=months, y=scores,
        name='Score',
        marker_color='rgba(255, 165, 0, 0.3)',
        marker_line_color='darkorange',
        marker_line_width=3, # 라인두께
        # marker_line_width=1,
        width=0.2, # 바 넓이
        opacity=1,
        yaxis='y2',
        hoverinfo='x+y',
        showlegend=True,
        visible=True # 항상 보임
    ))

    # 레이아웃 설정
    fig.update_layout(
        title={
            'text': '',
            # 'text': 'Prediction & Adjustment Method Comparison',
            'font': {'size': 18},
            'x': 0.5,
            'xanchor': 'center',
        },
        xaxis=dict(
            title_text='Month',
            tickvals=months,
            ticktext=[f'M{m}' for m in months],
            title_font_size=12,
            rangeslider_visible=False,
        ),
        yaxis=dict(
            title_text='Value',
            title_font_size=12,
            range=[9000, 21000],
            tickmode='array',
            tickvals=np.arange(10000, 20001, 2000),
            gridcolor='lightgray',
            griddash='dash',
        ),
        yaxis2=dict(
            title_text='Score',
            title_font_size=12,
            range=[-0.1, 1.1],
            tickmode='array',
            tickvals=np.arange(0.0, 1.1, 0.2),
            overlaying='y',
            side='right',
            showgrid=False 
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
        ),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig