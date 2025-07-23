import streamlit as st
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from utils import (
    generate_sample_data,
    adjust_by_weighted_averaging,
    adjust_by_stacking_model,
    adjust_by_bayesian_like_fusion,
    adjust_by_time_series_weighted,
    plot_results
)
from streamlit_vertical_slider import vertical_slider 

# --- Streamlit 페이지 설정 ---
st.set_page_config(layout="wide", page_title="Simulator")

st.title("시뮬레이션")
# st.title("월별 모델 예측 조정 시뮬레이터")
st.write("모델 예측 결과와 스코어 조합 결과를 시뮬레이션합니다.")

# --- Helper function for scaling predictions to scores ---
def scale_predictions_to_scores(predictions_array):
    min_pred = predictions_array.min()
    max_pred = predictions_array.max()

    if max_pred == min_pred:
        return np.full_like(predictions_array, 0.5, dtype=float)
    else:
        scaled_scores = (predictions_array - min_pred) / (max_pred - min_pred)
        scaled_scores = scaled_scores * 0.8 + 0.1 
        return np.round(np.clip(scaled_scores, 0.0, 1.0), 1)

# --- 세션 상태 초기화 ---
# 모든 그래프 라인의 이름 정의
ALL_GRAPH_METHODS = ['Original', 'Weighted Avg', 'Stacking', 'Bayesian', 'Time-Series']

if 'predictions' not in st.session_state:
    months, predictions, _, true_values, global_mean_prediction = generate_sample_data()
    st.session_state.months = months
    st.session_state.predictions = predictions
    st.session_state.scores = scale_predictions_to_scores(predictions)
    st.session_state.true_values = true_values
    st.session_state.global_mean_prediction = global_mean_prediction
    st.session_state.slider_reset_counter = 0 
    # --- 변경: 체크박스 상태를 저장할 딕셔너리 초기화 (모두 True) ---
    st.session_state.line_visibility = {method: True for method in ALL_GRAPH_METHODS}

# --- 샘플 데이터 생성 버튼 콜백 ---
def generate_new_data_callback():
    months, predictions, _, true_values, global_mean_data = generate_sample_data()
    st.session_state.months = months
    st.session_state.predictions = predictions
    st.session_state.scores = scale_predictions_to_scores(predictions)
    st.session_state.true_values = true_values
    st.session_state.global_mean_prediction = global_mean_data
    st.session_state.slider_reset_counter += 1
    # --- 변경: 새로운 데이터 생성 시 모든 체크박스 True로 초기화 ---
    st.session_state.line_visibility = {method: True for method in ALL_GRAPH_METHODS}


# # --- CSS 주입 (버튼 간격, 슬라이더 배경 및 폭) ---
# st.markdown("""
#     <style>
#     /* 1. 버튼 간격 조절 */
#     div[data-testid="stVerticalBlock"] > div:nth-of-type(1) > div[data-testid="stHorizontalBlock"] {
#         gap: 5px; /* 버튼 사이의 간격 */
#     }

#     /* 2. 슬라이더 영역 배경색 및 라벨 색상 */
#     div[data-testid="stVerticalBlock"] > div:nth-of-type(2) { 
#         background-color: #333; /* 어두운 배경색 */
#         padding: 15px 10px; /* 내부 여백 */
#         border-radius: 8px; /* 둥근 모서리 */
#         box-shadow: 0 4px 8px rgba(0,0,0,0.2);
#         margin-top: 20px; /* 위에 버튼과의 간격 */
#     }
    
#     /* 슬라이더 월 라벨 (M1, M2...) 색상을 밝게 */
#     div[data-testid="stVerticalBlock"] div[data-testid="stVerticalSlider"] label > div > span {
#         color: #ccc !important;
#         font-weight: bold;
#         font-size: 0.8em;
#     }

#     /* 3. 개별 수직 슬라이더 컨테이너 폭 고정 (12개 슬라이더 폭 일관성) */
#     div[data-testid="stVerticalSlider"] {
#         width: 60px !important; /* 각 슬라이더의 컨테이너 폭을 고정 */
#         display: flex;
#         flex-direction: column;
#         align-items: center;
#         padding-bottom: 20px; /* 하단 라벨 여백 확보 */
#     }

#     /* 4. vertical_slider 내부의 실제 range input 요소 스타일 */
#     div[data-testid="stVerticalSlider"] input[type="range"] {
#         height: 150px;
#         width: 15px;
#         -webkit-appearance: none;
#         appearance: none;
#         background: transparent;
#         outline: none;
#         cursor: pointer;
#     }
    
#     /* 5. vertical_slider 값 라벨 폰트 및 색상 */
#     .stVerticalSlider .st-emotion-cache-18rd50b { /* 이 클래스는 값을 표시하는 div (Streamlit 버전마다 다를 수 있음) */
#         font-size: 0.9em !important;
#         color: white !important; /* 값 라벨 색상 */
#         margin-top: -15px; /* 값 라벨 위치 조정 */
#     }
    
#     /* 6. 체크박스 라벨 스타일 */
#     /* Streamlit 라디오 버튼/체크박스 라벨의 기본 색상이 어두울 수 있어 밝게 조정 */
#     div[data-testid="stVerticalBlock"] > div:nth-of-type(3) { /* 그래프 위에 라디오 버튼/체크박스가 있을 블록 */
#         /* background-color: transparent; */ /* 배경은 제거 */
#         padding: 10px 0px; /* 좌우 패딩을 줄여서 그래프와 라벨 사이의 간격을 좁힘 */
#         /* color: #333; */ /* 텍스트 색상은 Streamlit 기본으로 */
#     }
#     div[data-testid="stVerticalBlock"] > div:nth-of-type(3) label.st-bh.st-cq { /* 체크박스 라벨 */
#         color: #333 !important; /* 라벨 색상 */
#         font-weight: bold;
#         font-size: 0.9em;
#     }
#     div[data-testid="stVerticalBlock"] > div:nth-of-type(3) span.css-10n2b5k.eqr8v9z2 { /* 체크박스 옵션 텍스트 */
#         color: #555 !important; /* 옵션 텍스트 색상 */
#     }
    
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# --- 버튼 배치 (1:1:5 구조) ---
col_btn_gen, col_btn_confirm, col_btn_spacer = st.columns([1, 1, 5]) 

with col_btn_gen:
    st.button("샘플 데이터 생성", on_click=generate_new_data_callback, key="generate_data_button")
    # st.button("새로운 샘플 데이터 생성", on_click=generate_new_data_callback, key="generate_data_button")

with col_btn_confirm:
    pass 


# --- 2. 월별 스코어 조정 (이제 메인 컬럼) ---
st.subheader("트랜드 조정")
# st.subheader("월별 스코어 조정 (0.0 ~ 1.0) - 이퀄라이저 스타일:")

slider_cols = st.columns(12) 

for i in range(12):
    with slider_cols[i]:
        st.session_state.scores[i] = vertical_slider(
            label=f"M{i+1}",
            key=f"vert_slider_{i}_{st.session_state.slider_reset_counter}",
            min_value=0.0,
            max_value=1.0,
            default_value=float(st.session_state.scores[i]),
            step=0.1, 
            height=150,
            value_always_visible=True,
            thumb_shape="circle",
            track_color="darkgray",
            slider_color=("red", "darkgray"),
            thumb_color="silver"
        )
        st.session_state.scores[i] = round(st.session_state.scores[i], 1) 


# --- 3. 조정된 예측 결과 비교 (그래프) ---
st.subheader("조정 결과 비교")
# st.subheader("3. 조정된 예측 결과 비교:")

# --- 변경된 부분: 체크박스로 그래프 표시 항목 선택 ---
# 체크박스들을 수평으로 배치하기 위해 columns 사용
cols_visibility_checkbox = st.columns(len(ALL_GRAPH_METHODS))

for i, method in enumerate(ALL_GRAPH_METHODS):
    with cols_visibility_checkbox[i]:
        st.session_state.line_visibility[method] = st.checkbox(
            method,
            value=st.session_state.line_visibility[method], # 현재 세션 상태의 값으로 초기화
            key=f"checkbox_{method}_{st.session_state.slider_reset_counter}" # 슬라이더와 동일하게 리셋 카운터 사용
        )
# -------------------------------------------------------------

graph_height_inches = 6 * 0.6 

current_months = st.session_state.months
current_predictions = st.session_state.predictions
current_scores = st.session_state.scores
current_true_values = st.session_state.true_values
current_global_mean_prediction = st.session_state.global_mean_prediction

adj_wa = adjust_by_weighted_averaging(current_predictions, current_scores, current_global_mean_prediction)
adj_stacking = adjust_by_stacking_model(current_predictions, current_scores, current_true_values)
adj_bayesian = adjust_by_bayesian_like_fusion(current_predictions, current_scores, current_global_mean_prediction)
adj_ts = adjust_by_time_series_weighted(current_predictions, current_scores, current_global_mean_prediction, current_months)

# plot_results 함수에 line_visibility_dict 전달
fig = plot_results(current_months, current_predictions, current_scores, 
                   adj_wa, adj_stacking, adj_bayesian, adj_ts, 
                   line_visibility_dict=st.session_state.line_visibility) # <-- 딕셔너리 전달

fig.update_layout(height=graph_height_inches * 96) # Plotly height는 픽셀 단위 (인치 * DPI)

st.plotly_chart(fig, use_container_width=True)