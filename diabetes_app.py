import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 페이지 설정
st.set_page_config(
    page_title="당뇨병 예측 AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 스타일
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1.2rem;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .normal {
        background-color: #d4edda;
        color: #155724;
        border-left: 5px solid #28a745;
    }
    .diabetes {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 5px solid #dc3545;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# 사이드바 정보
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("📋 정보")
    st.markdown("""
    이 앱은 머신러닝 모델을 기반으로 **당뇨병 발병 가능성**을 예측합니다.
    
    **8개의 임상 정보**를 입력하면:
    - 당뇨 여부 (0: 정상, 1: 당뇨)
    - 당뇨 확률 (%)
    
    를 제공합니다.
    
    ---
    **🩺 입력 항목 설명**
    - 임신횟수
    - 혈당 (Glucose)
    - 혈압 (Blood Pressure)
    - 피부두께 (Skin Thickness)
    - 인슐린 (Insulin)
    - BMI (체질량지수)
    - 당뇨 가족력 (Diabetes Pedigree Function)
    - 나이
    """)
    st.markdown("---")
    st.markdown("© 2025 당뇨 예측 AI")

# 메인 화면
st.title("🩺 당뇨병 예측 시스템")
st.markdown("아래 정보를 입력하고 **예측하기** 버튼을 눌러주세요.")

# 2열 레이아웃으로 입력 필드 배치
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.subheader("📊 기본 정보")
    pregnancies = st.number_input("임신 횟수", min_value=0, max_value=20, value=1, step=1, help="과거 임신 횟수")
    glucose = st.number_input("혈당 (Glucose)", min_value=0.0, max_value=300.0, value=120.0, step=1.0, help="포도당 수치 (mg/dL)")
    blood_pressure = st.number_input("혈압 (Blood Pressure)", min_value=0.0, max_value=200.0, value=70.0, step=1.0, help="이완기 혈압 (mm Hg)")
    skin_thickness = st.number_input("피부 두께 (Skin Thickness)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, help="삼두근 피부 주름 두께 (mm)")

with col2:
    st.subheader("🧬 대사 정보")
    insulin = st.number_input("인슐린 (Insulin)", min_value=0.0, max_value=900.0, value=80.0, step=1.0, help="2시간 혈청 인슐린 (mu U/ml)")
    bmi = st.number_input("BMI (체질량지수)", min_value=0.0, max_value=70.0, value=32.0, step=0.5, help="체중(kg) / 키(m)^2")
    dpf = st.number_input("당뇨 가족력 (DPF)", min_value=0.0, max_value=2.5, value=0.5, step=0.01, help="당뇨병 가족력 함수")
    age = st.number_input("나이", min_value=0, max_value=120, value=30, step=1, help="만 나이")

# 파생 변수 계산
blood_glucose_index = glucose + blood_pressure
pancreas_index = glucose + insulin
metabolic_risk_index = glucose + bmi
elderly = 1 if age >= 50 else 0

# 입력 데이터를 DataFrame으로 구성 (모델 학습 시 사용된 컬럼명과 동일해야 함)
input_data = pd.DataFrame(
    [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age,
      blood_glucose_index, pancreas_index, metabolic_risk_index, elderly]],
    columns=['임신횟수', '혈당', '혈압', '피부두께', '인슐린', 'BMI', '당뇨가족력', '나이',
             '혈당혈압지수', '췌장기능지수', '대사위험지수', '고령여부']
)

# 모델 및 스케일러 로드 함수 (파일명 수정)
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('diabetes_model.pkl')   # 수정됨
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ 파일을 찾을 수 없습니다: {e.filename}")
        st.info("diabetes_model.pkl과 scaler.pkl이 같은 디렉토리에 있는지 확인하세요.")
        return None, None

log_model_eng, scaler = load_model_and_scaler()

# 예측 버튼
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    predict_button = st.button("🔍 당뇨 예측하기", use_container_width=True)

# 예측 실행
if predict_button:
    if log_model_eng is not None and scaler is not None:
        # 스케일링
        try:
            input_data_scaled = scaler.transform(input_data)
        except Exception as e:
            st.error(f"스케일링 오류: {e}")
            st.stop()
        
        # 예측
        predicted = log_model_eng.predict(input_data_scaled)
        prob = log_model_eng.predict_proba(input_data_scaled)[0]
        diabetes_prob = prob[1] * 100
        
        # 결과 출력
        st.markdown("---")
        st.subheader("📈 예측 결과")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if predicted[0] == 0:
                st.markdown(f"""
                <div class="prediction-box normal">
                    <h2>✅ 정상</h2>
                    <p>당뇨병 발병 위험이 낮은 것으로 예측됩니다.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box diabetes">
                    <h2>⚠️ 당뇨 의심</h2>
                    <p>당뇨병 발병 위험이 높은 것으로 예측됩니다.<br>전문의와 상담하세요.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_res2:
            st.metric(label="당뇨 확률", value=f"{diabetes_prob:.1f}%", delta=None)
            st.progress(int(diabetes_prob))
            st.caption("확률이 50% 이상이면 당뇨로 분류됩니다.")
        
        # 추가 정보: 입력값 요약
        with st.expander("📋 입력 정보 요약"):
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.write(f"- 임신횟수: {pregnancies}")
                st.write(f"- 혈당: {glucose} mg/dL")
                st.write(f"- 혈압: {blood_pressure} mm Hg")
                st.write(f"- 피부두께: {skin_thickness} mm")
            with col_sum2:
                st.write(f"- 인슐린: {insulin} mu U/ml")
                st.write(f"- BMI: {bmi}")
                st.write(f"- 가족력: {dpf}")
                st.write(f"- 나이: {age}세")
            st.write("**파생 변수**")
            st.write(f"- 혈당혈압지수: {blood_glucose_index:.1f}")
            st.write(f"- 췌장기능지수: {pancreas_index:.1f}")
            st.write(f"- 대사위험지수: {metabolic_risk_index:.1f}")
            st.write(f"- 고령여부: {'예 (50세 이상)' if elderly else '아니오'}")
    else:
        st.warning("⚠️ 모델 또는 스케일러를 불러오지 못했습니다. 파일을 확인해주세요.")

# 푸터
st.markdown("""
<div class="footer">
    <p>⚠️ 본 예측은 참고용이며, 의학적 진단을 대체할 수 없습니다. 건강 문제가 있을 시 반드시 의사와 상담하세요.</p>
</div>
""", unsafe_allow_html=True)

# 실행 가이드 (최초 실행 시)
if 'show_help' not in st.session_state:
    st.session_state.show_help = True
    with st.expander("🚀 실행 가이드"):
        st.markdown("""
        1. 위 입력 필드에 환자의 정보를 입력하세요.
        2. **당뇨 예측하기** 버튼을 클릭하세요.
        3. 예측 결과(정상/당뇨)와 당뇨 확률이 표시됩니다.
        
        **필요 파일**: `diabetes_model.pkl`, `scaler.pkl` (동일 디렉토리에 있어야 함)
        """)
