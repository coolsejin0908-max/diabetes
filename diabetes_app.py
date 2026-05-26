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

# 커스텀 CSS (모던 그라데이션, 카드 스타일, 애니메이션)
st.markdown("""
    <style>
    /* 전체 배경 */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9edf2 100%);
    }
    
    /* 카드 스타일 */
    .css-1r6slb0, .css-1v3fvcr {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    /* 버튼 */
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        font-weight: bold;
        border-radius: 40px;
        padding: 0.7rem 2rem;
        font-size: 1.3rem;
        border: none;
        box-shadow: 0 4px 15px rgba(76,175,80,0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76,175,80,0.4);
        background: linear-gradient(90deg, #45a049, #3d8b40);
    }
    
    /* 슬라이더 스타일 */
    .stSlider > div > div > div {
        background-color: #4CAF50;
    }
    
    /* 예측 결과 박스 */
    .prediction-box {
        border-radius: 24px;
        padding: 28px;
        background: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        backdrop-filter: blur(2px);
        transition: all 0.3s;
    }
    .normal {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        border-left: 6px solid #28a745;
    }
    .diabetes {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        border-left: 6px solid #dc3545;
    }
    
    /* 제목 스타일 */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .stApp header {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
    }
    
    /* 사이드바 */
    .css-1d391kg {
        background: rgba(255,255,255,0.95);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    /* footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        font-size: 0.85rem;
        color: #6c757d;
        border-top: 1px solid rgba(0,0,0,0.05);
    }
    
    /* 인포 박스 */
    .info-box {
        background: #e7f3ff;
        border-radius: 16px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* 프로그레스 바 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #ffc107, #dc3545);
        border-radius: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=90)
    st.title("📘 도움말")
    st.markdown("""
    **당뇨 예측 모델 (로지스틱 회귀)**  
    Pima 인디언 당뇨 데이터셋 기반
    
    - **정확도**: 78%  
    - **특징**: 8개 임상 정보 + 4개 파생 변수
    
    ---
    ### 🎚️ 사용 방법
    1. 슬라이더로 수치 조정
    2. **예측하기** 클릭
    3. 결과 및 확률 확인
    
    ---
    ### 📊 입력 항목 기준
    - 정상 혈당 공복 기준: < 100 mg/dL  
    - 정상 혈압: < 120/80 mmHg  
    - 정상 BMI: 18.5 ~ 22.9
    """)
    st.markdown("---")
    st.caption("© 2025 당뇨 예측 AI · 참고용")

# 메인 타이틀
st.title("🩺 AI 기반 당뇨병 예측 시스템")
st.markdown("**슬라이더를 움직여 건강 정보를 입력하세요** → 아래 버튼으로 즉시 예측")

# 2열 레이아웃
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📊 기본 건강 지표")
    pregnancies = st.slider("👶 임신 횟수", 0, 17, 1, 1, help="과거 임신 횟수 (0~17)")
    glucose = st.slider("🩸 혈당 (Glucose mg/dL)", 0.0, 200.0, 120.0, 1.0, help="공복 혈당 수치")
    blood_pressure = st.slider("❤️ 혈압 (Diastolic mmHg)", 0.0, 130.0, 70.0, 1.0, help="이완기 혈압")
    skin_thickness = st.slider("📏 피부 두께 (Skin Thickness mm)", 0.0, 60.0, 20.0, 0.5, help="삼두근 피부 주름 두께")

with col2:
    st.subheader("🧬 대사 & 생활 정보")
    insulin = st.slider("💉 인슐린 (Insulin mu U/ml)", 0.0, 900.0, 80.0, 5.0, help="2시간 혈청 인슐린")
    bmi = st.slider("⚖️ BMI (체질량지수)", 10.0, 60.0, 32.0, 0.5, help="체중(kg)/키(m)^2")
    dpf = st.slider("👨‍👩‍👧‍👦 당뇨 가족력 (DPF)", 0.0, 2.5, 0.5, 0.01, help="가족력 함수 (높을수록 위험)")
    age = st.slider("🎂 나이", 0, 120, 30, 1, help="만 나이")

# 파생 변수 (자동 계산)
blood_glucose_index = glucose + blood_pressure
pancreas_index = glucose + insulin
metabolic_risk_index = glucose + bmi
elderly = 1 if age >= 50 else 0

# 입력 데이터 프레임 (컬럼명 일치 필수)
input_data = pd.DataFrame(
    [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age,
      blood_glucose_index, pancreas_index, metabolic_risk_index, elderly]],
    columns=['임신횟수', '혈당', '혈압', '피부두께', '인슐린', 'BMI', '당뇨가족력', '나이',
             '혈당혈압지수', '췌장기능지수', '대사위험지수', '고령여부']
)

# 모델 로드 (캐싱)
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ 파일 없음: {e.filename}")
        st.info("`diabetes_model.pkl`과 `scaler.pkl`을 앱과 같은 폴더에 넣어주세요.")
        return None, None

log_model_eng, scaler = load_model_and_scaler()

# 예측 버튼 (중앙 정렬)
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🔍 지금 당뇨 예측하기", use_container_width=True)

# 예측 결과 출력
if predict_btn:
    if log_model_eng is not None and scaler is not None:
        try:
            input_scaled = scaler.transform(input_data)
            pred = log_model_eng.predict(input_scaled)
            prob = log_model_eng.predict_proba(input_scaled)[0]
            diabetes_prob = prob[1] * 100
        except Exception as e:
            st.error(f"예측 오류: {e}")
            st.stop()
        
        st.markdown("---")
        st.subheader("📈 예측 결과 분석")
        
        # 결과 카드 2열
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if pred[0] == 0:
                st.markdown("""
                <div class="prediction-box normal">
                    <h2>✅ 정상</h2>
                    <p>당뇨병 위험이 낮은 것으로 예측됩니다.<br>현재 생활 습관을 유지하세요.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box diabetes">
                    <h2>⚠️ 당뇨 의심</h2>
                    <p>당뇨병 발병 위험이 높습니다.<br>병원 내원 및 추가 검사를 권장합니다.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_res2:
            st.metric("📊 당뇨 확률", f"{diabetes_prob:.1f}%", delta=None)
            st.progress(int(diabetes_prob))
            st.caption("`50% 이상`이면 당뇨로 분류됩니다.")
            # 추가 정보: 위험 수준 텍스트
            if diabetes_prob < 30:
                st.success("🎉 낮은 위험")
            elif diabetes_prob < 70:
                st.warning("⚠️ 중간 위험")
            else:
                st.error("🔥 고위험")
        
        # 상세 입력값 요약 (아코디언)
        with st.expander("📋 입력값 및 파생 변수 상세 보기"):
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.markdown("**원본 값**")
                st.write(f"- 임신횟수: {pregnancies}")
                st.write(f"- 혈당: {glucose} mg/dL")
                st.write(f"- 혈압: {blood_pressure} mmHg")
                st.write(f"- 피부두께: {skin_thickness} mm")
                st.write(f"- 인슐린: {insulin} mu U/ml")
                st.write(f"- BMI: {bmi}")
                st.write(f"- 가족력: {dpf}")
                st.write(f"- 나이: {age}세")
            with col_sum2:
                st.markdown("**파생 변수 (모델 입력)**")
                st.write(f"- 혈당+혈압 지수: {blood_glucose_index:.1f}")
                st.write(f"- 췌장기능지수: {pancreas_index:.1f}")
                st.write(f"- 대사위험지수: {metabolic_risk_index:.1f}")
                st.write(f"- 고령여부: {'✅ 50세 이상' if elderly else '❌ 50세 미만'}")
        
        # 건강 팁
        if pred[0] == 1:
            st.info("💡 **건강 관리 팁**: 규칙적인 유산소 운동, 저탄수화물 식단, 정기적인 혈당 체크를 추천합니다.")
        else:
            st.info("🎯 **현재 상태 유지 팁**: 적절한 체중 관리와 주기적인 건강검진을 계속하세요.")
    else:
        st.warning("⚠️ 모델 또는 스케일러를 불러오지 못했습니다. 파일을 확인해주세요.")

# 푸터
st.markdown("""
<div class="footer">
    <p>⚠️ 본 예측은 참고용이며, 의학적 진단을 대체하지 않습니다. 의료 상담은 반드시 전문의와 진행하세요.</p>
    <p>📊 모델 기준: Pima Indians Diabetes Database | 로지스틱 회귀 + 파생변수</p>
</div>
""", unsafe_allow_html=True)

# 처음 실행 시 가이드 표시
if "first_run" not in st.session_state:
    st.session_state.first_run = True
    with st.expander("🌟 **첫 사용자 가이드**"):
        st.markdown("""
        1. 좌우 슬라이더를 드래그해 본인의 건강 수치를 입력하세요.
        2. **예측하기** 버튼을 누르면 결과가 나타납니다.
        3. 당뇨 확률과 함께 개인화된 건강 팁을 확인할 수 있습니다.
        
        **파일 준비**: `diabetes_model.pkl`과 `scaler.pkl`이 현재 디렉토리에 있어야 합니다.
        """)
