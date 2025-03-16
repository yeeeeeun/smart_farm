import streamlit as st
import pandas as pd
import joblib

# 모델 불러오기
models = joblib.load('best_models.joblib')
best_model_ec = models['ec_model']
best_model_ph = models['ph_model']

# 사용자 입력 받기
st.title('Nutrient Recommendation System')

stem_length = st.number_input('Stem Length (cm)', min_value=0.0, value=36.0)
leaf_cnt = st.number_input('Leaf Count', min_value=0, value=7)
stem_thick = st.number_input('Stem Thickness (mm)', min_value=0.0, value=15.9)
ti_value = st.number_input('TI Value (℃)', min_value=0.0, value=25.0)

if st.button('Predict'):
    input_data = pd.DataFrame([[stem_length, leaf_cnt, stem_thick, ti_value]],
                              columns=['stem_length', 'leaf_cnt', 'stem_thick', 'ti_value'])

    try:
        ec_value = best_model_ec.predict(input_data)[0]
        ph_value = best_model_ph.predict(input_data)[0]

        st.write("Recommended Nutrient Information:")
        st.write(f"  EC Value: {ec_value:.2f} dS/m")
        st.write(f"  PH Value: {ph_value:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
