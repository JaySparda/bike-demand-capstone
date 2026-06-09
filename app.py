import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_prep import add_features, load_and_clean
from src.predict import load_model, predict

import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "hour.csv")


def main() -> None:
    st.set_page_config(page_title="Bike Demand Forecaster", page_icon="🚲", layout="wide")
    st.title("🚲 Bike Demand Forecaster")

    df = add_features(load_and_clean(DATA_PATH))

    # ---- Part 1: Analytics ----
    st.header("📊 Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        hourly = df.groupby("hr")["cnt"].mean().reset_index()
        fig1 = px.bar(hourly, x="hr", y="cnt",
                      title="Average Demand by Hour")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        avg_weather = df.groupby("weathersit")["cnt"].mean().reset_index()
        weather_labels = {1: "Clear", 2: "Mist", 3: "Light Rain/Snow", 4: "Heavy Rain/Snow"}
        avg_weather["weathersit"] = avg_weather["weathersit"].map(weather_labels)
        fig2 = px.bar(avg_weather, x="weathersit", y="cnt",
                      title="Average Demand by Weather Situation")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(df, x="temp", y="cnt", opacity=0.5,
                      title="Demand vs Temperature")
    st.plotly_chart(fig3, use_container_width=True)

    # ---- Part 2: Prediction ----
    st.header("🔮 Predict Hourly Demand")

    model = load_model()

    with st.form("predict_form"):
        r1, r2, r3 = st.columns(3)

        with r1:
            hour = st.slider("Hour", 0, 23, 8)
            weekday = st.selectbox("Weekday", options=list(range(7)),
                                   format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x])
            month = st.selectbox("Month", options=list(range(1, 13)))
            season = st.selectbox("Season", options=[1, 2, 3, 4],
                                  format_func=lambda x: ["Winter", "Spring", "Summer", "Fall"][x - 1])

        with r2:
            weathersit = st.selectbox("Weather", options=[1, 2, 3, 4],
                                      format_func=lambda x: ["Clear", "Mist", "Light Rain/Snow", "Heavy Rain/Snow"][x - 1])
            temp = st.slider("Temperature (norm.)", 0.0, 1.0, 0.5)
            atemp = st.slider("Feels-like (norm.)", 0.0, 1.0, 0.4)
            hum = st.slider("Humidity (norm.)", 0.0, 1.0, 0.5)

        with r3:
            windspeed = st.slider("Wind Speed (norm.)", 0.0, 1.0, 0.2)
            yr = st.selectbox("Year", options=[0, 1], format_func=lambda x: ["2011", "2012"][x])
            holiday = st.checkbox("Holiday")
            workingday = st.checkbox("Working Day", value=True)

        submitted = st.form_submit_button("Predict Demand", type="primary")

    if submitted:
        inputs = {
            "hr": hour, "weekday": weekday, "mnth": month,
            "season": season, "weathersit": weathersit,
            "temp": temp, "atemp": atemp, "hum": hum,
            "windspeed": windspeed, "yr": yr,
            "holiday": int(holiday), "workingday": int(workingday),
        }
        pred = predict(model, inputs)
        st.metric("Predicted Hourly Rentals", f"{pred:.0f}", delta=None)


if __name__ == "__main__":
    main()
