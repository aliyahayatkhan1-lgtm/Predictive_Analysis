import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Predictive Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Predictive Analytics Using Historical Data")
st.write(
    "A machine learning system that analyzes historical sales "
    "and predicts future sales trends."
)


# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("sales_data.csv")
    return data


data = load_data()


# -----------------------------
# Data Cleaning
# -----------------------------
data["Month"] = pd.to_datetime(data["Month"])

data = data.dropna()

data["Month_Number"] = np.arange(len(data))


# -----------------------------
# Dataset Information
# -----------------------------
st.header("1. Historical Sales Data")

st.dataframe(
    data[["Month", "Sales"]],
    use_container_width=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(data))

with col2:
    st.metric("Starting Sales", f"₹{data['Sales'].iloc[0]:,.0f}")

with col3:
    st.metric("Latest Sales", f"₹{data['Sales'].iloc[-1]:,.0f}")


# -----------------------------
# Historical Trend
# -----------------------------
st.header("2. Historical Sales Trend")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    data["Month"],
    data["Sales"],
    marker="o"
)

ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.set_title("Historical Monthly Sales")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


# -----------------------------
# Prepare Data
# -----------------------------
X = data[["Month_Number"]]
y = data["Sales"]


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()

model.fit(X_train, y_train)


# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# Model Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


st.header("3. Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Mean Absolute Error",
        f"₹{mae:,.2f}"
    )

with col2:
    st.metric(
        "RMSE",
        f"₹{rmse:,.2f}"
    )

with col3:
    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )


# -----------------------------
# Actual vs Predicted
# -----------------------------
st.header("4. Actual vs Predicted Sales")

comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

st.dataframe(
    comparison,
    use_container_width=True
)

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    label="Actual Sales"
)

ax2.plot(
    range(len(y_pred)),
    y_pred,
    marker="x",
    label="Predicted Sales"
)

ax2.set_xlabel("Test Data Points")
ax2.set_ylabel("Sales")
ax2.set_title("Actual vs Predicted Sales")
ax2.legend()

st.pyplot(fig2)


# -----------------------------
# Future Prediction
# -----------------------------
st.header("5. Future Sales Prediction")

months_to_predict = st.slider(
    "Select number of future months",
    min_value=1,
    max_value=12,
    value=6
)

future_numbers = np.arange(
    len(data),
    len(data) + months_to_predict
).reshape(-1, 1)

future_predictions = model.predict(future_numbers)


# Create future dates
last_date = data["Month"].iloc[-1]

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=months_to_predict,
    freq="MS"
)


future_data = pd.DataFrame({
    "Month": future_dates,
    "Predicted Sales": future_predictions
})


st.dataframe(
    future_data,
    use_container_width=True
)


# -----------------------------
# Future Prediction Graph
# -----------------------------
fig3, ax3 = plt.subplots(figsize=(10, 5))

ax3.plot(
    data["Month"],
    data["Sales"],
    marker="o",
    label="Historical Sales"
)

ax3.plot(
    future_data["Month"],
    future_data["Predicted Sales"],
    marker="x",
    linestyle="--",
    label="Predicted Sales"
)

ax3.set_xlabel("Month")
ax3.set_ylabel("Sales")
ax3.set_title("Future Sales Forecast")

ax3.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig3)


# -----------------------------
# Conclusion
# -----------------------------
st.header("6. Conclusion")

st.success(
    "The Linear Regression model successfully learned the historical "
    "sales trend and generated predictions for future months. "
    "The results can be used to understand sales trends and support "
    "future business planning."
)