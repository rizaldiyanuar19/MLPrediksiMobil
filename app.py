import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ===============================
# LOAD DATA DAN TRAIN MODEL
# ===============================
df = pd.read_csv("scrap_price_clean_full.csv")

fitur = ['enginesize', 'horsepower', 'curbweight', 'carwidth', 'highwaympg', 'citympg']
target = 'price'

X = df[fitur]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================
# STREAMLIT APP
# ===============================
st.title("Prediksi Harga Mobil - Random Forest Regressor")

st.subheader("Visualisasi Model")

# Kolom visualisasi
col1, col2 = st.columns([1.2, 1.5])

# 1️⃣ Scatter plot
with col1:
    y_pred = model.predict(X_test)
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.scatter(y_test, y_pred, color='blue', alpha=0.6)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1)
    ax1.set_xlabel('Actual Price', fontsize=8)
    ax1.set_ylabel('Predicted Price', fontsize=8)
    ax1.set_title('Actual vs Predicted', fontsize=10)
    ax1.tick_params(axis='both', labelsize=7)
    ax1.grid(True, linewidth=0.5)
    st.pyplot(fig1)

# 2️⃣ Feature Importance
with col2:
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=fitur).sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    feat_importances.plot(kind='barh', color='teal', ax=ax2)
    ax2.set_title("Feature Importances", fontsize=10)
    ax2.set_xlabel("Importance Score", fontsize=8)
    ax2.set_ylabel("Feature", fontsize=8)
    ax2.tick_params(axis='both', labelsize=7)
    ax2.grid(True, linewidth=0.5)
    st.pyplot(fig2)

# ===============================
# INPUT USER UNTUK PREDIKSI
# ===============================
st.subheader("Masukkan Spesifikasi Mobil")

enginesize = st.number_input("Engine Size", float(X['enginesize'].min()), float(X['enginesize'].max()), float(X['enginesize'].mean()))
horsepower = st.number_input("Horsepower", float(X['horsepower'].min()), float(X['horsepower'].max()), float(X['horsepower'].mean()))
curbweight = st.number_input("Curb Weight", float(X['curbweight'].min()), float(X['curbweight'].max()), float(X['curbweight'].mean()))
carwidth = st.number_input("Car Width", float(X['carwidth'].min()), float(X['carwidth'].max()), float(X['carwidth'].mean()))
highwaympg = st.number_input("Highway MPG", float(X['highwaympg'].min()), float(X['highwaympg'].max()), float(X['highwaympg'].mean()))
citympg = st.number_input("City MPG", float(X['citympg'].min()), float(X['citympg'].max()), float(X['citympg'].mean()))

toleransi = st.slider("Toleransi Selisih Harga (USD)", 100, 10000, 2000)

if st.button("Prediksi Harga"):
    input_data = np.array([[enginesize, horsepower, curbweight, carwidth, highwaympg, citympg]])
    prediksi = model.predict(input_data)[0]
    st.success(f"Perkiraan harga mobil: ${prediksi:,.2f}")

    # Prediksi harga semua data
    df['predicted_price'] = model.predict(X)

    # Filter mobil serupa
    df_serupa = df[
        (df['predicted_price'] >= prediksi - toleransi) &
        (df['predicted_price'] <= prediksi + toleransi)
    ][['name', 'predicted_price']].sort_values(by='predicted_price')

    st.subheader("Mobil Lain dengan Harga Mendekati Prediksi")
    st.dataframe(df_serupa.reset_index(drop=True).style.format({"predicted_price": "${:,.2f}"}))