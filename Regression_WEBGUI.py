#!/usr/bin/env python3

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import tkinter as tk
from tkinter import ttk

def preprocess_data(df):
    # Extract unique values before dropping columns
    conditions = df['condition'].astype(str).unique().tolist()
    periods = df['period'].astype(str).unique().tolist()
    movements = df['movement'].astype(str).unique().tolist()
    signed_values = df['signed'].astype(str).unique().tolist()

    movement_df = pd.get_dummies(df['movement'])
    period_df = pd.get_dummies(df['period'])
    if '[nan]' in movement_df.columns:
        movement_df.drop(columns='[nan]', inplace=True)
    if '[nan]' in period_df.columns:
        period_df.drop(columns='[nan]', inplace=True)

    df.drop(columns=['movement', 'period'], inplace=True)
    df = df.join(period_df)

    df['yearCreation'] = pd.to_numeric(df['yearCreation'], errors="coerce")
    df['yearCreation'] = df['yearCreation'].fillna(df['yearCreation'].median())

    df['price'] = df['price'].str.replace("USD", "").str.replace(".", "").astype(float)

    vectorizer = CountVectorizer()
    condition = vectorizer.fit_transform(df['condition'].astype(str))
    condition_labels = pd.DataFrame(condition.toarray(), columns=[f'condition_{col}' for col in vectorizer.get_feature_names_out()])

    vectorizer2 = CountVectorizer()
    signed = vectorizer2.fit_transform(df['signed'].astype(str))
    signed_labels = pd.DataFrame(signed.toarray(), columns=[f'signed_{col}' for col in vectorizer2.get_feature_names_out()])

    df.drop(columns=['signed', 'condition', 'artist'], inplace=True)
    signed_labels = signed_labels.astype(int)
    condition_labels = condition_labels.astype(int)
    df = df.join(condition_labels)
    df = df.join(signed_labels, lsuffix="_left", rsuffix="_right", how="right")
    df = df.drop(df.columns[[0, 1, 2, 5]], axis=1)

    return df, vectorizer, vectorizer2, conditions, periods, movements, signed_values

def train_model(df):
    X = df.loc[:, ~df.columns.isin(['artist', 'title', 'signed', 'condition', 'title_left', 'price'])]
    Y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    RF_regressor = RandomForestRegressor(max_depth=20, random_state=0)
    RF_regressor.fit(X_train, y_train)

    print("Training score: ", RF_regressor.score(X_train, y_train))
    print("Testing score: ", RF_regressor.score(X_test, y_test))
    print("Mean abs error: ", mean_absolute_error(y_test, RF_regressor.predict(X_test)))

    return RF_regressor, X.columns

def predict_price(new_data, model, vectorizer, vectorizer2, feature_labels):
    new_data_df = pd.DataFrame([new_data])

    new_data_df['yearCreation'] = pd.to_numeric(new_data_df['yearCreation'], errors="coerce")
    new_data_df['yearCreation'] = new_data_df['yearCreation'].fillna(new_data_df['yearCreation'].median())

    condition = vectorizer.transform(new_data_df['condition'].astype(str))
    condition_labels = pd.DataFrame(condition.toarray(), columns=[f'condition_{col}' for col in vectorizer.get_feature_names_out()])

    signed = vectorizer2.transform(new_data_df['signed'].astype(str))
    signed_labels = pd.DataFrame(signed.toarray(), columns=[f'signed_{col}' for col in vectorizer2.get_feature_names_out()])

    new_data_df = new_data_df.join(condition_labels)
    new_data_df = new_data_df.join(signed_labels, lsuffix="_left", rsuffix="_right", how="right")
    new_data_df = new_data_df.reindex(columns=feature_labels, fill_value=0)

    prediction = model.predict(new_data_df)
    return prediction[0]

# Load and preprocess data
df = pd.read_csv("Sothebys_Data.csv")
df, vectorizer, vectorizer2, conditions, periods, movements, signed_values = preprocess_data(df)

# Train the model
model, feature_labels = train_model(df)

# Create the GUI
class ArtPricePredictorApp:
    def __init__(self, root, conditions, periods, movements, signed_values):
        self.root = root
        self.root.title("Art Price Predictor")

        self.yearCreation_label = ttk.Label(root, text="Year of Creation:")
        self.yearCreation_label.grid(row=0, column=0)
        self.yearCreation_entry = ttk.Entry(root)
        self.yearCreation_entry.grid(row=0, column=1)

        self.condition_label = ttk.Label(root, text="Condition:")
        self.condition_label.grid(row=1, column=0)
        self.condition_combobox = ttk.Combobox(root, values=conditions)
        self.condition_combobox.grid(row=1, column=1)

        self.signed_label = ttk.Label(root, text="Signed:")
        self.signed_label.grid(row=2, column=0)
        self.signed_combobox = ttk.Combobox(root, values=signed_values)
        self.signed_combobox.grid(row=2, column=1)

        self.period_label = ttk.Label(root, text="Period:")
        self.period_label.grid(row=3, column=0)
        self.period_combobox = ttk.Combobox(root, values=periods)
        self.period_combobox.grid(row=3, column=1)

        self.movement_label = ttk.Label(root, text="Movement:")
        self.movement_label.grid(row=4, column=0)
        self.movement_combobox = ttk.Combobox(root, values=movements)
        self.movement_combobox.grid(row=4, column=1)

        self.predict_button = ttk.Button(root, text="Predict Price", command=self.predict_price)
        self.predict_button.grid(row=5, column=0, columnspan=2)

        self.result_label = ttk.Label(root, text="")
        self.result_label.grid(row=6, column=0, columnspan=2)

    def predict_price(self):
        new_art = {
            'popularity': 0,  # Default value since it's not included in the input fields
            'title': 'Sample Art',  # Default value since it's not included in the input fields
            'yearCreation': self.yearCreation_entry.get(),
            'signed': self.signed_combobox.get(),
            'condition': self.condition_combobox.get(),
            'period': self.period_combobox.get(),
            'movement': self.movement_combobox.get()
        }
        predicted_price = predict_price(new_art, model, vectorizer, vectorizer2, feature_labels)
        self.result_label.config(text=f"The predicted price for the artwork is: {predicted_price:.2f} USD")

if __name__ == "__main__":
    root = tk.Tk()
    app = ArtPricePredictorApp(root, conditions, periods, movements, signed_values)
    root.mainloop()
