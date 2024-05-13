import base64
import io

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import (Flask, jsonify, redirect, render_template, request, session,
                   url_for)

from flask_session import Session

matplotlib.use('Agg')

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load models and mappings
xgb_model = joblib.load('xgb_model.joblib')

country_mapping = joblib.load('country_mapping.joblib')
item_mapping = joblib.load('item_mapping.joblib')


def parse_float(value):
    try:
        return float(value) if value else None
    except ValueError:
        return None

# @app.route('/cropyieldprediction', methods=['GET', 'POST'])
# def index():
#     if 'data_entries' not in session or not session['data_entries']:
#         session['data_entries'] = []  # Initialize as an empty list if not already present

#     if request.method == 'POST' and 'add_entry' in request.form:
#         country_input = request.form.get('country')
#         item_input = request.form.get('item')

#         country_encoded = country_mapping.get(country_input)
#         item_encoded = item_mapping.get(item_input)

#         if country_encoded is None or item_encoded is None:
#             error_message = "Invalid country or item entered."
#             return render_template('index.html', error_message=error_message, entries=session['data_entries'])

#         entry = {
#             'Country': country_input,
#             'Item': item_input,
#             'Country_Encoded': country_encoded,
#             'Item_Encoded': item_encoded,
#             'Pesticides': parse_float(request.form.get('pesticides')),
#             'Avg_Temp': parse_float(request.form.get('avg_temp')),
#             'Rainfall': parse_float(request.form.get('rainfall'))
#         }
#         session['data_entries'].append(entry)
#         session.modified = True  # Mark session as modified to save changes
#         return redirect(url_for('index'))  # Redirect to the same page to display the table

#     # Serve the page with the data entries or an empty message
#     return render_template('index.html',
# entries=session.get('data_entries', []))


# my logic

@app.route('/')
def home():
    return render_template('wellcome.html')

@app.route('/cropyieldprediction', methods=['GET', 'POST'])
def index():
    if 'data_entries' not in session:
        session['data_entries'] = []

    if request.method == 'POST':
        if 'add_entry' in request.form:
            country_input = request.form.get('country')
            item_input = request.form.get('item')
            pesticides = parse_float(request.form.get('pesticides'))
            avg_temp = parse_float(request.form.get('avg_temp'))
            rainfall = parse_float(request.form.get('rainfall'))

            # Validate inputs
            if not (country_input and item_input and pesticides is not None and avg_temp is not None and rainfall is not None):
                error_message = "Please fill in all fields correctly."
                return render_template('index.html', error_message=error_message, entries=session['data_entries'])

            country_encoded = country_mapping.get(country_input)
            item_encoded = item_mapping.get(item_input)

            if country_encoded is None or item_encoded is None:
                error_message = "Invalid country or item entered."
                return render_template('index.html', error_message=error_message, entries=session['data_entries'])

            entry = {
                'Country': country_input,
                'Item': item_input,
                'Country_Encoded': country_encoded,
                'Item_Encoded': item_encoded,
                'Pesticides': pesticides,
                'Avg_Temp': avg_temp,
                'Rainfall': rainfall
            }
            session['data_entries'].append(entry)
            session.modified = True

        elif 'predict' in request.form:
            if not session['data_entries']:
                error_message = "No data available to predict."
                return render_template('index.html', error_message=error_message, entries=session['data_entries'])

            model_choice = request.form.get('model_choice')
            return redirect(url_for('get_predictions', model_choice=model_choice))

    return render_template('index.html', entries=session.get('data_entries', []))

@app.route('/predictions', methods=['GET'])
def get_predictions():
    model_choice = request.args.get('model_choice')
    if not model_choice or 'data_entries' not in session or not session['data_entries']:
        return jsonify({'error': 'Missing model choice or data entries'}), 400

    data = pd.DataFrame(session['data_entries'])
    X = data[['Country_Encoded', 'Item_Encoded', 'Pesticides', 'Avg_Temp', 'Rainfall']]

    if model_choice == 'XGBoost':
        model = xgb_model
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    predictions = model.predict(X)
    predictions_json = {'Predictions': predictions.tolist()}
    return jsonify(predictions_json)

@app.route('/reset', methods=['POST'])
def reset_entries():
    session.pop('data_entries', None)  # Clear the session data
    return redirect(url_for('index'))  # Redirect to clear the form



if __name__ == '__main__':
    app.run(debug=True, threaded=True)
