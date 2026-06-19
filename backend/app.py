# backend/app.py

import sqlite3
import os
from flask import Flask, request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS # Import CORS

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_strong_random_secret_key_here' # *** CHANGE THIS TO A UNIQUE, STRONG KEY IN PRODUCTION! ***

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, 'voice_disorder.db')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
BALANCED_SPEECH_DATASET_PATH = os.path.join(BASE_DIR, 'Balanced_Speech_Dataset.csv')
DOCTOR_DATASET_PATH = os.path.join(BASE_DIR, 'Doctor.csv')

# Simplified and robust CORS configuration
CORS(app)

# --- Database Helper Functions ---

def get_db():
    """Establishes a database connection or returns the existing one."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row # This allows accessing columns by name
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Closes the database connection at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()

        # --- CREATE TABLES ---

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS disorders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symptoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                expertise TEXT NOT NULL,
                experience_years INTEGER NOT NULL,
                is_available INTEGER NOT NULL,
                patients_allocated INTEGER DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symptoms_selected TEXT NOT NULL,
                predicted_disorder TEXT NOT NULL,
                report_date TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                report_id INTEGER UNIQUE,
                disorder_name TEXT NOT NULL,
                appointment_date TEXT NOT NULL,
                appointment_time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (doctor_id) REFERENCES doctors(id),
                FOREIGN KEY (report_id) REFERENCES user_reports(id) ON DELETE CASCADE
            )
        ''')

        db.commit()

        # --- INSERT DATA ONLY IF EMPTY ---

        # Disorders + Symptoms
        cursor.execute("SELECT COUNT(*) FROM disorders")
        if cursor.fetchone()[0] == 0:
            try:
                df = pd.read_csv(BALANCED_SPEECH_DATASET_PATH)

                disorders = df['Disorder'].unique().tolist()
                symptoms = [col for col in df.columns if col not in ['Patient_ID', 'Age', 'Gender', 'Disorder']]

                for d in disorders:
                    cursor.execute("INSERT INTO disorders (name) VALUES (?)", (d,))

                for s in symptoms:
                    cursor.execute("INSERT INTO symptoms (name) VALUES (?)", (s,))

                db.commit()
                print("Inserted disorders & symptoms")

            except Exception as e:
                print(f"Error loading speech dataset: {e}")

        # Doctors
        cursor.execute("SELECT COUNT(*) FROM doctors")
        if cursor.fetchone()[0] == 0:
            try:
                df = pd.read_csv(DOCTOR_DATASET_PATH)

                for _, row in df.iterrows():
                    cursor.execute('''
                        INSERT INTO doctors 
                        (doctor_id, name, expertise, experience_years, is_available, patients_allocated)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        row['Doctor_ID'],
                        row['Doctor_Name'],
                        row['Expertise'],
                        row['Experience (Years)'],
                        row['Availability'],
                        row['Patient_Allocated']
                    ))

                db.commit()
                print("Inserted doctors")

            except Exception as e:
                print(f"Error loading doctor dataset: {e}")

        print("DB Ready ✅")

# --- ML Model Loading ---
model = None
symptom_names = [] # To store the ordered list of symptoms used by the model
disorder_names = [] # To store the ordered list of disorders used by the model

def load_model_and_data():
    """Loads the trained ML model and associated data."""
    global model, symptom_names, disorder_names
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
                model = data['model']
                symptom_names = data['symptom_names']
                disorder_names = data['disorder_names']
            print("ML model loaded successfully.")
        except Exception as e:
            print(f"Error loading ML model: {e}")
            model = None
    else:
        print(f"ML model not found at {MODEL_PATH}. Please run ml_model.py first to train the model.")

# Load model when app starts
load_model_and_data()

# --- JWT Authentication Decorator ---

def token_required(f):
    """Decorator to protect routes that require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            db = get_db()
            cursor = db.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (data['user_id'],))
            current_user = cursor.fetchone()
            if not current_user:
                return jsonify({'message': 'Token is invalid!'}), 401
        except Exception as e:
            print(f"Token error: {e}")
            return jsonify({'message': 'Token is invalid or expired!'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# --- API Routes ---

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required!'}), 400

    db = get_db()
    cursor = db.cursor()

    try:
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        db.commit()
        return jsonify({'message': 'User registered successfully!'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Username already exists!'}), 409
    except Exception as e:
        return jsonify({'message': f'An error occurred: {e}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required!'}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if not user or not check_password_hash(user['password'], password):
        return jsonify({'message': 'Invalid username or password!'}), 401

    token = jwt.encode(
        {'user_id': user['id'], 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)},
        app.config['SECRET_KEY'],
        algorithm="HS256"
    )
    return jsonify({'token': token, 'username': user['username']}), 200

@app.route('/api/symptoms', methods=['GET'])
@token_required
def get_symptoms(current_user):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, name FROM symptoms ORDER BY name")
    symptoms = cursor.fetchall()
    return jsonify([dict(symptom) for symptom in symptoms]), 200

@app.route('/api/submit_test', methods=['POST'])
@token_required
def submit_test(current_user):
    if model is None or not symptom_names:
        return jsonify({'message': 'ML model not loaded or trained yet. Please train the model.'}), 500

    data = request.get_json()
    selected_symptom_ids = data.get('symptom_ids', [])

    if not selected_symptom_ids:
        return jsonify({'message': 'No symptoms selected.'}), 400

    db = get_db()
    cursor = db.cursor()

    # Get selected symptom names from IDs
    selected_symptom_names = []
    for symptom_id in selected_symptom_ids:
        cursor.execute("SELECT name FROM symptoms WHERE id = ?", (symptom_id,))
        symptom = cursor.fetchone()
        if symptom:
            selected_symptom_names.append(symptom['name'])

    # Prepare input for the ML model
    # Create a feature vector (1s for selected symptoms, 0s otherwise)
    input_features = np.zeros(len(symptom_names))
    for i, symptom_name in enumerate(symptom_names):
        if symptom_name in selected_symptom_names:
            input_features[i] = 1

    # Reshape for single prediction
    input_features = input_features.reshape(1, -1)

    # Predict disorder
    try:
        prediction_index = model.predict(input_features)[0]
        predicted_disorder = disorder_names[prediction_index]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'message': 'Error predicting disorder.'}), 500

    # Save report to database
    report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symptoms_str = ','.join(map(str, selected_symptom_ids)) # Store IDs for simplicity

    cursor.execute(
        "INSERT INTO user_reports (user_id, symptoms_selected, predicted_disorder, report_date) VALUES (?, ?, ?, ?)",
        (current_user['id'], symptoms_str, predicted_disorder, report_date)
    )
    db.commit()
    report_id = cursor.lastrowid # Get the ID of the newly created report

    return jsonify({
        'message': 'Test submitted successfully!',
        'predicted_disorder': predicted_disorder,
        'report_id': report_id # Return the report_id
    }), 200

@app.route('/api/reports', methods=['GET'])
@token_required
def get_reports(current_user):
    db = get_db()
    cursor = db.cursor()
    # Join with appointments and doctors to get linked appointment and doctor details
    cursor.execute(
        """
        SELECT
            ur.id AS report_id,
            ur.symptoms_selected,
            ur.predicted_disorder,
            ur.report_date,
            a.id AS appointment_id,
            a.appointment_date,
            a.appointment_time,
            a.status AS appointment_status,
            d.name AS doctor_name,
            d.expertise AS doctor_specialty
        FROM
            user_reports ur
        LEFT JOIN
            appointments a ON ur.id = a.report_id
        LEFT JOIN
            doctors d ON a.doctor_id = d.id
        WHERE
            ur.user_id = ?
        ORDER BY
            ur.report_date DESC
        """,
        (current_user['id'],)
    )
    reports = cursor.fetchall()

    # Convert symptom IDs back to names for display
    all_symptoms = {}
    cursor.execute("SELECT id, name FROM symptoms")
    for s in cursor.fetchall():
        all_symptoms[s['id']] = s['name']

    formatted_reports = []
    for report in reports:
        selected_ids = [int(s_id) for s_id in report['symptoms_selected'].split(',') if s_id]
        selected_names = [all_symptoms.get(s_id, f"Unknown Symptom ID: {s_id}") for s_id in selected_ids]

        report_data = {
            'id': report['report_id'],
            'symptoms_selected': selected_names,
            'predicted_disorder': report['predicted_disorder'],
            'report_date': report['report_date'],
            'appointment': None # Initialize appointment data
        }

        # If there's a linked appointment, add its details
        if report['appointment_id']:
            report_data['appointment'] = {
                'id': report['appointment_id'],
                'date': report['appointment_date'],
                'time': report['appointment_time'],
                'status': report['appointment_status'],
                'doctor_name': report['doctor_name'],
                'doctor_specialty': report['doctor_specialty']
            }
        formatted_reports.append(report_data)
    return jsonify(formatted_reports), 200

@app.route('/api/book_appointment', methods=['POST'])
@token_required
def book_appointment(current_user):
    data = request.get_json()
    disorder_name = data.get('disorder')
    report_id = data.get('report_id') # Expect report_id from frontend

    if not disorder_name or not report_id:
        return jsonify({'message': 'Disorder name and report ID are required.'}), 400

    db = get_db()
    cursor = db.cursor()

    # Check if this report already has an appointment
    cursor.execute("SELECT id FROM appointments WHERE report_id = ?", (report_id,))
    if cursor.fetchone():
        return jsonify({'message': 'An appointment for this report already exists.'}), 409

    # --- Doctor Allocation Logic ---
    # 1. Try to find an available doctor specializing in the predicted disorder,
    #    prioritizing by experience (descending).
    cursor.execute(
        "SELECT id, name, expertise, experience_years FROM doctors WHERE expertise = ? AND is_available = 1 ORDER BY experience_years DESC LIMIT 1",
        (disorder_name,)
    )
    doctor = cursor.fetchone()

    if not doctor:
        # 2. If no available specialist, try to find an unavailable specialist (for future booking consideration)
        cursor.execute(
            "SELECT id, name, expertise, experience_years FROM doctors WHERE expertise = ? ORDER BY experience_years DESC LIMIT 1",
            (disorder_name,)
        )
        doctor = cursor.fetchone()
        if doctor: # If an unavailable specialist is found
            return jsonify({
                'message': f'No available doctor found for {disorder_name} at this time. Dr. {doctor["name"]} specializes in this, but is currently unavailable. Please try again later or contact them directly.',
                'doctor_name': doctor['name'],
                'doctor_specialty': doctor['expertise'],
                'is_available': False
            }), 404 # Using 404 to indicate no *available* resource

    if not doctor:
        # 3. If no specialist found at all (available or unavailable), return no doctor found.
        return jsonify({'message': f'No suitable doctor found for {disorder_name} at this time.'}), 404

    # Mock appointment time (e.g., next available hour)
    appointment_date = datetime.date.today().strftime("%Y-%m-%d")
    appointment_time = (datetime.datetime.now() + datetime.timedelta(hours=1)).strftime("%H:00")

    try:
        # Update doctor's availability and allocated patients
        cursor.execute(
            "UPDATE doctors SET is_available = 0, patients_allocated = patients_allocated + 1 WHERE id = ?",
            (doctor['id'],)
        )

        # Insert new appointment linked to the report
        cursor.execute(
            "INSERT INTO appointments (user_id, doctor_id, report_id, disorder_name, appointment_date, appointment_time, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (current_user['id'], doctor['id'], report_id, disorder_name, appointment_date, appointment_time, 'pending')
        )
        db.commit()
        appointment_id = cursor.lastrowid

        return jsonify({
            'message': 'Appointment booked successfully!',
            'appointment_id': appointment_id,
            'doctor_name': doctor['name'],
            'doctor_specialty': doctor['expertise'], # Use 'expertise' as specialty
            'appointment_date': appointment_date,
            'appointment_time': appointment_time,
            'is_available': True,
            'appointment_status': 'pending'
        }), 201
    except Exception as e:
        print(f"Error booking appointment: {e}")
        return jsonify({'message': f'Error booking appointment: {e}'}), 500

@app.route('/api/finish_appointment', methods=['POST'])
@token_required
def finish_appointment(current_user):
    data = request.get_json()
    appointment_id = data.get('appointment_id')

    if not appointment_id:
        return jsonify({'message': 'Appointment ID is required.'}), 400

    db = get_db()
    cursor = db.cursor()

    # Verify appointment belongs to user and is pending
    cursor.execute(
        "SELECT a.id, a.doctor_id, a.status, d.patients_allocated FROM appointments a JOIN doctors d ON a.doctor_id = d.id WHERE a.id = ? AND a.user_id = ?",
        (appointment_id, current_user['id'])
    )
    appointment = cursor.fetchone()

    if not appointment:
        return jsonify({'message': 'Appointment not found or does not belong to user.'}), 404
    if appointment['status'] == 'completed':
        return jsonify({'message': 'Appointment already marked as completed.'}), 400

    try:
        # Update appointment status
        cursor.execute(
            "UPDATE appointments SET status = 'completed' WHERE id = ?",
            (appointment_id,)
        )

        # Reset doctor availability and decrement allocated patients
        # Ensure patients_allocated is treated as an integer, defaulting to 0 if None
        current_patients = appointment['patients_allocated'] if appointment['patients_allocated'] is not None else 0
        new_patients_allocated = max(0, current_patients - 1)
        
        cursor.execute(
            "UPDATE doctors SET is_available = 1, patients_allocated = ? WHERE id = ?",
            (new_patients_allocated, appointment['doctor_id'])
        )
        db.commit()
        return jsonify({'message': 'Appointment marked as finished and doctor availability updated.'}), 200
    except Exception as e:
        print(f"Error finishing appointment: {e}")
        return jsonify({'message': f'Error finishing appointment: {e}'}), 500

@app.route('/api/delete_report/<int:report_id>', methods=['DELETE'])
@token_required
def delete_report(current_user, report_id):
    db = get_db()
    cursor = db.cursor()

    # Check if report belongs to the current user
    cursor.execute("SELECT id FROM user_reports WHERE id = ? AND user_id = ?", (report_id, current_user['id']))
    report_exists = cursor.fetchone()
    if not report_exists:
        return jsonify({'message': 'Report not found or does not belong to user.'}), 404

    try:
        # Check for and handle linked appointment
        cursor.execute("SELECT id, doctor_id FROM appointments WHERE report_id = ?", (report_id,))
        linked_appointment = cursor.fetchone()

        if linked_appointment:
            appointment_id_to_delete = linked_appointment['id']
            doctor_id_to_update = linked_appointment['doctor_id']

            # Delete the linked appointment
            cursor.execute("DELETE FROM appointments WHERE id = ?", (appointment_id_to_delete,))

            # Decrement doctor's allocated patients and set to available if no other pending appointments
            cursor.execute("SELECT patients_allocated FROM doctors WHERE id = ?", (doctor_id_to_update,))
            
            # Ensure patients_allocated is treated as an integer, defaulting to 0 if None
            doctor_data = cursor.fetchone()
            current_patients = doctor_data['patients_allocated'] if doctor_data and doctor_data['patients_allocated'] is not None else 0
            new_patients_allocated = max(0, current_patients - 1)

            # Check if this doctor has other pending appointments
            cursor.execute("SELECT COUNT(*) FROM appointments WHERE doctor_id = ? AND status = 'pending'", (doctor_id_to_update,))
            other_pending_appointments = cursor.fetchone()[0]

            if other_pending_appointments == 0:
                cursor.execute(
                    "UPDATE doctors SET is_available = 1, patients_allocated = ? WHERE id = ?",
                    (new_patients_allocated, doctor_id_to_update)
                )
            else:
                cursor.execute(
                    "UPDATE doctors SET patients_allocated = ? WHERE id = ?",
                    (new_patients_allocated, doctor_id_to_update)
                )
            print(f"Linked appointment {appointment_id_to_delete} deleted and doctor {doctor_id_to_update} updated.")


        # Delete the user report
        cursor.execute("DELETE FROM user_reports WHERE id = ?", (report_id,))
        db.commit()

        return jsonify({'message': 'Report and associated appointment (if any) deleted successfully.'}), 200
    except Exception as e:
        db.rollback() # Rollback changes if an error occurs
        print(f"Error deleting report: {e}")
        return jsonify({'message': f'Error deleting report: {e}'}), 500

# If you run this file directly, it will start the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
