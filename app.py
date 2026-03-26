import os
import datetime
import joblib
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory, flash
from flask_sqlalchemy import SQLAlchemy
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super_secure_medical_key_2026')

# --- RENDER DATABASE CONFIGURATION ---
db_url = os.environ.get('DATABASE_URL', 'sqlite:///medical_system.db')
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

# --- DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(100), nullable=False)

class PatientDetail(db.Model):
    patient_id = db.Column(db.String(50), db.ForeignKey('user.id'), primary_key=True)
    dob = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False, default='Female')
    risk_status = db.Column(db.String(50), default="Pending Evaluation")
    risk_probability = db.Column(db.Float, nullable=True)

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), db.ForeignKey('user.id'), nullable=False)
    author_name = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), db.ForeignKey('user.id'), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    filename = db.Column(db.String(255), nullable=False)

class Bill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), default="Unpaid")

# --- INITIALIZATION ---
with app.app_context():
    db.create_all()
    if not User.query.filter_by(id='admin1').first():
        staff = [
            User(id='admin1', password_hash=generate_password_hash('Admin@123'), role='admin', name='System Admin'),
            User(id='doc1', password_hash=generate_password_hash('Doc@123'), role='doctor', name='Dr. Smith'),
            User(id='rad1', password_hash=generate_password_hash('Rad@123'), role='radiologist', name='Chief Radiologist'),
            User(id='pharm1', password_hash=generate_password_hash('Pharm@123'), role='pharmacist', name='Head Pharmacist')
        ]
        db.session.bulk_save_objects(staff)
        db.session.commit()

try:
    model = joblib.load('cervical_cancer_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selected_features = joblib.load('selected_features.pkl')
    feature_names = joblib.load('feature_names.pkl')
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    feature_names = ['Age', 'Smokes (years)', 'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)']

# --- ROUTES ---
@app.route('/')
def login_page():
    if 'user' in session: return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    user = User.query.get(request.form.get('username'))
    if user and check_password_hash(user.password_hash, request.form.get('password')):
        session['user'] = user.id
        session['role'] = user.role
        session['name'] = user.name
        return redirect(url_for('dashboard'))
    flash("Invalid Credentials", "danger")
    return redirect(url_for('login_page'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('login_page'))
    
    # Fetch data based on role
    patients = User.query.filter_by(role='patient').all() if session['role'] != 'patient' else []
    my_details = PatientDetail.query.get(session['user']) if session['role'] == 'patient' else None
    all_details = {p.patient_id: p for p in PatientDetail.query.all()}
    notes = Note.query.all()
    uploads = Upload.query.all()
    bills = Bill.query.all()

    return render_template('dashboard.html', 
                           role=session['role'], 
                           name=session['name'],
                           patients=patients,
                           my_details=my_details,
                           all_details=all_details,
                           notes=notes, uploads=uploads, bills=bills,
                           feature_names=feature_names)

@app.route('/create_patient', methods=['POST'])
def create_patient():
    if session.get('role') not in ['admin', 'doctor']: return redirect(url_for('dashboard'))
    
    gender = request.form.get('gender', '').lower()
    if gender not in ['female', 'f']:
        flash('System restricted to female patients only.', 'danger')
        return redirect(url_for('dashboard'))

    try:
        dob = datetime.datetime.strptime(request.form.get('dob'), '%Y-%m-%d').date()
        if dob >= datetime.date.today():
            flash('Date of birth must be a past date.', 'danger')
            return redirect(url_for('dashboard'))
    except:
        flash('Invalid date format.', 'danger')
        return redirect(url_for('dashboard'))

    if User.query.get(request.form.get('patient_id')):
        flash('Patient ID already exists.', 'danger')
        return redirect(url_for('dashboard'))

    new_user = User(id=request.form.get('patient_id'), password_hash=generate_password_hash(request.form.get('password')), role='patient', name=request.form.get('name'))
    new_patient = PatientDetail(patient_id=request.form.get('patient_id'), dob=dob, gender='Female')
    
    db.session.add(new_user)
    db.session.flush()  # <--- NEW: Forces the database to create the User first
    
    db.session.add(new_patient)
    db.session.commit() # <--- Finalizes both operations safely
    flash('Patient created successfully.', 'success')
    return redirect(url_for('dashboard'))

@app.route('/predict', methods=['POST'])
def predict():
    if session.get('role') != 'doctor': return redirect(url_for('dashboard'))
    patient_id = request.form.get('patient_id')
    
    if not MODEL_LOADED:
        flash('ML Model is offline.', 'danger')
        return redirect(url_for('dashboard'))

    input_data = {feat: float(request.form.get(feat, 0.0)) for feat in feature_names}
    df = pd.DataFrame([input_data])
    df_imputed = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(df), columns=feature_names)
    df_scaled = scaler.transform(df_imputed)
    df_selected = df_scaled[:, selected_features]
    
    prediction = int(model.predict(df_selected)[0])
    prob = float(model.predict_proba(df_selected)[0][1])
    
    patient = PatientDetail.query.get(patient_id)
    patient.risk_status = "High Risk" if prediction == 1 else "Low Risk"
    patient.risk_probability = prob
    db.session.commit()
    
    flash(f'Prediction Complete: {patient.risk_status}', 'success')
    return redirect(url_for('dashboard'))

@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        filename = secure_filename(f"{request.form.get('patient_id')}_{file.filename}")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        new_upload = Upload(patient_id=request.form.get('patient_id'), file_type=request.form.get('file_type'), filename=filename)
        db.session.add(new_upload)
        db.session.commit()
        flash('File uploaded.', 'success')
    return redirect(url_for('dashboard'))

@app.route('/add_note', methods=['POST'])
def add_note():
    db.session.add(Note(patient_id=request.form.get('patient_id'), author_name=session['name'], content=request.form.get('content')))
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/add_bill', methods=['POST'])
def add_bill():
    db.session.add(Bill(patient_id=request.form.get('patient_id'), amount=float(request.form.get('amount')), description=request.form.get('description')))
    db.session.commit()
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)