from bz2 import compress
import email
from xml.dom import ValidationErr
from flask import Flask, render_template, redirect, url_for, request, flash, send_file
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import pip
import sys
import zipfile
import data_collection
import knn_impute
from datetime import datetime
from neural_network import nn_predictor, create_baseline
import lime_applied
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

glob_model = 0
glob_ypred = []
glob_y = []


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))


class UsernameErr(Exception):
    pass


class EmailErr(Exception):
    pass


db.create_all()

"""
remove_user_id = 1
User.query.filter(User.id == remove_user_id).delete()
db.session.commit()
"""

print(User.query.with_entities(User.email).all())


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')


class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])

    def validate_username(self, username):
        username_exists = User.query.filter_by(username=username.data).first()
        if username_exists:
            raise UsernameErr(
                "Username taken"
            )

    def validate_email(self, email):
        email_exists = User.query.filter_by(email=email.data).first()
        if email_exists:
            raise EmailErr(
                "Email taken"
            )


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('predict'))

        flash("Username or Password incorrect", 'error')
        return render_template('login.html', form=form)

    return render_template('login.html', form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    try:
        form = RegisterForm()
        if form.validate_on_submit():
            hashed_password = generate_password_hash(form.password.data, method='sha256')
            new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            og_directory = os.getcwd()
            new_directory = og_directory + '/registered/'
            if os.path.isdir(new_directory) == False:
                os.mkdir(og_directory + "/registered/")
            new_directory = og_directory + '/registered/' + str(new_user.id)
            os.mkdir(new_directory)
            flash("New User Created", 'success')
            return redirect('/login')
    except UsernameErr:
        flash("Username taken", "error")
        return render_template('signup.html', form=form)
    except EmailErr:
        flash("Email taken", "error")
        return render_template("signup.html", form=form)

    return render_template('signup.html', form=form)


@app.route('/file_upload', methods=['GET'])
@login_required
def hello_world():
    return render_template("file_upload.html")


@app.route('/file_upload', methods=["POST", "GET"])
@login_required
def predict():
    
    og_directory = os.getcwd()
    os.chdir(og_directory)
    
    compressedfile = request.files['compressedfile']
    checkfile = compressedfile

    compressed_path = "./registered/" + str(current_user.id) + '/' + compressedfile.filename

    try:
        compressedfile.save(compressed_path)
    except PermissionError:
        flash("Please upload a file", "error")
        return redirect(url_for('predict'))

    if os.path.isdir("./registered/" + str(current_user.id) + '/' + 'whole_test/'):
        shutil.rmtree("./registered/" + str(current_user.id) + '/' + 'whole_test/')
    print(compressed_path)

    try:
        with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
            print(zip_ref)
            zip_ref.extractall("./registered/" + str(current_user.id))
    except zipfile.BadZipFile:
        os.remove(compressed_path)
        flash("Please upload a zip file", "error")
        return redirect(url_for('predict'))

    try:
        full_dataset, file_id = data_collection.collect_data(
            "./registered/" + str(current_user.id) + '/' + 'whole_test/test_set/',
            "./registered/" + str(current_user.id) + '/' + 'whole_test/')
    except FileNotFoundError:
        os.remove(compressed_path)
        new_files = os.listdir("./registered/" + str(current_user.id))
        for h in range(len(new_files)):
            if new_files[h][-3:] != 'zip':
                os.remove("./registered/" + str(current_user.id) + "/" + new_files[h])
        flash("Please set up zip file correctly", "error")
        return redirect(url_for('predict'))

    knn_impute_dataset = knn_impute.knn_impute_data(full_dataset)

    os.chdir(og_directory)

    global glob_model,glob_y,glob_ypred,glob_ypred_prob

    glob_model, cr,glob_y,glob_ypred, glob_ypred_prob = nn_predictor(knn_impute_dataset, create_baseline())

    #explainer = lime_applied.LIME_explainer(model)

    #lime_applied.LIME_sample(knn_impute_dataset, model, explainer, 0)

    return redirect(url_for('analysis'))#render_template('file_upload.html', prediction=cr)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/history', defaults={'pathing': ''})
@app.route('/history/<path:pathing>')
@login_required
def history(pathing):
    og_directory = os.getcwd()
    compressed_path = og_directory + "./registered/" + str(current_user.id)

    if os.path.isfile(pathing):
        return send_file(pathing)

    def historystats(x):
        fileinfo = x.stat()
        filestime = datetime.fromtimestamp(fileinfo.st_mtime)
        filestime = datetime.strftime(filestime, '%d-%m-%Y %H:%M')
        filespath = os.path.relpath(x.path, compressed_path)
        if not os.path.isfile(filespath):
            folderexist = True
        else:
            folderexist = False
        return {'names': x.name, 'times': filestime, 'downloads': filespath, 'folder': folderexist}

    fileData = [historystats(x) for x in os.scandir(compressed_path)]
    fileDataMod = []
    for filed in fileData:
        if filed['folder'] == False:
            fileDataMod.append(filed)
    return render_template('history.html', files=fileDataMod)

@app.route('/history_analysis', methods=["POST", "GET"], defaults={'pathing': ''})
@app.route('/history_analysis/<path:pathing>')
@login_required
def history_analysis(pathing):
    og_directory = os.getcwd()
    os.chdir(og_directory)

    compressed_path = "./registered/" + str(current_user.id) + '/' + pathing
    if os.path.isdir("./registered/" + str(current_user.id) + '/' + 'whole_test/'):
        shutil.rmtree("./registered/" + str(current_user.id) + '/' + 'whole_test/')

    with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
        print(zip_ref)
        zip_ref.extractall("./registered/" + str(current_user.id))

    full_dataset, file_id = data_collection.collect_data(
        "./registered/" + str(current_user.id) + '/' + 'whole_test/test_set/',
        "./registered/" + str(current_user.id) + '/' + 'whole_test/')
    knn_impute_dataset = knn_impute.knn_impute_data(full_dataset)

    os.chdir(og_directory)

    global glob_model,glob_y,glob_ypred,glob_ypred_prob

    glob_model, cr,glob_y,glob_ypred,glob_ypred_prob = nn_predictor(knn_impute_dataset, create_baseline())

    #explainer = lime_applied.LIME_explainer(model)

    #lime_applied.LIME_sample(knn_impute_dataset, model, explainer, 0)

    return redirect(url_for('analysis'))#render_template('file_upload.html', prediction=cr)


@app.route('/analysis', methods=["POST", "GET"])
@login_required
def analysis():
    og_directory = os.getcwd()

    full_dataset, file_id = data_collection.collect_data(
        "./registered/" + str(current_user.id) + '/' + 'whole_test/test_set/',
        "./registered/" + str(current_user.id) + '/' + 'whole_test/')

    os.chdir(og_directory)

    list_id = []
    for i in file_id:
        i = i.split('.')
        list_id.append(i[0])

    print(list_id)
    os.chdir(og_directory)

    list_y = []
    list_ypred = []
    list_confidence = []
    total_patients = len(glob_y)
    total_y_survive = 0
    total_y_nsurvive = 0
    total_ypred_survive = 0
    total_ypred_nsurvive = 0
    expected_survive = 0
    likely_survive = 0
    probably_survive = 0
    probably_nsurvive = 0
    likely_nsurvive = 0
    expected_nsurvive = 0
    overall_score = 0
    expected_s_actual_s = 0
    expected_d_actual_d = 0

    # don't include expected correct

    for i in range(len(glob_ypred_prob)):
        actual_outcome = glob_y[i]
        if actual_outcome == 0.0:
            if glob_ypred_prob[i][0] > 0.95:
                expected_survive += 1
                overall_score += 50
                list_confidence.append("Expected")
                expected_s_actual_s += 1
            elif glob_ypred_prob[i][0] > 0.8:
                likely_survive += 1
                overall_score += 60
                list_confidence.append("Most Likely")
            elif glob_ypred_prob[i][0] > 0.5:
                probably_survive += 1
                overall_score += 70
                list_confidence.append("Likely")
            elif glob_ypred_prob[i][0] > 0.2:
                likely_nsurvive += 1
                overall_score += 80
                list_confidence.append("Unlikely")
            elif glob_ypred_prob[i][0] > 0.05:
                probably_nsurvive += 1
                overall_score += 90
                list_confidence.append("Most Unlikely")
            elif glob_ypred_prob[i][0] >= 0:
                expected_nsurvive += 1
                overall_score += 100
                list_confidence.append("Unexpected")
        elif actual_outcome == 1.0:
            if glob_ypred_prob[i][0] > 0.95:
                expected_survive += 1
                overall_score += 0
                list_confidence.append("Unexpected")
            elif glob_ypred_prob[i][0] > 0.8:
                likely_survive += 1
                overall_score += 10
                list_confidence.append("Most Unlikely")
            elif glob_ypred_prob[i][0] > 0.5:
                probably_survive += 1
                overall_score += 20
                list_confidence.append("Unlikely")
            elif glob_ypred_prob[i][0] > 0.2:
                likely_nsurvive += 1
                overall_score += 30
                list_confidence.append("Likely")
            elif glob_ypred_prob[i][0] > 0.05:
                probably_nsurvive += 1
                overall_score += 40
                list_confidence.append("Most Likely")
            elif glob_ypred_prob[i][0] >= 0:
                expected_nsurvive += 1
                overall_score += 50
                expected_d_actual_d += 1
                list_confidence.append("Expected")
    
    for i in range(len(glob_y)):
        if glob_y[i] == 1.0:
            list_y.append('Not Survived')
            total_y_nsurvive += 1
        elif glob_y[i] == 0.0:
            list_y.append('Survived')
            total_y_survive += 1

    for i in range(len(glob_ypred)):
        if glob_ypred[i] == 1.0:
            list_ypred.append('Not Survived')
            total_ypred_nsurvive += 1
        elif glob_ypred[i] == 0.0:
            list_ypred.append('Survived')
            total_ypred_survive += 1

    performance = overall_score - 50*(expected_d_actual_d + expected_s_actual_s) 
    final_score = performance/(total_patients - (expected_d_actual_d + expected_s_actual_s))

    post_id = list_id
    post_html = url_for('static', filename="lime_test.html")
    post_y = list_y
    post_ypred = list_ypred

    return render_template('analysis.html', post_html=post_html, post_id=post_id, post_y=post_y, 
    post_ypred=post_ypred, total_patients=total_patients, total_y_survive=total_y_survive, 
    total_ypred_survive=total_ypred_survive, final_score=final_score, list_confidence=list_confidence)


@app.route('/analysis_id', methods=["POST", "GET"])
@app.route('/analysis_id/<patient_id>')
def analysis_id(patient_id):
    og_directory = os.getcwd()
    patient_id = int(patient_id)

    full_dataset, file_id = data_collection.collect_data(
        "./registered/" + str(current_user.id) + '/' + 'whole_test/test_set/',
        "./registered/" + str(current_user.id) + '/' + 'whole_test/')

    knn_impute_dataset = knn_impute.knn_impute_data(full_dataset)

    os.chdir(og_directory)

    explainer = lime_applied.LIME_explainer(glob_model)

    lime_applied.LIME_sample(knn_impute_dataset, glob_model, explainer, patient_id)

    post_html = url_for('static', filename="lime_test.html")
    post_id = "Patient " + str(patient_id + 1)

    return render_template('analysis_id.html', post_html=post_html, post_id=post_id)


if __name__ == '__main__':
    app.run(debug=True)
