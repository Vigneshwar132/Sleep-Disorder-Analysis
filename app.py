from flask import Flask, url_for, redirect, render_template, request, session
import mysql.connector, os, re
import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report
# from xgboost import XGBClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import StackingClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
# import joblib
# from imblearn.over_sampling import SMOTE
import pickle
# import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)
app.secret_key = 'admin'

mydb = mysql.connector.connect(
    host="db4free.net",
    user="vigneshwar",
    password="$d.9%P_A%j98JLb",
    port="3306",
    database='sleepingdisorder'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return render_template('home.html')
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')






### Prediction part

# Load the scaler
with open(r'Models\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the stacking classifier model
model_path = r'Models\Stacking Classifier_model.pkl'
with open(model_path, 'rb') as f:
    stacking_classifier = pickle.load(f)

class_labels = {0: 'Insomnia', 1: 'Healthy', 2: 'Sleep Apnea'}

def prediction_func(input):

    # Standardize the features
    input_scaled = scaler.transform([input])

    # Predict using the stacking classifier model
    prediction = stacking_classifier.predict(input_scaled)

    # Map prediction to class labels
    predicted_class = class_labels[prediction[0]]

    return predicted_class


@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    result = None
    if request.method == "POST":
        Gender = request.form['Gender']
        Age = int(request.form['Age'])
        Occupation = request.form['Occupation']
        Sleep_Duration = float(request.form['Sleep_Duration'])
        Quality_of_Sleep = int(request.form['Quality_of_Sleep'])
        Physical_Activity_Level = int(request.form['Physical_Activity_Level'])
        Stress_Level = int(request.form['Stress_Level'])
        BMI_Category = request.form['BMI_Category']
        Heart_Rate = int(request.form['Heart_Rate'])
        Daily_Steps = int(request.form['Daily_Steps'])


        result = prediction_func([Gender, Age, Occupation, Sleep_Duration, Quality_of_Sleep, Physical_Activity_Level, Stress_Level, BMI_Category, Heart_Rate, Daily_Steps])


    
    # Load the dataset
    df = pd.read_csv(r"Dataset\Sleep_health_and_lifestyle_dataset.csv")

    # Drop columns
    columns_to_drop = ['Sleep Disorder', 'Blood Pressure']
    df = df.drop(columns=columns_to_drop)

    # Replace spaces in column names with underscores
    df.columns = [re.sub(r'\s+', '_', col) for col in df.columns]

    # Define object columns to be encoded
    object_columns = df.select_dtypes(include=['object']).columns

    # Store label counts before encoding
    labels = {col: df[col].value_counts().to_dict() for col in object_columns}

    # Initialize LabelEncoder
    le = LabelEncoder()

    # # Encode categorical columns and store the encoded value counts
    encodes = {}
    for col in object_columns:
        df[col] = le.fit_transform(df[col])
        value_counts = df[col].value_counts().to_dict()
        encodes[col] = value_counts

    dic = {}

    for key in labels.keys():
        dic[key] = []
        for sub_key, value in labels[key].items():
            for id_key, id_value in encodes[key].items():
                if value == id_value:
                    dic[key].append((sub_key, id_key))
                    break

    return render_template('prediction.html', data=dic, prediction=result)








@app.route('/upload', methods=["GET", "POST"])
def upload():
    message = ""
    img_base64 = None
    df = None

    if request.method == "POST":
        file = request.files['file']
        extn = file.filename.rsplit('.', 1)[1].lower()

        if extn == "csv":
            df = pd.read_csv(file)
        elif extn in ["xls", "xlsx"]:
            df = pd.read_excel(file)
        else:
            message = "Unsupported file type. Please upload a .csv or .xls/.xlsx file."
            return render_template('upload.html', message=message)


        if "target" in df.columns:
            value_counts = df["target"].value_counts()

            # Create a pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
            ax.axis('equal')

            # Save the figure to a BytesIO object
            img = io.BytesIO()
            fig.savefig(img, format='png')
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
            return render_template('upload.html', img_data=img_base64)
        else:
            message = "'target' column not found in the uploaded file. Please ensure the file has a 'target' column."
            return render_template('upload.html', message=message)
    return render_template('upload.html', message=message)





@app.route('/result')
def result():
    return render_template('result.html')



if __name__ == '__main__':
    app.run(debug = True, port=8080)
