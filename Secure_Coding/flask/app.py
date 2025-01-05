from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load your dataset and models
data = pd.read_csv("C:\\Users\\kowsi\\Desktop\\sem 5\\Secure_coding\\flask\\new.csv")
data["Sentence"].fillna("", inplace=True)
data.dropna(subset=["Label"], inplace=True)

X = data['Sentence']
y = data['Label']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_vectorized, y)

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=16)
random_forest_classifier.fit(X_vectorized, y)

decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_vectorized, y)

# Function to predict SQL injection attempts
def detect_sql_injection(query, classifier):
    query_vectorized = vectorizer.transform([query])
    prediction = classifier.predict(query_vectorized)
    return "Malicious SQL Injection Attempt" if prediction == '0' else "Benign SQL Query"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/newpage', methods=['GET', 'POST'])
def newpage():
    if request.method == 'POST':
        user_input = request.form['user_input']
        
        logistic_result = detect_sql_injection(user_input, logistic_classifier)
        random_forest_result = detect_sql_injection(user_input, random_forest_classifier)
        decision_tree_result = detect_sql_injection(user_input, decision_tree_classifier)
        
        return render_template('result.html', logistic_result=logistic_result,
                                random_forest_result=random_forest_result,
                                decision_tree_result=decision_tree_result)
    
    return render_template('newpage.html')
    

if __name__ == '__main__':
    app.run(debug=True)
