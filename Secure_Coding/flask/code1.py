import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your dataset with 'sentence' and 'label' columns
# Replace 'your_dataset.csv' with the actual path to your dataset file
data = pd.read_csv("new.csv")

data1 = data.copy()

data["Sentence"].fillna("", inplace=True)

data.dropna(subset=["Label"], inplace=True)

data2 = data.copy()

# Preprocess the data
X = data['Sentence']
y = data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorize the text data using TF-IDF (you can try different vectorization methods)
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Build Logistic Regression classifier
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train_vectorized, y_train)

# Build Random Forest classifier

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=0, max_depth = 16)
random_forest_classifier.fit(X_train_vectorized, y_train)

#print("the score of Random forest algorithm accuracy : ",random_forest_classifier.score(X_train_vectorized, y_train))

# random_forest_classifier = RandomForestClassifier()
# random_forest_classifier.fit(X_train_vectorized, y_train)

# Build Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train_vectorized, y_train)

# Evaluate Logistic Regression model on the test set
logistic_y_pred = logistic_classifier.predict(X_test_vectorized)
logistic_accuracy = accuracy_score(y_test, logistic_y_pred)
print(f'The score of Logistic regression accuracy: {logistic_accuracy:.2f}')

# Evaluate Random Forest model on the test set
random_forest_y_pred = random_forest_classifier.predict(X_test_vectorized)
random_forest_accuracy = accuracy_score(y_test, random_forest_y_pred)
print(f'The score of Random forest algorithm accuracy :  {random_forest_accuracy:.2f}')

# Evaluate Decision Tree model on the test set
decision_tree_y_pred = decision_tree_classifier.predict(X_test_vectorized)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_y_pred)
print(f'The score of Decision Tree accuracy algorithm: {decision_tree_accuracy:.2f}')

# Function to predict SQL injection attempts using Logistic Regression
def detect_sql_injection_logistic(query):
    # Vectorize the user input
    query_vectorized = vectorizer.transform([query])

    # Predict using the Logistic Regression classifier
    prediction = logistic_classifier.predict(query_vectorized)

    # Return the result
    if prediction == '0':
        return "Malicious SQL Injection Attempt (Logistic Regression)"
    else:
        return "Benign SQL Query (Logistic Regression)"

# Function to predict SQL injection attempts using Random Forest
def detect_sql_injection_random_forest(query):
    # Vectorize the user input
    query_vectorized = vectorizer.transform([query])

    # Predict using the Random Forest classifier
    prediction = random_forest_classifier.predict(query_vectorized)

    # Return the result
    if prediction == '0':
        return "Malicious SQL Injection Attempt (Random Forest)"
    else:
        return "Benign SQL Query (Random Forest)"

# Function to predict SQL injection attempts using Decision Tree
def detect_sql_injection_decision_tree(query):
    # Vectorize the user input
    query_vectorized = vectorizer.transform([query])

    # Predict using the Decision Tree classifier
    prediction = decision_tree_classifier.predict(query_vectorized)

    # Return the result
    if prediction == '0':
        return "Malicious SQL Injection Attempt (Decision Tree)"
    else:
        return "Benign SQL Query (Decision Tree)"

# Example user input
user_input = input("Enter an SQL query: ")

# Predict using Logistic Regression
logistic_result = detect_sql_injection_logistic(user_input)
print(logistic_result)

# Predict using Random Forest
random_forest_result = detect_sql_injection_random_forest(user_input)
print(random_forest_result)

# Predict using Decision Tree
decision_tree_result = detect_sql_injection_decision_tree(user_input)
print(decision_tree_result)

#select * from users where id  =  1 +$+ or 1  =  1 -- 1
