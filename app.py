from flask import Flask, request, render_template
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import json
import plotly
import plotly.express as px

app = Flask(__name__)

# تحميل مجموعة بيانات Iris وتدريب النموذج
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]
        
        iris_types = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
        result = iris_types[prediction[0]]
        
        # إنشاء رسم بياني للاحتمالات
        fig = px.bar(x=iris_types, y=probabilities, labels={'x': 'نوع الزهرة', 'y': 'الاحتمال'})
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template('result.html', result=result, graph_json=graph_json)
    
    # إنشاء رسم بياني لتوزيع البيانات
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = pd.Categorical.from_codes(y, iris.target_names)
    fig = px.scatter(df, x="sepal length (cm)", y="sepal width (cm)", color="target", title="توزيع بيانات Iris")
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # حساب دقة النموذج
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return render_template('index.html', graph_json=graph_json, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)