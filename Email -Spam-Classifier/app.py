from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('svm_spam.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        message = request.form['message']
        prediction = model.predict([message])[0]
        result = 'Spam' if prediction == 1 else 'Ham'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
