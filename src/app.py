from flask import Flask, request, render_template_string
from .model_deployment import load_model_and_vectorizer, predict_sentiment
import os

app = Flask(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(current_dir)

# Load model and vectorizer at app startup
model, vectorizer = load_model_and_vectorizer()

HTML = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Sentiment Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        form {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
            min-height: 100px;
        }
        input[type="submit"] {
            display: block;
            width: 100%;
            padding: 10px;
            background-color: #3498db;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        input[type="submit"]:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4fd;
            border-radius: 4px;
            text-align: center;
        }
        .positive {
            color: #27ae60;
            font-weight: bold;
        }
        .negative {
            color: #c0392b;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Sentiment Analysis</h1>
    <form method="post" action="/predict">
        <textarea name="text" placeholder="Enter your text here..."></textarea>
        <input type="submit" value="Analyze Sentiment">
    </form>
    {% if sentiment %}
    <div class="result">
        Sentiment: <span class="{{ 'positive' if sentiment == 'positive' else 'negative' }}">{{ sentiment }}</span>
    </div>
    {% endif %}
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get('text', '')
        try:
            sentiment = predict_sentiment(text, model, vectorizer)
        except Exception as e:
            sentiment = f"Error: {str(e)}"
        return render_template_string(HTML, sentiment=sentiment)
    return render_template_string(HTML)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)