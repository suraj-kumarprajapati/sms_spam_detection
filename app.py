from flask import Flask, request, jsonify
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('NB.pkl', 'rb'))
LR = pickle.load(open('LR.pkl', 'rb'))
AdaBoost = pickle.load(open('AdaBoost.pkl', 'rb'))
BgC = pickle.load(open('BgC.pkl', 'rb'))
DT = pickle.load(open('DT.pkl', 'rb'))
ETC = pickle.load(open('ETC.pkl', 'rb'))
GBDT = pickle.load(open('GBDT.pkl', 'rb'))
KN = pickle.load(open('KN.pkl', 'rb'))
RF = pickle.load(open('RF.pkl', 'rb'))
SVC = pickle.load(open('SVC.pkl', 'rb'))
xgb = pickle.load(open('xgb.pkl', 'rb'))

#  Homepage route
@app.route('/')
def home():
    return '''
    <h2>📧 Spam Classifier API</h2>
    <p>Send a POST request to <code>/predict</code> with a JSON like:</p>
    <pre>
    {
        "message": "Your message here"
    }
    </pre>
    '''

#  Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    input_sms = data['message']

    # Preprocess
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    vector_input_dense = vector_input.toarray()

    # Predict
    results = {
        "Naive Bayes": int(model.predict(vector_input)[0]),
        "Logistic Regression": int(LR.predict(vector_input)[0]),
        "AdaBoost": int(AdaBoost.predict(vector_input)[0]),
        "Bagging Classifier": int(BgC.predict(vector_input)[0]),
        "Decision Tree": int(DT.predict(vector_input)[0]),
        "Extra Trees Classifier": int(ETC.predict(vector_input)[0]),
        "Gradient Boosting": int(GBDT.predict(vector_input)[0]),
        "K-Nearest Neighbors": int(KN.predict(vector_input)[0]),
        "Random Forest": int(RF.predict(vector_input)[0]),
        "Support Vector Classifier": int(SVC.predict(vector_input_dense)[0]),
        "XGBoost": int(xgb.predict(vector_input)[0])
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
