import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('NB.pkl','rb'))
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










st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message", height = 200)

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Modify the prediction for SVC to use dense input
    vector_input_dense = vector_input.toarray()  # Convert sparse matrix to dense array

    # 3. Predict
    results = {
        "Naive Bayes": model.predict(vector_input)[0],
        "Logistic Regression": LR.predict(vector_input)[0],
        "AdaBoost": AdaBoost.predict(vector_input)[0],
        "Bagging Classifier": BgC.predict(vector_input)[0],
        "Decision Tree": DT.predict(vector_input)[0],
        "Extra Trees Classifier": ETC.predict(vector_input)[0],
        "Gradient Boosting": GBDT.predict(vector_input)[0],
        "K-Nearest Neighbors": KN.predict(vector_input)[0],
        "Random Forest": RF.predict(vector_input)[0],
        "Support Vector Classifier": SVC.predict(vector_input_dense)[0],
        "XGBoost": xgb.predict(vector_input)[0]
    }




    # 4. Display

    # if modelResult == 1:
    #     st.header("Naive Bayes - Spam")
    # else:
    #     st.header("Naive Bayes - Not Spam")
    #
    # if lrResult == 1:
    #     st.header("Logistic Regression - Spam")
    # else:
    #     st.header("LR - Not Spam")

    st.header("Prediction Results")
    for model_name, result in results.items():
        if result == 1:
            st.write(f"{model_name}: **Spam**")
        else:
            st.write(f"{model_name}: **Not Spam**")

