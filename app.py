import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("d_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def build_features(q1, q2):
    q1, q2 = q1.strip(), q2.strip()
    w1, w2 = set(q1.lower().split()), set(q2.lower().split())
    common = len(w1 & w2)
    total = len(w1) + len(w2)
    share = round(common / total, 2) if total > 0 else 0
    basic = np.array([[len(q1), len(q2), len(q1.split()), len(q2.split()),
                       common, total, share]])
    v1 = tfidf.transform([q1]).toarray()
    v2 = tfidf.transform([q2]).toarray()
    return np.hstack([basic, v1, v2])

st.header("DUPLICATE QUE PAIR FINDER")
q1 = st.text_input("ENTER THE QUESTION1")
q2 = st.text_input("ENTER THE QUESTION2")

if st.button("FIND"):
    x_new = build_features(q1, q2)
    if model.predict(x_new)[0] == 1:
        st.success("Duplicate")
    else:
        st.error("Not Duplicate")