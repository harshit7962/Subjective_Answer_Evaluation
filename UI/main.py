from flask import Flask, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

class similarity():
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2
    
    def showTexts(self):
        print(self.text1)
        print(self.text2)

t1 = "This is the first text to be passed to the similarity class"
t2 = "This is the second text to be passed to the similarity class"

@app.route('/')
def home():
    return render_template('index.html', text1=similarity(t1, t2).text1, text2=similarity(t1, t2).text2)


app.run(debug=True)