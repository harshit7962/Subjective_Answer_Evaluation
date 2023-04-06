# Flask Imports
from flask import Flask, render_template, request, session, redirect, url_for, flash
import pymongo
import json
import os
from flask_bcrypt import Bcrypt
from datetime import datetime
from bson import ObjectId

# Similarity Module Imports
import pandas as pd
import numpy as np
from numpy.linalg import norm
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import torch
from transformers import T5Tokenizer,T5ForConditionalGeneration
# NER Module Imports
import spacy
import spacy_transformers
import en_core_web_trf

# Keyword Module Imports
import yake
import re
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

with open("config.json", "r") as f:
    params = json.load(f)["params"]

app = Flask(__name__)

# Session Secret Key
app.secret_key = os.urandom(24)

# Encryption Initialization
bcrypt = Bcrypt(app)

# MongoDB Database Connection
client = pymongo.MongoClient("mongodb+srv://" + params["db_id"] +":" + params["db_pssd"] + "@cluster0.rq92bhz.mongodb.net/?retryWrites=true&w=majority")
db = client["data_db"]

# MongoDB Collection Names
student_details = db["student_details"]
faculty_details = db["faculty_details"]
test_details = db["test_details"]
questionnaire_details = db["questionnaire_details"]

# Similarity Module
class similarity():
    model_sbert = SentenceTransformer('stsb-mpnet-base-v2')
    model_summary = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')

    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2
    
    def cosine_similarity(self, text1_emb, text2_emb):
        cos = np.dot(text1_emb, text2_emb)/(norm(text1_emb)*norm(text2_emb))
        return cos
    
    def similarity_score(self):
        text1 = self.text1
        text2 = self.text2

        text2 += self.summarize(text2)

        mod_sent = sent_tokenize(text1)
        usr_sent = sent_tokenize(text2)

        mod_emb = []
        usr_emb = []

        for sent in mod_sent:
            sent_emb = self.model_sbert.encode(sent)
            mod_emb.append(sent_emb)
        
        for sent in usr_sent:
            sent_emb = self.model_sbert.encode(sent)
            usr_emb.append(sent_emb)
        
        n = len(mod_sent)
        m = len(usr_sent)

        sim_ans = 0

        for i in range(n):
            for j in range(m):
                if(self.cosine_similarity(mod_emb[i], usr_emb[j]) >= 0.7):
                    sim_ans += 1
                    break
        
        sim_ans /= n
        return sim_ans*10

    def summarize(self, text):
        preprocessed_text = text.strip().replace('\n', '')
        t5_input_text = 'summarize: ' + preprocessed_text

        tokenized_text = self.tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)

        summary_ids = self.model_summary.generate(tokenized_text, min_length=30, max_length=120)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

# NER Module
class ner():
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2

    def spacy_name(self, text):
        nlp = en_core_web_trf.load()

        doc = nlp(text)
        named_entities = []

        for entity in doc.ents:
            named_entities.append((entity.text, entity.label_))
        
        return named_entities
    
    def compute_score_spacy(self, modal_entity, user_entity):
        user_arr = []
        modal_arr = []

        for i in range(0, len(user_entity)):
            if(user_entity[i][1]!='CARDINAL'):
                user_arr.append(user_entity[i][0].lower())

        for i in range(0, len(modal_entity)):
            if(modal_entity[i][1]!='CARDINAL'):
                modal_arr.append(modal_entity[i][0].lower())

        user_entity = list(set(user_arr))
        modal_entity = list(set(modal_arr))

        n = len(modal_entity)
        m = len(user_entity)
        count = 0

        if(n == 0):
            return 10
        
        for i in range(n):
            if(modal_entity[i] in user_entity):
                count += 1
        
        count = count*10/(n)
        return count
    
    def ner_score(self):
        text1 = self.text1
        text2 = self.text2

        modal_entity = self.spacy_name(text1)
        user_entity = self.spacy_name(text2)

        score = self.compute_score_spacy(modal_entity, user_entity)

        return score

# Keyword Module
class keyword():
    R_patterns = [
        (r'won\'t', 'will not'),
        (r'can\'t', 'can not'),
        (r'(\w+)\'m', '\g<1> am'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)\'d like to', '\g<1> would like to'),
    ]

    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in self.R_patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        
        return s
    
    def text_lower(self, x):
        return x.lower()
    
    def corrector(self, x):
        return x.replace('. ', '.')
    
    def replace_punct(self, x):
        pattern = "[^\w\d\+\*\-\\\=\s]"
        repl = " "

        return re.sub(pattern, repl, x)
    
    def remove_extra(self, text):
        return " ".join(text.strip().split())
    
    def keywordExtractor(self, text):
        custom_kw_extractor = yake.KeywordExtractor(
            lan = "en",
            n = 1,
            dedupLim = 0.9,
            dedupFunc= 'seqm',
            windowsSize= 2,
            top = 101,
            features = None
            )
        
        return custom_kw_extractor.extract_keywords(text)
    
    def scoring_unit(self, keywords_text1, keywords_text2):
        match = 0
        total = 0
        synonym_dict = []

        for token in keywords_text1:
            total += token[1]
        
        if(total == 0):
            return 10
        
        for var in keywords_text2:
            syn = wordnet.synsets(var[0])
            syn_words = [x.lemma_names() for x in syn]
            syn_words = [x for elem in syn_words for x in elem]
            syn_words.append(var[0])
            syn_words = list(set(syn_words))

            temp = []

            wt = word_tokenize(var[0])
            pos = pos_tag(wt)[0][1]

            for i in range(0, len(syn_words)):
                checker_wt = word_tokenize(syn_words[i])
                checker_pos = pos_tag(wt)[0][1]
                if(pos == checker_pos):
                    temp.append(syn_words[i])
            
            synonym_dict = synonym_dict + temp

        for token in keywords_text1:
            syn = wordnet.synsets(token[0])
            syn_words = [x.lemma_names() for x in syn]
            syn_words = [x for elem in syn_words for x in elem]
            syn_words.append(token[0])
            syn_words = list(set(syn_words))

            if len(set(syn_words).intersection(set(synonym_dict)))!=0:
                match += token[1]
    
        return match*10/total
    
    def keyword_score(self):
        text1 = self.text1
        text2 = self.text2

        text1 = self.replace(text1)
        text2 = self.replace(text2)
        text1 = self.text_lower(text1)
        text2 = self.text_lower(text2)
        text1 = self.corrector(text1)
        text2 = self.corrector(text2)
        text1 = self.replace_punct(text1)
        text2 = self.replace_punct(text2)
        text1 = self.remove_extra(text1)
        text2 = self.remove_extra(text2)

        yake_keyword_model = self.keywordExtractor(text1)
        yake_keyword_user = self.keywordExtractor(text2)

        return self.scoring_unit(yake_keyword_model, yake_keyword_user)

    
t1 = '''
BCNF stands for Boyce-Codd Normal Form, which is a normal form used in database normalization. BCNF is based on the concept of functional dependencies between attributes in a relation.A relation is said to be in BCNF if and only if every determinant (i.e., every candidate key) of the relation is a superkey of the relation. In simpler terms, a relation is in BCNF if every non-trivial functional dependency in the relation has a determinant that is a candidate key. The process of decomposing a relation into smaller, normalized relations is called normalization. Normalization is important for database design because it helps to eliminate redundancy, minimize data inconsistencies, and ensure data integrity. However, it is important to note that normalization is not always the best solution, and it may not always be possible to achieve BCNF.
'''

t2 = '''
Boyce–Codd Normal Form (BCNF) is based on functional dependencies that take into account all candidate keys in a relation; however, BCNF also has additional constraints compared with the general definition of 3NF. A relation is in BCNF if, X is superkey for every functional dependency (FD) X?Y in given relation. In other words, A relation is in BCNF, if and only if, every determinant is a Form (BCNF) candidate key.
'''

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.route('/')
def index():
    return render_template('index.ejs')

@app.route('/learn')
def learn():
    return render_template('learn.ejs')

@app.route('/team')
def team():
    return render_template('team.ejs')

'''
Need to implement contact.ejs and link to database
'''
@app.route('/contact')
def contact():
    return render_template('contact.ejs')

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if "email" in session:
        session.pop("email", None)

    message = ""   
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        college = request.form.get('college')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        author = request.form.get('author')

        # Data Verification
        if(db.student_details.find_one({"email": email}) or db.faculty_details.find_one({"email": email})):
            message="User is already registered"

        elif password != confirm_password:
            message = "Passwords do not match"

        else:
            # Generating Encrypted Password along with Creation Time
            password = bcrypt.generate_password_hash(password).decode('utf-8')
            created = datetime.now()
        
            # Inserting the record to database
            try:
                if(author=='Student'):
                    db.student_details.insert_one({"name": name, "email": email, "college":college, "password": password, "author": author, "created": created})
                else:
                    db.faculty_details.insert_one({"name": name, "email": email, "college":college, "password": password, "author": author, "created": created})
                message = "User Registered Successfully"
                return redirect(url_for("login"))

            except Exception as ex:
                message = f"{ex}"
    
    return render_template("signup.ejs", message=message)

@app.route('/login', methods=['POST', 'GET'])
def login():
    message = None
    if "email" in session:
        session.pop("email", None)

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = db.student_details.find_one({"email": email})

        if user:
            if bcrypt.check_password_hash(user["password"], password):
                session["email"] = email
                return redirect(url_for("logged_in"))
            else:
                message = "Incorrect Password"
                
        else:
            message = "User not found"
            
    return render_template("signin.html", message=message)

@app.route('/logged_in')
def logged_in():
    if "email" in session:
        user = db.student_details.find_one({"email": session["email"]})

        tests = db.test_details.find()

        '''
        TODO: replace the test_home.html with student home page containing all the tests available
        '''
        return render_template("test_home.html", user = user, tests = tests)
    return render_template("signin.html", message="You are not Logged In")

@app.route('/logout')
def logout():
    if "email" in session:
        session.pop("email", None)
    
    return redirect(url_for("index"))

@app.route("/test/<string:test_slug>/<int:question_number>", methods = ["GET", "POST"])
def test_route(test_slug, question_number=1):
    if "email" in session:
        already_present = False
        test_number = db.test_details.find_one({"_id": ObjectId(test_slug)})["test_number"]

        # Check if current question is already attempted by the candidate
        current_size = db.answer_collection.count_documents({"email": session["email"], "test_number": test_number, "question_number": question_number})
        if(current_size >0):
            already_present = True

        if request.method == "POST":
            answer = request.form.get("answer")

            if(already_present):
                db.answer_collection.update_one({"email": session["email"], "test_number": test_number, "question_number": question_number}, {"$set": {"answer": answer}})
            else:
                db.answer_collection.insert_one({"email": session["email"], "test_number": test_number, "question_number": question_number, "answer": answer})
        
        questions = list(db.questionnaire_details.find({"test_number": test_number, "question_number": question_number}))
        total_questions = db.questionnaire_details.count_documents({"test_number": test_number})

        return render_template("show_question.html",
                               question = questions, 
                               test_slug = test_slug, 
                               total_questions = total_questions
                            )
    return render_template("signin.html", message="You are not Logged In")

app.run(debug=True)