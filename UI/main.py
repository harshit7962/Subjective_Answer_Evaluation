# Flask Imports
from flask import Flask, render_template, request, session, redirect, url_for, flash
import pymongo
import json
import os
from flask_bcrypt import Bcrypt
from datetime import datetime
from bson import ObjectId
from keras.models import load_model

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
            return 10.0
        
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
    

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.route('/')
def index():
    if "email" in session:
        session.pop("email", None)

    if "message" in session:
        session.pop("message", None)
    
    if "test_number" in session:
        session.pop("test_number", None)

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

    if "message" in session:
        session.pop("message", None)
    
    if "test_number" in session:
        session.pop("test_number", None)

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

    if "message" in session:
        session.pop("message", None)
    
    if "test_number" in session:
        session.pop("test_number", None)

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = db.student_details.find_one({"email": email})

        if user:
            if bcrypt.check_password_hash(user["password"], password):
                session["email"] = email
                return redirect(url_for("exams"))
            else:
                message = "Incorrect Password"
                
        else:
            message = "User not found"
            
    return render_template("signin.html", message=message)

@app.route('/exams')
def exams():
    message = ""
    if "email" in session:
        user = db.student_details.find_one({"email": session["email"]})

        tests = db.test_details.find()
        questions = db.questionnaire_details.find()

        if "message" in session:
            message = session["message"]
            session.pop("message", None)

        return render_template("exam.html",
                                user = user,
                                tests = tests,
                                questions = questions,
                                message = message
                            )
    return render_template("signin.html", message="You are not Logged In")

@app.route('/logout')
def logout():
    if "email" in session:
        session.pop("email", None)

    if "faculty_email" in session:
        session.pop("faculty_email", None)
    
    return redirect(url_for("index"))

@app.route("/test/<string:test_slug>/<int:question_number>", methods = ["GET", "POST"])
def test_route(test_slug, question_number=1):
    if "email" in session:
        already_present = False
        test = db.test_details.find_one({"_id": ObjectId(test_slug)})
        test_number = test["test_number"]
        test_name = test["test_name"]

        session["test_number"] = test_number

        # Check if current question is already attempted by the candidate
        current_size = db.answer_collection.count_documents({"email": session["email"], 
                                                             "test_number": test_number, 
                                                             "question_number": question_number
                                                            })
        if(current_size >0):
            already_present = True

        if request.method == "POST":
            answer = request.form.get("answer")

            if(already_present):
                db.answer_collection.update_one({"email": session["email"], "test_number": test_number, "question_number": question_number}, {"$set": {"answer": answer}})
            else:
                db.answer_collection.insert_one({"email": session["email"], "test_number": test_number, "question_number": question_number, "answer": answer})
        
        total_questions = db.questionnaire_details.count_documents({"test_number": test_number})

        print(request.form.get('submit'))
              
        if request.form.get("submit") == "non_final":
            questions = list(db.questionnaire_details.find({"test_number": test_number, "question_number" : question_number+1}))
        else:
            questions = list(db.questionnaire_details.find({"test_number": test_number, "question_number" : question_number}))

        
        return render_template("questions.html",
                               question = questions, 
                               test_slug = test_slug, 
                               test_name = test_name, 
                               total_questions = total_questions
                            )
        

    return render_template("signin.html", message="You are not Logged In")

@app.route("/submit")
def submit():
    if "email" in session:
        message = "You have submitted your test successfully."
        session["message"] = message
        '''
        answer_collection se answer fetch karne hai, is particular test ke
        fir model answer fetch karne hai from questionnaire details se
        add similarity keyword ner
        and fir model se score fetch karna hai
        fir update bhi karna hai answer collection
        '''
        return redirect(url_for("exams"))

@app.route("/results")
def results():
    if "email" in session:
        tests = db.test_details.find()
        attempted_tests = []

        for test in tests:
            test_number = test["test_number"]

            attempted = db.answer_collection.find_one({"email": session["email"], "test_number": test_number})
            
            if attempted:
                test = db.test_details.find_one({"test_number": test_number})
                attempted_tests.append(test)
        
        user = db.student_details.find_one({"email": session["email"]})

        return render_template("results.html", 
                               tests = attempted_tests, 
                               user = user
                            )
    return render_template("signin.html", message="You are not Logged In")

@app.route("/result/<string:test_slug>/<int:question_number>")
def show_result(test_slug, question_number=1):
    if "email" in session:
        test = db.test_details.find_one({"_id": ObjectId(test_slug)})
        test_number = test["test_number"]
        test_name = test["test_name"]
        model_answer_base = db.questionnaire_details.find_one({"test_number": test_number, "question_number": question_number})
        user_answer_base = db.answer_collection.find_one({"test_number": test_number, 
                                                          "question_number": question_number, 
                                                          "email": session["email"]
                                                        })
        
        total_questions = db.questionnaire_details.count_documents({"test_number": test_number})

        return render_template("exam_results.html",
                               modal = model_answer_base,
                               user = user_answer_base,
                               total_questions = total_questions,
                               test_slug = test_slug,
                               test_name = test_name
                            )
    return render_template("signin.html", message="You are not Logged In")



# Computation of result
@app.route("/computation")
def computation():
    if "email" in session:
        # We will do our computation here
        if "test_number" in session:
            test_number = session["test_number"]
            session.pop("test_number", None)

            # Count the number of quetions in attempted test
            total_questions = db.questionnaire_details.count_documents({"test_number": test_number})

            # Load our saved model
            model = load_model('model_ann.h5')

            # For each question we get user answer and modal answer and print them
            for i in range(total_questions):
                # user answer
                user_answer = db.answer_collection.find_one({"test_number": test_number,
                                                             "question_number": i+1,
                                                             "email": session["email"]
                                                            })
                
                if user_answer:
                    user_answer = user_answer["answer"]
                else:
                    user_answer = ""
                
                # modal answer
                modal_answer = db.questionnaire_details.find_one({"test_number": test_number, "question_number": i+1})["modal_answer"]

                # Need to implement module wise computation here
                similarity_score = similarity(modal_answer, user_answer).similarity_score()
                ner_score = ner(modal_answer, user_answer).ner_score()
                keyword_score = keyword(modal_answer, user_answer).keyword_score()

                # Rounding of the individaul scores
                similarity_score = round(similarity_score, 1)
                ner_score = round(ner_score, 1)
                keyword_score = round(keyword_score, 1)

                # Here we need to handle edge cases, in case the scores are greater than 10 or less than 0
                if similarity_score < 0:
                    similarity_score = 0.0
                elif similarity_score > 10:
                    similarity_score = 10.0

                if ner_score < 0:
                    ner_score = 0.0
                elif ner_score > 10:
                    ner_score = 10.0
                
                if keyword_score < 0:
                    keyword_score = 0.0
                elif keyword_score > 10:
                    keyword_score = 10.0

                # Final Scoring
                input = [[keyword_score, similarity_score, ner_score]]
                final_score = model.predict(input)[0][0]
                final_score = final_score.item()
                
                final_score = round(final_score, 1)

                if final_score < 0:
                    final_score = 0.0
                elif final_score > 10:
                    final_score = 10.0

                # Updating the scores to db
                db.answer_collection.update_one({
                    "test_number": test_number,
                    "question_number": i+1,
                    "email": session["email"]
                }, {
                    "$set": {
                        "similarity_score": similarity_score,
                        "ner_score": ner_score,
                        "keyword_score": keyword_score,
                        "final_score": final_score
                    }
                })

            print("\n\n\n\nTest Evaluation Completed\n\n\n\n")
        
    return render_template("signin.html", message="You are logged in")


# ----------------------------------------------------------------
# Admin Routes
# ----------------------------------------------------------------

@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    message = None
    if "faculty-email" in session:
        session.pop("faculty-email", None)

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        faculty = db.faculty_details.find_one({"email": email})
        

        if faculty:
            if bcrypt.check_password_hash(faculty["password"], password):
                session["faculty_email"] = email
                return redirect(url_for("admin_home"))
            else:
                message = "Incorrect Password"
        else:
            message = "Admin not found"

    return render_template("admin-signin.html", message=message)

@app.route("/admin_home")
def admin_home():
    if "faculty_email" in session:
        message = ""
        if "test_message" in session:
            message = session["test_message"]
            session.pop("test_message", None)

        faculty = db.faculty_details.find_one({"email": session["faculty_email"]})
        tests = db.test_details.find()
        questions = db.questionnaire_details.find()

        return render_template("admin-home.html",
                               faculty = faculty,
                               tests = tests,
                               questions = questions,
                               message = message
                            )
    
    return render_template("signin.html", message = "You are not Logged in")

# Need to add edit exam route
@app.route("/set-test/<string:test_slug>/<int:question_number>", methods = ["GET", "POST"])
def set_test(test_slug, question_number):
    if "faculty_email" in session:
        message = ""
        if "addition_question" in session:
            message = session["addition_question"]
            session.pop("addition_question", None)
        
        if "empty_answer" in session:
            message = session["empty_answer"]
            session.pop("empty_answer", None)

        test = db.test_details.find_one({"_id": ObjectId(test_slug)})
        test_number = test["test_number"]

        question = db.questionnaire_details.find_one({"test_number": test_number, "question_number": question_number})
        total_questions = db.questionnaire_details.count_documents({"test_number": test_number})
        
        if request.method == "POST":
            if question_number == total_questions+1:
                question = request.form.get("question")
                modal_answer = request.form.get("answer")

                if question == "" or modal_answer == "":
                    session["addition_question"] = "Cannot add empty fields"

                    return redirect(url_for("set_test", test_slug=test_slug, question_number = total_questions))

                db.questionnaire_details.insert_one({
                    "test_number": test_number,
                    "question": question,
                    "modal_answer": modal_answer,
                    "question_number": question_number
                })

                return redirect(url_for("set_test", test_slug=test_slug, question_number=question_number))

            answer = request.form.get("answer")

            if answer == "":
                session["empty_answer"] = "Cannot have Empty Modal Answer"
                return redirect(url_for("set_test", test_slug=test_slug, question_number=question_number))

            db.questionnaire_details.update_one({
                "test_number": test_number,
                "question_number": question["question_number"]
            }, {
                "$set" : {
                    "modal_answer": answer
                }
            })

        return render_template("edit-questions.html",
                               test = test,
                               question = question,
                               test_slug = test_slug,
                               total_questions = total_questions,
                                message = message
                            )

    return render_template("signin.html", message = "You are not Logged in")

@app.route('/edit-question/<string:test_slug>/<int:question_number>')
def edit_question(test_slug, question_number):
    if "faculty_email" in session:        
        test = db.test_details.find_one({"_id": ObjectId(test_slug)})

        test_number = test["test_number"]
        question = db.questionnaire_details.find_one({"test_number": test_number, "question_number": question_number})

        return render_template("change-answer.html", 
                               question = question,
                               test = test
                            )
    
    return render_template("signin.html", message = "You are not Logged in.")

#need to implement one route for adding questions to already created test
@app.route('/add-question/<string:test_slug>/<int:question_number>')
def add_question(test_slug, question_number):
    if "faculty_email" in session:
        test = db.test_details.find_one({"_id": ObjectId(test_slug)})
        
        test_number = test["test_number"]
        question = db.questionnaire_details.find_one({"test_number": test_number, "question_number": question_number})

        total_questions = db.questionnaire_details.count_documents({"test_number": test_number})

        return render_template("add-question.html",
                                question = question,
                                test = test,
                                total_questions = total_questions
                            )
    
    return render_template("signin.html", message = "You are not Logged in")

# Need another route to create a new test from scratch
@app.route("/add-test", methods = ["GET", "POST"])
def add_test():
    if "faculty_email" in session:
        if request.method == "POST":
            total_tests = db.test_details.count_documents({})
            test_name = request.form.get("test_name")

            db.test_details.insert_one({"test_name": test_name, "test_number": total_tests+1, "create_on": datetime.now() })

            test = db.test_details.find_one({"test_number": total_tests+1})

            return redirect(url_for(
                "add_new_question", 
                test_slug = test["_id"],
                question_number = 1
                ))

        return render_template("get-name.html")
    
    return render_template("signin.html", message = "You are not Logged In")

@app.route("/add-new-question/<string:test_slug>/<int:question_number>", methods = ["POST", "GET"])
def add_new_question(test_slug, question_number):
    if "faculty_email" in session:
        message = ""
        if "question_message" in session:
            message = session["question_message"]
            session.pop("question_message", None)
        
        test = db.test_details.find_one({"_id": ObjectId(test_slug)})
        test_name = test["test_name"]
        test_number = test["test_number"]

        total_questions = db.questionnaire_details.count_documents({"test_number": test_number})

        if request.method == "POST":                
            question = request.form.get("question")
            modal_answer = request.form.get("modal_answer")

            if question=="" or modal_answer == "":
                if question == "":
                    question_message = "Cannot Add Empty Question"
                else:
                    question_message = "Cannot Add Empty Modal Answer"
                session["question_message"] = question_message
                
                return redirect(url_for(
                    "add_new_question",
                    test_slug = test_slug,
                    question_number = question_number
                ))

            if db.questionnaire_details.find_one({"test_number": test_number, "question_number": question_number}):
                # Update the modal answer and question
                db.questionnaire_details.update_one(
                    {'test_number': test_number,
                        'question_number': question_number
                    }, {
                    "$set": {
                        "question": question,
                        "modal_answer": modal_answer
                        }
                    }
                )
            else:
                # Insert the current document
                db.questionnaire_details.insert_one({
                    "test_number": test_number,
                    "question_number": question_number,
                    "question": question,
                    "modal_answer": modal_answer
                })

            if request.form.get("submit-btn") == "next-question":
                question_message = "Question Saved Successfully!"
                session["question_message"] = question_message
                return redirect(url_for(
                    "add_new_question",
                    test_slug = test_slug,
                    question_number = question_number+1
                ))
            else:
                test_message = "Test Saved Successfully!"
                session["test_message"] = test_message
                return redirect(url_for(
                    "admin_home"
                ))
        
        return render_template(
            "new-test.html",
            test = test,
            test_name = test_name,
            question_number = question_number,
            total_questions = total_questions,
            message = message
        )

    return render_template("signin.html", message = "You are not Logged In")

@app.route("/check-validity")
def check_validity():
    total_tests = db.test_details.count_documents({})
    
    test_message = "Test Not Added"
    session["test_message"] = test_message
    db.test_details.delete_one({"test_number": total_tests})

    return redirect(url_for("admin_home"))

@app.route("/performance")
def performance():
    if "faculty_email" in session:
        faculty = db.faculty_details.find_one({"email": session["faculty_email"]})
        students = db.student_details.find()

        return render_template("performance.html", students = students, faculty = faculty)

    return render_template("signin.html", message = "You are not Logged In")

@app.route("/student/<string:student_slug>")
def student_viewing(student_slug):
    if "faculty_email" in session:
        student = db.student_details.find_one({"_id": ObjectId(student_slug)})
        tests = db.test_details.find()
        faculty = db.faculty_details.find_one({"email": session["faculty_email"]})

        attempted_tests = []

        for test in tests:
            test_number = test["test_number"]

            attempted = db.answer_collection.find_one({
                "email": student["email"],
                "test_number": test_number
            })

            if attempted:
                test = db.test_details.find_one({"test_number": test_number})
                attempted_tests.append(test)
        
        return render_template(
            "student-result.html",
            tests = attempted_tests,
            user = student,
            faculty = faculty
        )

    return render_template("signin.html", message = "You are not Logged In")

@app.route("/student_result/<string:student_slug>/<string:test_slug>/<int:question_number>")
def student_result(student_slug, test_slug, question_number):
    if "faculty_email" in session:
        test = db.test_details.find_one({"_id": ObjectId(test_slug)})
        test_number = test["test_number"]
        test_name = test["test_name"]
        student = db.student_details.find_one({"_id": ObjectId(student_slug)})

        modal_answer_base = db.questionnaire_details.find_one({
            "test_number": test_number,
            "question_number": question_number
        })

        user_answer_base = db.answer_collection.find_one({
            "test_number": test_number,
            "question_number": question_number,
            "email": student["email"]
        })

        total_questions = db.questionnaire_details.count_documents({"test_number": test_number})

        return render_template(
            "student_exam_result.html",
            modal = modal_answer_base,
            user = user_answer_base,
            total_questions = total_questions,
            test_slug = test_slug,
            test_name = test_name,
            student = student
        )

    return render_template("signin.html", message = "You are not Logged In")
app.run(debug=True)
