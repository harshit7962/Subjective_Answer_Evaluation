# Flask Imports
from flask import Flask, render_template

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

app = Flask(__name__)

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

        tokenized_text = self.tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512).to(self.device)

        summary_ids = self.model_summary.generate(tokenized_text, min_length=30, max_length=120)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

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
        n = len(modal_entity)
        m = len(user_entity)
        count = 0

        if(n == 0):
            return 10
        
        for i in range(m):
            if(user_entity[i] in modal_entity):
                count += 1
                continue
        
        count = count*10/(n)
        return min(count, 10)
    
    def ner_score(self):
        text1 = self.text1
        text2 = self.text2

        modal_entity = self.spacy_name(text1)
        user_entity = self.spacy_name(text2)

        score = self.compute_score_spacy(modal_entity, user_entity)

        return score





t1 = "Data independence is the ability of a system to make changes to its data storage structures without affecting the way users access or manage the data. This means that users can modify and update databases without having to rewrite code, which saves time and resources. Data independence also reduces the risk of errors due to incorrect coding, as well as ensuring that any changes made are consistent across all systems."

t2 = "Data independence is the ability to modify the scheme without affecting the programs and the application to be rewritten. Data is separated from the programs, so that the changes made to the data will not affect the program execution and the application."

@app.route('/')
def home():
    return render_template('index.html', 
                           sim_score=similarity(t1,t2).similarity_score(),
                           ner_score=ner(t1,t2).ner_score())


app.run(debug=True)