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

# Keyword Module Imports
import yake
import re
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

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

        tokenized_text = self.tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)

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
            deduplim = 0.9,
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
            syn = wordnet.sysnets(var[0])
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
            syn = wordnet.sysnets(token[0])
            syn_words = [x.lemma_names() for x in syn]
            syn_words = [x for elem in syn_words for x in elem]
            syn_words.append(token[0])
            syn_words = list(set(syn_words))

            if len(set(syn_words).intersection(set(synonym_dict)))!=0:
                match += token[1]
    
        return match*10/total

        


    
    
t1 = "Data independence is the ability of a system to make changes to its data storage structures without affecting the way users access or manage the data. This means that users can modify and update databases without having to rewrite code, which saves time and resources. Data independence also reduces the risk of errors due to incorrect coding, as well as ensuring that any changes made are consistent across all systems."

t2 = "Data independence is the ability to modify the scheme without affecting the programs and the application to be rewritten. Data is separated from the programs, so that the changes made to the data will not affect the program execution and the application."

@app.route('/')
def home():
    return render_template('index.html', 
                           sim_score=similarity(t1,t2).similarity_score(),
                           ner_score=ner(t1,t2).ner_score())


app.run(debug=True)