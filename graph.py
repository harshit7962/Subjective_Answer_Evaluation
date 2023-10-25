import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Scores.csv")

x = np.arange(1, 21)
y1 = df.iloc[:20, 9] # TODO - replace 9 with column number of Yake scores
y2 = df.iloc[:20, 8] # TODO - replace 9 with column number of KeyBERT scores
y3 = df.iloc[:20,  -1] # TODO - replace 9 with column number of Actual scores

plt.stackplot(
    x, 
    y1, 
    y2, 
    y3,
    labels=['Yake Score', 'KeyBert Score', 'Actual Score']    
)

plt.legend(loc="upper left")
plt.xlabel("Stacked Scores")
plt.ylabel("Distribution of Indexes")
plt.title("Keyword Scores and affect on Actual Score")

plt.show()


# similarity distribution graph
x = np.arange(1, 31)

sbert_score = df['SBERT Score'].values # TODO replace the name accordingly
simcse_score = df['Sim_CSE Score'].values # TODO replace the name accordingly
hf_score = df['HF_Score'].values # TODO replace the name accordingly

plt.plot(x, sbert_score[0: 30], label="SBERT")
plt.plot(x, simcse_score[0: 30], label = "SimCSE")
plt.plot(x, hf_score[0: 30], label="Hugging Face", linestyle="--")
plt.xlabel("Index Spanning")
plt.ylabel("Similarity Scores")
plt.title("Distribution of Similarity Scores")
plt.legend()
plt.show()

# ner scoring graph
cambert_freq = [0]*11
spacy_freq = [0]*11
nltk_freq = [0]*11

marks = np.arange(0, 11)

cambert_score = df["Score_camembert"]
nltk_score = df["Score_NLTK"]
spacy_score = df["Score_spacy"]

for i in cambert_score:
    cambert_freq[int(i)] += 1

for i in nltk_score:
    nltk_freq[int(i)] += 1

for i in spacy_score:
    spacy_freq[int(i)] += 1

barWidth = 0.5

br1 = np.arange(len(cambert_freq))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]


plt.bar(br1, nltk_freq, color='coral', label='NLTK Count')
plt.bar(br2, spacy_freq, color='cyan', label='Spacy Count')
plt.bar(br3, cambert_freq, color='pink', label='Camembert Count')

plt.xlabel("Score")
plt.ylabel("Frequency of Occurance")
plt.legend()
plt.title("Frequency of Occurance of NER Scores")
plt.show()