import re
import sys
import getopt
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from utils.cosine_similarity import cosine_similarity as cs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

argumentList = sys.argv[1:]
options = "o:f:"
long_options = ["option", "filename"]
arguments, values = getopt.getopt(argumentList, options, long_options)

filename, option = "input.txt", "IT"

for argument, value in arguments:
    if argument in ["-f", "--filename"]:
        filename = value
    elif argument in ["-o", "--option"]:
        option = value

if option == "IT":
    path = "../Cleaned_Datasets/JobsIT_Dataset.csv"
elif option == "NON-IT":
    path = "../Cleaned_Datasets/JobsNonIT_Dataset.csv"
else:
    print("Invalid value for argument option")
    sys.exit(1)

all_stopwords = stopwords.words('english')

tf = TfidfVectorizer()
jobs = pd.read_csv(path)
with open(f"../predictor_files/{filename}",encoding="utf8") as f:
    prediction_text = f.readlines()

prediction_text = ' '.join(prediction_text)
prediction_text = re.sub('[^a-zA-Z]', ' ', prediction_text)
prediction_text = prediction_text.lower()
prediction_text = prediction_text.split()
ps = PorterStemmer()
prediction_text = [ps.stem(word) for word in prediction_text if not word in set(all_stopwords)]
prediction_text = ' '.join(prediction_text)

tfidf_jobs = tf.fit_transform(jobs["Description"])
tfidf_prediction_text = tf.transform([prediction_text])

similarity_measure = cosine_similarity(tfidf_jobs, tfidf_prediction_text)
labels = jobs["Query"].unique()
similarity_scores = {label: {"sum": 0, "count": 0} for label in labels} 
for i in range(len(similarity_measure)):
    similarity_scores[jobs["Query"][i]]["sum"] += similarity_measure[i][0]
    similarity_scores[jobs["Query"][i]]["count"] += 1

predictions = []
for label in similarity_scores:
    avg = similarity_scores[label]["sum"]/similarity_scores[label]["count"]
    predictions.append([avg, label])
predictions.sort(key = lambda key: -key[0])
output_text = f"Top3 predictions for you in {option} Industry:\n" + '\n'.join(x[1] for x in predictions[:3])
print(output_text)