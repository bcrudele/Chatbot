import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import neattext.functions as nfx
###
from textblob import TextBlob
###
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import words
###
from collections import Counter
###
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
### 

nltk.download('punkt')
nltk.download('words')

sample_text = ["Good morning what a wonderful day"]
df = pd.read_csv("train_data/emotion_dataset.csv")
#print(df.head())
#print(df.shape)
#print(df.isnull().sum())
#df['Emotion'].value_counts().plot(kind='bar')
#plt.show()

# Plot Count Plot
#plt.figure(figsize=(5,3))
#sns.countplot(x='Emotion',data=df)
#plt.show()

# Sentiment Analysis
# Keyword Extraction for each emotion

# Sentiment Analysis ##############################
# Returns Positive, Negative, or Neutral for a word
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        result = "Positive"
    elif sentiment < 0:
        result = "Negative"
    else:
        result = "Neutral"
    return result

df['Sentiment'] = df['Text'].apply(get_sentiment)
# compare emotion v sentiment
df.groupby(['Emotion', 'Sentiment']).size()

# Plot Word Counts ##############################
#sns.catplot(x='Emotion', hue='Sentiment', data=df, kind='count')
#plt.show()

# Download NLTK resources
#nltk.download('averaged_perceptron_tagger')
#english_words = set(words.words())

# Text clean ######################
# remove noise
    # stopwords, special chars, punctuation
df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_punctuations)
# Additional step to remove non-English words and small single characters using NLTK
#df['Clean_Text'] = df['Clean_Text'].apply(lambda x: ' '.join(word for word in word_tokenize(x) if len(word) > 1 and word.lower() in english_words))
# Additional step to remove proper nouns and small single characters using NLTK
#df['Clean_Text'] = df['Clean_Text'].apply(lambda x: ' '.join(word for word, pos in pos_tag(word_tokenize(x)) if len(word) > 1 and (word != "feel" and word != "feeling")and word.lower() in english_words and pos != 'NNP'))
## UNCOMMENT ABOVE OR BELOW FOR MODEL, ABOVE LINE CAUSES SLOWDOWN
#df['Clean_Text'] = df['Clean_Text'].apply(lambda x: ' '.join(word for word, pos in pos_tag(word_tokenize(x)) if len(word) > 1 and word.lower() in english_words and pos != 'NNP'))

# Drop rows where 'Clean_Text' is empty
#df = df[df['Clean_Text'].str.strip() != '']

# Keyword Extraction
# find most common words per emotion

def extract_keywords(text, num=50):
    tokens = [tok for tok in text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)

emotion_list = df['Emotion'].unique().tolist()
print(emotion_list)

happy_list = df[df['Emotion'] == 'happy']['Clean_Text'].tolist()
happy_docx = ' '.join(happy_list)

anger_list = df[df['Emotion'] == 'anger']['Clean_Text'].tolist()
anger_docx = ' '.join(anger_list)

sadness_list = df[df['Emotion'] == 'sadness']['Clean_Text'].tolist()
sadness_docx = ' '.join(sadness_list)

love_list = df[df['Emotion'] == 'love']['Clean_Text'].tolist()
love_docx = ' '.join(love_list)

surprise_list = df[df['Emotion'] == 'surprise']['Clean_Text'].tolist()
surprise_docx = ' '.join(surprise_list)

fear_list = df[df['Emotion'] == 'fear']['Clean_Text'].tolist()
fear_docx = ' '.join(fear_list)

# Extracting Keywords
keywords_happy = extract_keywords(happy_docx)
keywords_anger = extract_keywords(anger_docx)
keywords_sadness = extract_keywords(sadness_docx)
keywords_love = extract_keywords(love_docx)
keywords_surprise = extract_keywords(surprise_docx)
keywords_fear = extract_keywords(fear_docx)
#print(keywords_happy)

# Word Bar Graph for Common Words ##########################
def plot_most_common_words(mydict):
    df_01 = pd.DataFrame(mydict.items(), columns=['token','count'])
    sns.barplot(x='token', y='count', data=df_01)
    plt.show()

#plot_most_common_words(keywords_happy)
#plot_most_common_words(keywords_anger)
#plot_most_common_words(keywords_sadness)
#plot_most_common_words(keywords_love)
#plot_most_common_words(keywords_surprise)
#plot_most_common_words(keywords_fear)

## Word Cloud Image ###########################
from wordcloud import WordCloud

def plot_wordcloud(docx):
    mywordcloud = WordCloud().generate(docx)
    plt.figure(figsize=(20,10))
    plt.imshow(mywordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()

#plot_wordcloud(surprise_docx)

## ML ####################################
# NB
# Logistic Reg
# KNN

# Building
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']

# Vectorize
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)  # Fit and transform the text data

# Get features by name
cv.get_feature_names_out()

# Convert to dense array
X.toarray()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,ylabels,test_size=0.3, random_state=42)

# Build Model
nv_model = MultinomialNB()
nv_model.fit(X_train, y_train)

# Accuracy
# method 1
nv_model.score(X_test, y_test)

# Predictions 
y_pred_for_nv = nv_model.predict(X_test)

### Make a prediction
vect = cv.transform(sample_text).toarray()
nv_model.predict(vect)

## Check for accuracy/confidence
nv_model.predict_proba(vect)
nv_model.classes_
np.max(nv_model.predict_proba(vect))

def predict_emotion(sample_text, model):
    myvect = cv.transform(sample_text).toarray()
    prediction = model.predict(myvect)
    pred_proba = model.predict_proba(myvect)
    pred_percentage_for_all = dict(zip(model.classes_, pred_proba[0]))
    print(pred_percentage_for_all)
    print("NB Model -> Prediction:{}, confidence: {}".format(prediction[0], np.max(pred_proba)))
    #print(prediction[0])  # prints the emotion name
    return pred_percentage_for_all

predict_emotion(sample_text, nv_model)

### Model evaluation #################
#print(classification_report(y_test, y_pred_for_nv))

# Confusion matrix
#confusion_matrix(y_test, y_pred_for_nv)

### Save Model
import joblib
model_file = open("emotion_classifier_nv_model_22_january_2024.pkl", "wb")
joblib.dump(nv_model,model_file)
model_file.close()

### Model Interpretation
# Eli5
# Lime
# Shap

# Log Regression
#lr_model = LogisticRegression()
#lr_model.fit(X_train,y_train)

## UNFINISHED
#print("Printing for LR model")
# Accuracy
#print("LR Score ->", lr_model.score(X_test, y_test))

# Single Predict
#print(predict_emotion(sample_text, lr_model))