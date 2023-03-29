#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy seaborn matplotlib scikit-learn nltk')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[3]:


# Load the dataset into a pandas dataframe
df = pd.read_csv('tweet_data.csv')

# Print the first few rows of the dataframe
print(df.head())


# In[4]:


# Convert all text to lowercase
df['tweet_text'] = df['tweet_text'].apply(lambda x: x.lower())

# Remove unnecessary characters, numbers and symbols
df['tweet_text'] = df['tweet_text'].str.replace("[^a-zA-Z#]", " ")

# Remove stop words
stopwords_set = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [word for word in text.split() if word not in stopwords_set]
    return " ".join(text)
df['tweet_text'] = df['tweet_text'].apply(lambda x: remove_stopwords(x))

# Tokenize the text
df['tokenized_text'] = df['tweet_text'].apply(lambda x: x.split())

# Print the first few rows of the cleaned data
print(df.head())


# In[5]:


# Count the number of positive and negative tweets
sns.countplot(df['sentiment'])

# Print the percentage of positive and negative tweets
positive_tweets = len(df[df['sentiment'] == 'positive'])
negative_tweets = len(df[df['sentiment'] == 'negative'])
print('Percentage of positive tweets: {}%'.format(round(positive_tweets/len(df)*100, 2)))
print('Percentage of negative tweets: {}%'.format(round(negative_tweets/len(df)*100, 2)))

# Plot the distribution of tweet lengths
df['tweet_length'] = df['tweet_text'].apply(lambda x: len(x))
sns.histplot(df['tweet_length'], kde=True)

# Print the average tweet length
print('Average tweet length: {}'.format(round(np.mean(df['tweet_length']), 2)))


# In[ ]:





# In[6]:


import nltk

nltk.download('punkt')
nltk.download('wordnet')


# In[7]:


from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define a function for stemming
def stem_text(text):
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word.isalpha()]
    return ' '.join(words)

# Define a function for lemmatization
def lemmatize_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha()]
    return ' '.join(words)

# Apply the functions to the tweet text
df['stemmed_text'] = df['tweet_text'].apply(stem_text)
df['lemmatized_text'] = df['tweet_text'].apply(lemmatize_text)




# Write the cleaned dataframe to a CSV file
df.to_csv('cleaned_tweet2.csv', index=False)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer

# Create vectorizer object
vectorizer = CountVectorizer(stop_words='english')


# In[9]:


# Fit vectorizer to stemmed text
X_stemmed = vectorizer.fit_transform(df['stemmed_text'])

# Print the shape of the matrix
print(X_stemmed.shape)


# In[10]:


# Fit vectorizer to lemmatized text
X_lemmatized = vectorizer.fit_transform(df['lemmatized_text'])

# Print the shape of the matrix
print(X_lemmatized.shape)


# In[11]:


from sklearn.model_selection import train_test_split





X = df['lemmatized_text']  # assuming the text data is in a column named 'text'
y = df['sentiment']  # assuming the labels are in a column named 'label'

#X = vectorizer.fit_transform(df['lemmatized_text'])
#y = df['sentiment']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

nb_pipeline = Pipeline([
    ('vect', CountVectorizer(lowercase=False)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])


# In[ ]:





# In[13]:


nb_pipeline.fit(X_train, y_train)

y_pred = nb_pipeline.predict(X_test)


# In[14]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Compute accuracy, precision, recall, and F1 score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[15]:


from sklearn.linear_model import LogisticRegression

lr_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression())
])

lr_pipeline.fit(X_train, y_train)

y_pred_lr = lr_pipeline.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr, average='weighted')
rec_lr = recall_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

print("Accuracy (Logistic Regression):", acc_lr)
print("Precision (Logistic Regression):", prec_lr)
print("Recall (Logistic Regression):", rec_lr)
print("F1 Score (Logistic Regression):", f1_lr)

# Plot confusion matrix for logistic regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()


# In[25]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

svm_pipeline = Pipeline([
    ('vect', TfidfVectorizer(lowercase=False)),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(kernel='linear'))
])

svm_pipeline.fit(X_train, y_train)
y_pred = svm_pipeline.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1 Score: {f1}")



# In[18]:


from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', RandomForestClassifier())
])

rf_pipeline.fit(X_train, y_train)

y_pred_rf = rf_pipeline.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
rec_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print("Accuracy (Random Forest):", acc_rf)
print("Precision (Random Forest):", prec_rf)
print("Recall (Random Forest):", rec_rf)
print("F1 Score (Random Forest):", f1_rf)


cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Random Forest)')
plt.show()


# In[21]:


from sklearn.tree import DecisionTreeClassifier

dt_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', DecisionTreeClassifier())
])

dt_pipeline.fit(X_train, y_train)

y_pred_dt = dt_pipeline.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average='weighted')
rec_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

print("Accuracy (Decision Tree):", acc_dt)
print("Precision (Decision Tree):", prec_dt)
print("Recall (Decision Tree):", rec_dt)
print("F1 Score (Decision Tree):", f1_dt)

# Plot confusion matrix for decision tree classifier
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Decision Tree Classifier)')
plt.show()


# In[30]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define the pipeline with hyperparameters to search
lr_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('scale', StandardScaler(with_mean=False)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Define the hyperparameter grid to search over
param_grid = {
    'vect__max_features': [1000, 5000, 10000],
    'tfidf__use_idf': [True, False],
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [0.1, 1, 10]
}

# Perform the grid search to find the best hyperparameters
grid_search = GridSearchCV(lr_pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and their scores
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)

# Evaluate the model on the test set using the best hyperparameters
y_pred_lr = grid_search.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr, average='weighted')
rec_lr = recall_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

print("Accuracy (Logistic Regression):", acc_lr)
print("Precision (Logistic Regression):", prec_lr)
print("Recall (Logistic Regression):", rec_lr)
print("F1 Score (Logistic Regression):", f1_lr)

# Plot confusion matrix for logistic regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()


# In[126]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

# Define the logistic regression pipeline
lr_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression())
])

# Train the logistic regression model on the training data
lr_pipeline.fit(X_train, y_train)

# Define a sentence to predict
new_sentence = "Thank goodness its Friday"
# Predict the sentiment of the new sentence using the logistic regression model
predicted_probabilities = lr_pipeline.predict_proba([new_sentence])[0]
positive_probability = predicted_probabilities[1]

# Set a threshold for the positive sentiment probability
threshold = 0.5

# Print the predicted sentiment and probabilities
if positive_probability >= threshold:
    print("The sentence '{}' is positive with probability {:.2f}.".format(new_sentence, positive_probability))
else:
    print("The sentence '{}' is negative with probability {:.2f}.".format(new_sentence, 1 - positive_probability))

