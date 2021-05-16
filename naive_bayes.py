import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

processed_path = 'processed\\'
model_path = 'model\\'

vectorizer = joblib.load(processed_path + 'vectorizer.pkl')
X_train = joblib.load(processed_path + 'train_tfidf.pkl')
X_test = joblib.load(processed_path + 'test_tfidf.pkl')

y_train = []
y_test = []
with open(processed_path + 'processed_train_data.txt', encoding = 'utf-16') as f:
    lines = f.read().splitlines()
    for line in lines:
        label = line.split('<fff>')[0]
        y_train.append(int(label))

with open(processed_path + 'processed_test_data.txt', encoding = 'utf-16') as f:
    lines = f.read().splitlines()
    for line in lines:
        label = line.split('<fff>')[0]
        y_test.append(int(label))

y_train = np.array(y_train)
y_test = np.array(y_test)
assert X_train.shape[0] == len(y_train)
assert X_test.shape[0] == len(y_test)

classifier = MultinomialNB().fit(X_train, y_train)

#evaluate on training set
y_pred = classifier.predict(X_train)
print('accuracy on training set: ', accuracy_score(y_pred, y_train))

#evaluate on test set
y_pred = classifier.predict(X_test)
print('accuracy on test set: ', accuracy_score(y_pred, y_test))

#save classifier

joblib.dump(classifier, model_path + 'naive_bayes.pkl')