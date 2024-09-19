import os
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random
random.seed(6215)

spam_folder_path = '20021010_spam/spam'
spam_files = os.listdir(spam_folder_path)

easy_ham_folder_path = "20021010_easy_ham/easy_ham"
easy_ham_files = os.listdir(easy_ham_folder_path)

hard_ham_folder_path = "20021010_hard_ham/hard_ham"
hard_ham_files = os.listdir(hard_ham_folder_path)

all_spam = []
all_easy_ham = []
all_hard_ham = []

for folder_path, content_list, file_list in zip([spam_folder_path, easy_ham_folder_path, hard_ham_folder_path], [all_spam, all_easy_ham, all_hard_ham], [spam_files, easy_ham_files, hard_ham_files]):
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        f = open(file_path, 'r', encoding="latin_1")
        content_list.append(f.read())
    random.shuffle(content_list)



X_train_easy, X_test_easy, y_train_easy, y_test_easy = train_test_split(all_easy_ham + all_spam, [0] * len(all_easy_ham) + [1] * len(all_spam), test_size=0.5, random_state=42)

# Multinomial Naive Bayesian Classyfier 
vectorizer = CountVectorizer(encoding="latin_1")
X_train_counts = vectorizer.fit_transform(X_train_easy)
X_test_counts = vectorizer.transform(X_test_easy)
clf = MultinomialNB()
clf.fit(X_train_counts, y_train_easy)
y_pred = clf.predict(X_test_counts)
accuracy = accuracy_score(y_test_easy, y_pred)
conf_matrix = confusion_matrix(y_test_easy, y_pred)
class_report = classification_report(y_test_easy, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)



# Bernoulli Naive Bayesian Classyfier
binary_vectorizer = CountVectorizer(encoding="latin_1", binary=True)
X_train_binary = vectorizer.fit_transform(X_train_easy)
X_test_binary = vectorizer.transform(X_test_easy)
clf = BernoulliNB()
clf.fit(X_train_binary, y_train_easy)
y_pred = clf.predict(X_test_binary)
accuracy = accuracy_score(y_test_easy, y_pred)
conf_matrix = confusion_matrix(y_test_easy, y_pred)
class_report = classification_report(y_test_easy, y_pred)

# Output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)