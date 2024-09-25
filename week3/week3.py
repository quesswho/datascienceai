import os
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random
random.seed(6215)

spam_folder_path = 'spam'
spam_files = os.listdir(spam_folder_path)

easy_ham_folder_path = "easy_ham"
easy_ham_files = os.listdir(easy_ham_folder_path)

hard_ham_folder_path = "hard_ham"
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


# Split dataset (Spam vs Hard Ham)
X_train_hard, X_test_hard, y_train_hard, y_test_hard = train_test_split(
    all_hard_ham + all_spam, 
    [0] * len(all_hard_ham) + [1] * len(all_spam), 
    test_size=0.5, random_state=42
)

# Multinomial Naive Bayes Classifier 
vectorizer = CountVectorizer(encoding="latin_1")
X_train_counts = vectorizer.fit_transform(X_train_hard)
X_test_counts = vectorizer.transform(X_test_hard)
clf_multinomial = MultinomialNB()
clf_multinomial.fit(X_train_counts, y_train_hard)
y_pred_multinomial = clf_multinomial.predict(X_test_counts)

# Evaluation for Multinomial NB
accuracy_multinomial = accuracy_score(y_test_hard, y_pred_multinomial)
conf_matrix_multinomial = confusion_matrix(y_test_hard, y_pred_multinomial)
class_report_multinomial = classification_report(y_test_hard, y_pred_multinomial)

print(f"Multinomial Naive Bayes Accuracy: {accuracy_multinomial * 100:.2f}%")
print("\nConfusion Matrix (Multinomial Naive Bayes):")
print(conf_matrix_multinomial)
print("\nClassification Report (Multinomial Naive Bayes):")
print(class_report_multinomial)


# Bernoulli Naive Bayes Classifier (binary)
binary_vectorizer = CountVectorizer(encoding="latin_1", binary=True)
X_train_binary = binary_vectorizer.fit_transform(X_train_hard)
X_test_binary = binary_vectorizer.transform(X_test_hard)
clf_bernoulli = BernoulliNB()
clf_bernoulli.fit(X_train_binary, y_train_hard)
y_pred_bernoulli = clf_bernoulli.predict(X_test_binary)

# Evaluation for Bernoulli NB
accuracy_bernoulli = accuracy_score(y_test_hard, y_pred_bernoulli)
conf_matrix_bernoulli = confusion_matrix(y_test_hard, y_pred_bernoulli)
class_report_bernoulli = classification_report(y_test_hard, y_pred_bernoulli)

# Output the results
print(f"\nBernoulli Naive Bayes Accuracy: {accuracy_bernoulli * 100:.2f}%")
print("\nConfusion Matrix (Bernoulli Naive Bayes):")
print(conf_matrix_bernoulli)
print("\nClassification Report (Bernoulli Naive Bayes):")
print(class_report_bernoulli)