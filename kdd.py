"""
Author : Jyotsna Namdeo Nakte jnn2078
Author : Jairaj Tikam
Date: 2nd December,2018
This program helps us cleaning the data to prepare it for visualizations in Tableau

"""


# Dealing with data frame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
nltk.download('wordnet')
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Classification Algorithms
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# Classification evaluation metrics
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

#Data Splitting
from sklearn.model_selection import train_test_split

# Sampling of the dataset for fine tuning of the model
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

'''
Method that takes the values from csv files stores the data in list
'''

stop = text.ENGLISH_STOP_WORDS
smote=SMOTE()

tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def data_preprocessing(text):
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace('\d+', '')
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return text

def lemmatization(text):
    return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text)]

def get_data():
    df = pd.read_table( '/Users/j317592/Desktop/KDD/WomensClothingE-CommerceReviews.csv', delimiter=',')

    #print(df)
    return df


def importance_plotting(data, x, y, palette, title):
    sns.set(style="whitegrid")
    ft = sns.PairGrid(data, y_vars=y, x_vars=x, size=5, aspect=1)
    ft.map(sns.stripplot, orient='h', palette=palette, edgecolor="black", size=15)
    for ax, title in zip(ft.axes.flat, title):
        # Set a different title for each axes
        ax.set(title=title)

        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    plt.show()

def model111(mod, model_name, x_train, y_train, x_test, y_test):
    mod.fit(x_train, y_train)
    print(model_name)
    acc = cross_val_score(mod, x_train, y_train, scoring = "accuracy", cv = 5)
    predictions = cross_val_predict(mod, x_test, y_test, cv = 5)
    cm = confusion_matrix(predictions, y_test)
    print("Confusion Matrix:  \n", cm)
    print ('Accuracy:', accuracy_score(y_test, predictions))
    print ('F1 score:', f1_score(y_test, predictions))
    print('Recall:', recall_score(y_test, predictions))
    print ('Precision:', precision_score(y_test, predictions))


def model_sm(mod, model_name, x_train_sm, y_train_sm ,x_test_sm, y_test_sm):
    mod.fit(x_train_sm, y_train_sm)
    print(model_name)
    acc = cross_val_score(mod, x_train_sm, y_train_sm, scoring = "accuracy", cv = 5)
    predictions = cross_val_predict(mod, x_test_sm, y_test_sm, cv = 5)
    cm = confusion_matrix(predictions, y_test_sm)
    print("Confusion Matrix:  \n", cm)
    print('Accuracy:', accuracy_score(y_test_sm, predictions))
    print('F1 score:', f1_score(y_test_sm, predictions))
    print('Recall:', recall_score(y_test_sm, predictions))
    print('Precision:', precision_score(y_test_sm, predictions))


def remove_noise(text):
    # Make lowercase
    text = text.apply(lambda x: " ".join(x.lower() for x in x.split()))

    # Remove whitespaces
    text = text.apply(lambda x: " ".join(x.strip() for x in x.split()))

    # Remove special characters
    text = text.apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

    # Remove punctuation
    text = text.str.replace('[^\w\s]', '')

    # Remove numbers
    text = text.str.replace('\d+', '')

    # Remove Stopwords
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    # Convert to string
    text = text.astype(str)

    return text



def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("Threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

def main():
    #receiving the data
    print(stop)
    df=get_data()
    df = df[['Clothing ID', 'Review Text', 'Recommended IND']]
    df.columns = ['EmployeeID', 'Review Text', 'Recommend']
    #df.info()
    df['Review Text'].fillna('unknown', inplace=True)
    #df.info()
    df['Cleaned Text'] = remove_noise(df['Review Text'])
    df['Cleaned Text'] = df['Cleaned Text'].apply(lemmatization)
    df.head()
    #print(df.head().to_string())
    cvec = CountVectorizer(min_df=.005, max_df=.9, ngram_range=(1, 2), tokenizer=lambda doc: doc, lowercase=False)
    cvec.fit(df['Cleaned Text'])

    print("Length of vocabulary found: "+str(len(cvec.vocabulary_)))
    cvec_counts = cvec.transform(df['Cleaned Text'])
    print('sparse matrix shape:', cvec_counts.shape)
    print('nonzero count:', cvec_counts.nnz)
    print('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))
    transformer = TfidfTransformer()

    # Fitting and transforming n-grams
    transformed_weights = transformer.fit_transform(cvec_counts)
    #print(transformed_weights)
    transformed_weights = transformed_weights.toarray()
    vocab = cvec.get_feature_names()

    # Putting weighted n-grams into a DataFrame and computing some summary statistics
    model = pd.DataFrame(transformed_weights, columns=vocab)
    print(model.head().to_string())



    model['Keyword'] = model.idxmax(axis=1)
    model['Max'] = model.max(axis=1)
    model['Sum'] = model.drop('Max', axis=1).sum(axis=1)
    model = pd.merge(df, model, left_index=True, right_index=True)
    #print(model.head(10))
    #print(model.columns)
    ml_model = model.drop(['EmployeeID', 'Review Text', 'Cleaned Text', 'Keyword', 'Max', 'Sum'],
                          axis=1)

    # Create X & y variables for Machine Learning
    X = ml_model.drop('Recommend', axis=1)
    y = ml_model['Recommend']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)


    X_sm, y_sm = smote.fit_sample(X, y)
    X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, test_size=0.3, random_state=100)
    print(X_train_sm.size)
    print(X_test_sm.size)


    randum = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = randum.fit_resample(X, y)
    X_train_un, X_test_un, y_train_un, y_test_un = train_test_split( X_resampled, y_resampled, test_size=0.3, random_state=100)
    print(X_train_un.size)
    print(X_test_un.size)



    gnb = GaussianNB()
    model_sm(gnb, "Gaussian Naive Bayes", X_train_un, X_test_un, y_train_un, y_test_un)
    ran = RandomForestClassifier(n_estimators=50)
    model_sm(ran, "Random Forest Classifier", X_train_un, X_test_un, y_train_un, y_test_un)
    log = LogisticRegression()
    model_sm(log, "Logistic Regression", X_train_un, X_test_un, y_train_un, y_test_un)
    svc = LinearSVC()
    model_sm(svc, "Linear SVC", X_train_un, X_test_un, y_train_un, y_test_un)
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
   model_sm(NN, "NN", X_train_un, X_test_un, y_train_un, y_test_un)

'''
    gnb = GaussianNB()
    model_sm(gnb, "Gaussian Naive Bayes", X_train_sm, y_train_sm, X_test_sm, y_test_sm)
    ran = RandomForestClassifier(n_estimators=50)
    model_sm(ran, "Random Forest Classifier", X_train_sm, y_train_sm, X_test_sm, y_test_sm)
    log = LogisticRegression()
    model_sm(log, "Logistic Regression", X_train_sm, y_train_sm, X_test_sm, y_test_sm)
    svc = LinearSVC()
    model_sm(svc, "Linear SVC", X_train_sm, y_train_sm, X_test_sm, y_test_sm)
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
   model_sm(NN, "NN", X_train_sm, y_train_sm, X_test_sm, y_test_sm)
'''

'''
    gnb = GaussianNB()
    model111(gnb, "Gaussian Naive Bayes", X_train, y_train, X_test, y_test)
    ran = RandomForestClassifier(n_estimators=50)
    model111(ran, "Random Forest Classifier", X_train, y_train, X_test, y_test)
    log = LogisticRegression()
    model111(log, "Logistic Regression", X_train, y_train, X_test, y_test)
    svc = LinearSVC()
    model111(svc, "Linear SVC", X_train, y_train, X_test, y_test)
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
    model111(NN, "NN", X_train, y_train, X_test, y_test)
    
    
    
    
    y_scores = gnb.predict_proba(X_train)
    y_scores = y_scores[:, 1]

    precision, recall, threshold = precision_recall_curve(y_train, y_scores)
    plt.figure(figsize=(14, 7))
    plot_precision_and_recall(precision, recall, threshold)
    plt.show()
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)
    plt.figure(figsize=(14, 7))
    plot_roc_curve(false_positive_rate, true_positive_rate)
    plt.show()
'''

if __name__ == '__main__':
    main()