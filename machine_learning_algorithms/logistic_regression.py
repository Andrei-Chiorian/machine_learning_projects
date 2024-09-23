import utils.data_treatment as dt
import pprint
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics import accuracy_score
from utils.formatting.Spinner import Spinner
from utils.formatting.colors import Colors
import numpy as np
from joblib import load, dump

# This exercise shows the fundamentals of Logistic Regression,
# posing one of the first problems that were solved through the
# use of Machine Learning techniques: the detection of SPAM.

# The construction of a machine learning system capable of
# predicting whether a given email corresponds to a SPAM email
# or not is proposed. To do this, the following set of data will be used:

# 2007 TREC Public Spam Corpus
# The corpus trec07p contains 75,419 messages:
# - 25220 ham
# - 50199 spam

# In this practical case related to the detection of SPAM emails,
# the data set we have is made up of emails, with their corresponding
# headers and additional fields. Therefore, they require preprocessing
# before they are ingested by the Machine Learning algorithm.
# Class used is MLStripper located in the folder /utils/data_treatment

# In addition to eliminating possible HTML tags found in the email
# electronic, other preprocessing actions must be taken to avoid
# that messages contain unnecessary noise. Among them is the
# removal of punctuation marks, removal of possible fields from the
# email that are not relevant or removing affixes from a
# word keeping only the root of it (Stemming). The class that
# shown below performs these transformations.


def logistic_regression():
    clf = None
    spinner = Spinner()

    print(Colors.HEADER, Colors.UNDERLINE, "\nLOGISTIC REGRESSION ALGORITHM \n", Colors.ENDC)
    print(Colors.HEADER, "\nThis exercise shows the fundamentals of Logistic Regression\n"
                         "posing one of the first problems that were solved through the \n"
                         "use of Machine Learning techniques: the detection of SPAM. \n"
                         "The dataset contains 75.419 messages\n\n", Colors.ENDC)

    try:
        # Intentamos cargar el modelo
        clf = load('logi_reg_trained_model.joblib')
        print("Model loaded successfully.")

        test_input_checked = False
        num_sample_test = 0
        while test_input_checked is False:
            num_sample_test = int(input(
                Colors.OKCYAN + "\nPlease enter the number of emails for testing the model: " + Colors.ENDC))
            if 0 < num_sample_test <= 75419:
                test_input_checked = True

        X, y = dt.create_prep_dataset("datasets/full/index", num_sample_test)

        X_test, y_test = X, y

        print(Colors.WARNING, f"\nNumbers of email for testing:", Colors.ENDC, Colors.OKGREEN, len(X_test),
              '\n', Colors.ENDC)

        vectorizer = CountVectorizer()

        X_test = vectorizer.transform(X_test)
        y_pred = spinner.run_with_spinner(lambda: clf.predict(X_test), "predicting result")

        train_spam_count = np.count_nonzero(y_pred == 'spam')
        test_spam_count = y_test.count('spam')

        print(Colors.OKBLUE, f'\nNumbers of spam emails in the test set (total: {len(y_test)}):', Colors.OKCYAN,
              {test_spam_count}, Colors.ENDC)

        print(Colors.OKBLUE, f'\nNumbers of spam emails in the predicted set (total: {len(y_test)}):', Colors.OKCYAN,
              {train_spam_count}, Colors.ENDC)

        print(Colors.OKBLUE, '\nAccuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)), Colors.ENDC)

    except FileNotFoundError:
        # Si no existe el archivo, capturamos la excepción
        clf = LogisticRegression()
        print("Model file not found. Continuing without loading the model.")

        train_input_checked = False
        num_sample_train = 0
        while train_input_checked is False:
            num_sample_train = int(input(
                Colors.OKCYAN + "\nPlease enter the number of emails for training the model: " + Colors.ENDC))
            if 0 < num_sample_train <= 75419:
                train_input_checked = True

        test_input_checked = False
        num_sample_test = 0
        while test_input_checked is False:
            num_sample_test = int(input(
                Colors.OKCYAN + "\nPlease enter the number of emails for testing the model: " + Colors.ENDC))
            if 0 < num_sample_test <= (75419 - num_sample_train):
                test_input_checked = True

        # We read the total number of emails
        X, y = dt.create_prep_dataset("datasets/full/index", num_sample_train + num_sample_test)

        # We get the first n elements, where n is the value of the variable num_sample_train, from the total number of
        # parsed emails
        X_train, y_train = X[:num_sample_train], y[:num_sample_train]

        # We skip the first n elements, where n is the value of the variable num_sample_train, from the total number of
        # parsed emails and retrieve the remaining ones
        X_test, y_test = X[num_sample_train:], y[num_sample_train:]

        print(Colors.WARNING, f"\n\nNumbers of email for training:", Colors.ENDC, Colors.OKGREEN, len(X_train),
              '\n', Colors.ENDC)
        print(Colors.WARNING, f"\nNumbers of email for testing:", Colors.ENDC, Colors.OKGREEN, len(X_test),
              '\n', Colors.ENDC)

        print(Colors.WARNING,
              "\nGenerated data set, each element of the array are the words extracted from the body and "
              "subject \nof one email.These words are obtained by applying the stemming technique, which "
              "involves reducing \nwords to their base form to enhance text processing and analysis.",
              Colors.ENDC)

        print()
        pprint.pprint(X_train[:3])
        print()

        # TRAINING

        vectorizer = CountVectorizer()
        X_train = spinner.run_with_spinner(lambda: vectorizer.fit_transform(X_train), "vectorization")

        print(Colors.WARNING, "\nThe result of applying the vectorization is a sparse matrix where each row represents"
                              " a document, \neach column represents a unique word from the vocabulary, and the values "
                              "indicate if \neach word appears in the corresponding document. This process "
                              "converts text data into a numerical format \nsuitable for machine learning algorithms\n",
              Colors.ENDC)

        X_train_subset = X_train[:10, :50].toarray()
        feature_names_subset = vectorizer.get_feature_names_out()[:50]

        x_train_df = spinner.run_with_spinner(lambda:
                                              pd.DataFrame(X_train_subset, columns=[feature_names_subset]
                                                           ),
                                              "dataframe")
        print(x_train_df.iloc[:, :6].to_markdown())

        print(Colors.WARNING, "\nFeatures (words):", Colors.ENDC, Colors.OKGREEN,
              len(vectorizer.get_feature_names_out()),
              Colors.ENDC)

        spinner.run_with_spinner(lambda: clf.fit(X_train, y_train), "training model")

        print(Colors.OKGREEN, "\nModel trained", Colors.ENDC)

        # TESTING

        X_test = vectorizer.transform(X_test)
        y_pred = spinner.run_with_spinner(lambda: clf.predict(X_test), "predicting result")

        train_spam_count = np.count_nonzero(y_pred == 'spam')
        test_spam_count = y_test.count('spam')

        print(Colors.OKBLUE, f'\nNumbers of spam emails in the test set (total: {len(y_test)}):', Colors.OKCYAN,
              {test_spam_count}, Colors.ENDC)

        print(Colors.OKBLUE, f'\nNumbers of spam emails in the predicted set (total: {len(y_test)}):', Colors.OKCYAN,
              {train_spam_count}, Colors.ENDC)

        print(Colors.OKBLUE, '\nAccuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)), Colors.ENDC)

        to_save_input = False
        while to_save_input is False:
            to_save = input(Colors.OKCYAN + "\nDo you want to save the model? yes(y,Y) no(n,N)" + Colors.ENDC)
            to_save.lower()
            if to_save == 'n' or to_save == 'y':
                if to_save == 'y':
                    try:
                        dump(clf, 'logi_reg_trained_model.joblib')
                        print("Model saved successfully.")
                    except Exception as e:
                        print("Model could not be saved.")
                to_save_input = True
