from sklearn.feature_extraction.text import CountVectorizer


def vectorize(mail):
    prep_email = [" ".join(mail['subject']) + " ".join(mail['body'])]

    vectorizer = CountVectorizer()
    X = vectorizer.fit(prep_email)

    print("Email:", prep_email, "\n")
    print("Características de entrada:", vectorizer.get_feature_names_out())

