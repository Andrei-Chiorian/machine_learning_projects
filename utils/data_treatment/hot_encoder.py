from sklearn.preprocessing import OneHotEncoder


def hot_enc(mail):
    prep_email = [[w] for w in mail['subject'] + mail['body']]

    enc = OneHotEncoder(handle_unknown='ignore')
    X = enc.fit_transform(prep_email)

    print("Features:\n", enc.get_feature_names_out())
    print("\nValues:\n", X.toarray())
