from sklearn.preprocessing import LabelEncoder


def encode_categorical_variables(df, labels):
    """
    Encodes categorical variables of a dataframe into numerical
    variables using LabelEncoder.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be encoded. This dataframe should have
        columns that are categorical variables to be encoded.
    labels : list of str
        The columns to be encoded. These labels should match the
        names of the columns in the dataframe.

    Returns
    -------
    pandas.DataFrame
        The dataframe with the categorical variables encoded.
        The categorical variables are replaced by numerical
        variables. The numerical values are determined by the
        LabelEncoder.
    """
    # Create a LabelEncoder object
    encoder = LabelEncoder()

    # Iterate over the columns given in the labels list
    for label in labels:
        # Fit the encoder to the column and transform the column
        # into numerical values.
        df[label] = encoder.fit_transform(df[label])

    # Return the dataframe with the categorical variables encoded
    return df
