import arff
import pandas as pd


def load_arff_dataset(data_path):
    """
    Load an ARFF file and return a Pandas DataFrame

    Parameters
    ----------
    data_path : str
        The path to the ARFF file to load

    Returns
    -------
    df : pd.DataFrame
        A Pandas DataFrame containing the data from the ARFF file

    Notes
    -----
    The function uses the arff.load() function from the arff library
    to load the ARFF file into a dictionary. The dictionary is then
    converted into a Pandas DataFrame and returned.

    """
    # Open the ARFF file and read it into a dictionary
    with open(data_path, 'r') as train_set:
        # Use the arff.load() function to load the ARFF file into a dictionary
        dataset = arff.load(train_set)

    # Extract the feature names from the dictionary
    # The feature names are the first element of each tuple in the attributes list
    attributes = [attr[0] for attr in dataset["attributes"]]

    # Create a Pandas DataFrame from the data
    # The DataFrame is created with the feature names as the column names
    # and the data from the ARFF file as the rows
    df = pd.DataFrame(dataset["data"], columns=attributes)

    return df


