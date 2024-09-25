from utils.data_treatment.load_arff_dataset import load_arff_dataset
from utils.data_treatment.label_encoder import encode_categorical_variables
from utils.formatting.colors import Colors
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def data_visualization():
    """
    Visualizes the data using pandas and matplotlib.

    This function reads the data from the KDDTrain+.arff file and displays
    the first 10 rows, the info of the dataframe, and the descriptive statistics.
    """
    print("\n\n" + Colors.HEADER + Colors.UNDERLINE + "\nDATA VISUALIZATION\n" + Colors.ENDC)

    # Load the dataset
    df_orig = load_arff_dataset('archive/KDDTrain+.arff')

    # Create a copy of the dataframe
    df = df_orig.copy()

    # Print the first 10 rows of the dataset
    print(Colors.WARNING + "First 10 rows of the dataset:" + Colors.ENDC)
    print(df_orig.head(10))

    # Print the info of the dataframe
    print(Colors.WARNING + "\nInfo of the dataframe:" + Colors.ENDC)
    print(df.info())

    # Print the descriptive statistics of the dataframe
    print(Colors.WARNING + "\nDescriptive statistics of the dataframe:" + Colors.ENDC)
    print(df.describe())

    # Get the value counts of the protocol_type column
    print(Colors.WARNING + "\nValue counts of the protocol_type column:" + Colors.ENDC)
    print(df["protocol_type"].value_counts())

    # Get the value counts of the class column
    print(Colors.WARNING + "\nValue counts of the class column:" + Colors.ENDC)
    print(df["class"].value_counts())

    # Plot the histogram of the protocol_type column
    plt.title('Histogram of the protocol_type column')
    df["protocol_type"].hist()

    # Plot the histogram of all the columns
    plt.title('Histogram of all the columns')
    df.hist(bins=50, figsize=(20, 15))

    # Print the values of the class column
    print(Colors.WARNING + "\nValues of the class column:" + Colors.ENDC, '\n\n')
    print(df["class"])

    # Print the values of the protocol_type column
    print(Colors.WARNING + "\nValues of the protocol_type column:" + Colors.ENDC, '\n\n')
    print(df["protocol_type"])

    # Encode the categorical variables
    # The encode_categorical_variables function takes the dataframe and a dictionary of parameters
    # The dictionary of parameters has the key 'labels' that is a list of the names of the categorical variables
    df = encode_categorical_variables(df, **{'labels': ['class', 'protocol_type', 'service', 'flag']})

    # Print the values of the class column after encoding
    print(Colors.WARNING + "\nValues of the class column (after encoding):" + Colors.ENDC, '\n\n')
    print(df["class"])

    # Print the values of the protocol_type column after encoding
    print(Colors.WARNING + "\nValues of the protocol_type column (after encoding):" + Colors.ENDC, '\n\n')
    print(df["protocol_type"])

    # Correlation matrix of the atributes in the dataset
    corr_matrix = df.corr()

    print(
        Colors.WARNING + "\nLinear Correlation matrix of the class attribute with all the attributes in the dataset:" + Colors.ENDC,
        '\n\n')
    print(corr_matrix['class'].sort_values(ascending=False))

    print(
        Colors.WARNING + "\nLinear correlation matrix of the all attribute with all the attributes in the dataset:" + Colors.ENDC,
        '\n\n')
    print(df.corr())

    # Correlation matrix
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.matshow(corr)
    plt.title('Correlation Matrix')
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)

    # Scatter matrix
    attributes = ["same_srv_rate", "dst_host_srv_count", "class", "dst_host_same_srv_rate"]
    scatter_matrix(df[attributes], figsize=(12, 8))
    plt.show()
