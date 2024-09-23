import email
import string
import nltk
from utils.data_treatment.ml_stripper import MLStripper
import os

nltk.download('stopwords')

DATASET_PATH = os.path.join("datasets")


def parse_index(path_to_index, n_elements):
    """
       Parses the index file to extract email labels and their paths.

       Args:
           path_to_index (str): The file path to the index file that contains the email metadata.
           n_elements (int): The number of elements to parse from the index.

       Returns:
           list: A list of dictionaries, where each dictionary contains:
               - "label" (str): The label/category of the email (e.g., spam, ham).
               - "email_path" (str): The full path to the email file.
    """
    ret_indexes = []
    index = open(path_to_index).readlines()
    for i in range(n_elements):
        mail = index[i].split(" ../")
        label = mail[0]
        path = mail[1][:-1]
        path_mail = path.split("/")[-1]
        ret_indexes.append({"label": label, "email_path": os.path.join(DATASET_PATH, os.path.join("data", path_mail))})
    return ret_indexes


def parse_email(index):
    """
        Parses the email content from the given index.

        Args:
            index (dict): A dictionary containing:
                - "label" (str): The label/category of the email.
                - "email_path" (str): The full path to the email file.

        Returns:
            tuple: A tuple containing:
                - pmail: The parsed email content.
                - label: The label/category of the email.
    """
    p = Parser()
    pmail = p.parse(index["email_path"])
    return pmail, index["label"]


class Parser:

    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)

    @staticmethod
    def strip_tags(html):
        s = MLStripper()
        s.feed(html)
        return s.get_data()

    def parse(self, email_path):
        """Parse an email."""
        with open(email_path, errors='ignore') as e:
            msg = email.message_from_file(e)
        return None if not msg else self.get_email_content(msg)

    def get_email_content(self, msg):
        """Extract the email content."""
        subject = self.tokenize(msg['Subject']) if msg['Subject'] else []
        body = self.get_email_body(msg.get_payload(),
                                   msg.get_content_type())
        content_type = msg.get_content_type()
        # Returning the content of the email
        return {"subject": subject,
                "body": body,
                "content_type": content_type}

    def get_email_body(self, payload, content_type):
        """Extract the body of the email."""
        body = []
        if type(payload) is str and content_type == 'text/plain':
            return self.tokenize(payload)
        elif type(payload) is str and content_type == 'text/html':
            return self.tokenize(self.strip_tags(payload))
        elif type(payload) is list:
            for p in payload:
                body += self.get_email_body(p.get_payload(),
                                            p.get_content_type())
        return body

    def tokenize(self, text):
        """Transform a text string in tokens. Perform two main actions,
        clean the punctuation symbols and do stemming of the text."""
        for c in self.punctuation:
            text = text.replace(c, "")
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        tokens = list(filter(None, text.split(" ")))
        # Stemming of the tokens
        return [self.stemmer.stem(w) for w in tokens if w not in self.stopwords]
