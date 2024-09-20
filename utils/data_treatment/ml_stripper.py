from html.parser import HTMLParser


# This class facilitates the preprocessing of emails that have HTML code
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


# This function is responsible for removing HTML tags found in the email text
def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()