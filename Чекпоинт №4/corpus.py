class Corpus:

    def __init__(self, authors: list = None, books: list = None, tokens: list = None):
        if authors and books and tokens:
            self.authors = authors
            self.books = books
            self.tokens = tokens
            self.texts = None
        else:
            self.authors = []
            self.books = []
            self.tokens = []
            self.texts = []