class Lang:
    def __init__(self):
        self.word2index = {'UNK': 0, "PAD": 1}
        self.tag2index = {}
        self.word2count = {}
        self.tag2count = {}
        self.index2word = {0: 'UNK', 1: "PAD"}
        self.index2tag = {}
        self.n_words = 2  # Count default tokens
        self.n_tags = 0  # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split():
            self.index_word(word)

    def index_tags(self, sentence):
        for word in sentence.split():
            self.index_tag(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def index_tag(self, word):
        if word not in self.tag2index:
            self.tag2index[word] = self.n_tags
            self.tag2count[word] = 1
            self.index2tag[self.n_tags] = word
            self.n_tags += 1
        else:
            self.tag2count[word] += 1