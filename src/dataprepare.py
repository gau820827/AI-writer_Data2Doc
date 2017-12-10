"""This is the module for preparing data."""
from preprocessing import readfile
from settings import file_loc


class Lang:
    """A class to summarize encoding information.

    This class will build three dicts:
    word2index, word2count, and index2word for
    embedding information. Once a set of data is
    encoded, we can transform it to corrsponding
    indexing use the word2index, and map it back
    using index2word.

    Attributes:
        word2index: A dict mapping word to index.
        word2count: A dict mapping word to counts in the corpus.
        index2word: A dict mapping index to word.

    """

    def __init__(self, name):
        """Init Lang with a name."""
        self.name = name
        self.word2index = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}
        self.word2count = {"<EOS>": 0}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: "<UNK>"}
        self.n_words = 4  # Count SOS and EOS

    def addword(self, word):
        """Add a word to the dict."""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# embedding(r.t) embedding(r.e) embedding(r.m)
# embedding_dim = 600
# Linear([r.t, r.e, r.m], embedding_dim)

def readLang(data_set):
    """The function to wrap up a data_set.

    This funtion will wrap up a extracted data_set
    into word2index inforamtion. The data set should
    be a list of tuples containing ([triplets], summarize).

    Args:
        data_set: A list of tuples containig 2 items.

    Returns:
        4 Langs for (r.t, r.e, r.m, 'summary')
        (ex. AST, 'Al Hortford', 10, 'summary')

    """
    rt = Lang('rt')
    re = Lang('re')
    rm = Lang('rm')
    summarize = Lang('summarize')

    for v in data_set:
        for triplet in v[0]:
            rt.addword(triplet[0])
            re.addword(triplet[1])
            rm.addword(triplet[2])
        for word in v[1]:
            summarize.addword(word)

    return rt, re, rm, summarize


def loaddata(data_dir, mode='train'):
    """The function for loading data.

    This function will load the data, and then turns it into
    Lang Object.

    Args:
        data_dir: A string indicates the location of data set
        mode: A string indicates to load train, valid, or test.

    Returns:
        A list of reading dataset and a dictionary of Langs
    """
    data_set = readfile(data_dir + mode + '.json')

    rt, re, rm, summary = readLang(data_set)

    print("Read %s data" % mode)
    print("Read %s box score summary" % len(data_set))
    print("Embedding size of (r.t, r.e, r.m) and summary:")
    print("({}, {}, {}), {}".format(rt.n_words, re.n_words,
                                    rm.n_words, summary.n_words))

    langs = {'rt': rt, 're': re, 'rm': rm, 'summary': summary}
    return data_set, langs


def data2index(data_set, langs):
    """The function for indexing the data.

    This function will extending the dataset applying
    the given langs.

    Args:
        dataset: A list which read from preprocessing.readfile()
        langs: A dictionary of Langs containing rt, re, rm, and summary

    Returns:
        A list, the orginal dataset appending with idx_triplets and idx_summary

    """
    # A helper function for indexing
    def findword2index(lang, word):
        try:
            return lang.word2index[word]
        except KeyError:
            return lang.word2index['<UNK>']

    # Extend the dataset with indexing
    for i in range(len(data_set)):
        idx_triplets = []
        for triplet in data_set[i][0]:
            idx_triplet = [None, None, None]
            idx_triplet[0] = findword2index(langs['rt'], triplet[0])
            idx_triplet[1] = findword2index(langs['re'], triplet[1])
            idx_triplet[2] = findword2index(langs['rm'], triplet[2])
            idx_triplets.append(tuple(idx_triplet))

        idx_summary = []
        for word in data_set[i][1]:
            idx_summary.append(findword2index(langs['summary'], word))
        idx_summary.append(1)   # Append 'EOS' at the end

        data_set[i].append([idx_triplets] + [idx_summary])

    return data_set

if __name__ == '__main__':
    train_data, train_lang = loaddata(file_loc, 'train')
    valid_data, _ = loaddata(file_loc, 'valid')
    test_data, _ = loaddata(file_loc, 'test')
    train_data = data2index(train_data, train_lang)
    valid_data = data2index(valid_data, train_lang)
    test_data = data2index(test_data, train_lang)
