"""Preprocessing the data."""
import random
import json
from pprint import pprint

from settings import file_loc


def doc2vec(doc):
    """The function to extract information to triplets.

    Extract box_score informations to the format (r.t, r.e, r.m),
    for example, (AST, 'Al Hortford', 10)

    Args:
        doc: A dict loaded from the json file

    Return:
        triplets: A list contains all triplet vectors extracting from
        the doc.

    """
    triplets = []

    # A helper funtion to make triplet
    def maketriplets(doc, key, ignore, title):
        new_triplets = []
        for _type, _type_dic in doc[key].items():

            # ignore name column
            if _type in ignore:
                continue

            # Work Around QQ
            if key == 'box_score':
                for num, value in _type_dic.items():
                    entity = doc[key][title][num]
                    new_triplets.append((_type, entity, value))
            else:
                entity = doc[key][title]
                new_triplets.append((_type, entity, _type_dic))

        return new_triplets

    for k, v in doc.items():
        if k in ['vis_line', 'home_line']:
            ignore = ['TEAM-NAME']
            title = 'TEAM-NAME'
            new_triplets = maketriplets(doc, k, ignore, title)
            triplets += new_triplets

        elif k == 'box_score':
            ignore = ['FIRST_NAME', 'SECOND_NAME', 'PLAYER_NAME']
            title = 'PLAYER_NAME'
            new_triplets = maketriplets(doc, k, ignore, title)
            triplets += new_triplets

        # Home or Away
        else:
            if 'name' in k:
                new_triplets = [('name', k, v)]
            elif 'city' in k:
                new_triplets = [('city', k, v)]
            triplets += new_triplets

    return triplets


def readfile(filename):
    """The function to prepare data.

    Read the json file into vectors, and then wrap it into

    Args:
        filename: A string indicates which json file to read.

    Return:
        A list of the extracted file vectors

    """
    result = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for d in data:
            d['summary'].append('<EOS>')
            result.append([doc2vec(d), d['summary']])
    return result


def data_iter(source, batch_size=32):
    """The iterator to give batch data while training.

    Args:
        source: the source file to batchify
        batch_size: the batch_size

    Return:
        A generator to yeild batch from random order.
        Will start another random epoch while one epoch
        finished.
    """
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            start = 0   # Start another epoch.
            random.shuffle(order)
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield batch


def main():
    """A minitest function."""
    train_set = readfile(file_loc + 'train.json')
    valid_set = readfile(file_loc + 'valid.json')
    test_set = readfile(file_loc + 'test.json')
    train_iter = data_iter(train_set)
    valid_iter = data_iter(valid_set)
    test_iter = data_iter(test_set)

if __name__ == '__main__':
    main()
