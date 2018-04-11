"""Preprocessing the data."""
import random
import json
from pprint import pprint

from settings import file_loc, MAX_PLAYERS
"""
udpate:
Ken
04/01, 2018
1. add MAX_PLAYER = 26 in settings.py
04/04, 2018
1. added end of block at each row
2. aligned player numbers to 26 using PAD
"""

def doc2vec(doc):
    """The function to extract information to triplets.

    Extract box_score informations to the format (r.t, r.e, r.m),
    for example, (AST, 'Al Hortford', 10)

    Args:
        doc: A dict loaded from the json file
           {'box_score': 
               { 'AST': { player_number: value, player_number: value .... } 
                 'BLK': { player_number: value, player_number: value .... } 
                }  
            }
    Return:
        triplets: A list contains all triplet vectors extracting from
        the doc.
    """
    triplets = []

    # A helper funtion to make triplet
    def maketriplets(doc, key, ignore, title):
        """
        Args:
            doc:
                {'box_score': 
                   { 'AST': { player_number: value, player_number: value .... } 
                     'BLK': { player_number: value, player_number: value .... } 
                    }  
                }
        Return:
            A list of triplets indication the box score relationship
            [('TO', 'Ron Baker', 'N/A'), ('FG3A', 'Isaiah Thomas', '13'), ...]
        """
        new_triplets = []
        for _type, _type_dic in doc[key].items():

            # ignore name column
            if _type in ignore:
                continue
            # Work Around QQ
            if key == 'box_score':
                for num, value in _type_dic.items():
                    """
                    key: box_score
                    title: PLAYER_NAME
                    num: player_number, see "Return"
                    entity: string of player name
                    """
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
            else:
                continue
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
    roster = 0
    with open(filename, 'r') as f:
        data = json.load(f)
        for d in data:           
            # # # # # # # # # # # # # 
            # Added by Ken:
            #   data:  a list of dictionary containing each game information
            #      [{}, {}, {} ... {}], {box_score, summary, home, ...}
            #   d: a dictionary of {box_score, summary, ... }
            #     align all block size in columnwise and rowise direction for each 
            #     box score
            d = align_box_score(d)
            #test_box_score_aligned(d)
            d['summary'].append('<EOS>')
            result.append([doc2vec(d), d['summary']])
    return result

def test_box_score_aligned(d):
    """
    Ken: 
        This is a test function to test if box table all aligns with 30 rows
        (30 players)
    """
    tables = 0
    EB_ATTR = 0
    for k in d['box_score']:
        tables = tables + 1
        if 'ENDBLOCK' in d['box_score']:
            EB_ATTR = EB_ATTR + 1
        if len(d['box_score'][k]) != MAX_PLAYERS or tables != EB_ATTR:
            print("Preprocessing Error")
            pprint(d['box_score'][k])
    return

def data_iter(source, batch_size=32, shuffle=True):
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
    # TODO Change to Permute
    order = list(range(dataset_size))
    if shuffle:
        random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            start = 0   # Start another epoch.
            if shuffle:
                random.shuffle(order)
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield batch


def align_box_score(doc):
    """This function aligns number of players in each box over all boxes"""
    # Example:
    #   box1.shape = (players = 20, attributes = MAX_ATTRIBUTE)
    #   box2.shape = (players = 23, attributes = MAX_ATTRIBUTE)
    # Given the maximum number of players = 26 (from https://github.com/harvardnlp/boxscore-data)
    # , and the number of attributes is fixed
    # 
    # Then for each block (rowise and columnwise), we append END_OF_BLOCK
    # Args:
    #   a dictionary of each game information
    #  {home_name, box_score, home_city, vis_name, summary, vis_line, vis_city
    #   day, home_line}
    #
    # Retruns:
    #   box.shape = (players = 30, attributes = MAX_ATTRIBUTE) 
    #
    NULL_VAL = 'N/A'
    END_OF_BLOCK = '<EOB>'
    NULL_ENTITIES = []
    ignore = ['PLAYER_NAME', 'FIRST_NAME', 'SECOND_NAME']
    # add new entity (player) if total players less than MAX_PALYERS
    TEAM_SIZE = len(doc['box_score']['PLAYER_NAME'])
    if TEAM_SIZE < MAX_PLAYERS:
        NULL_ENTITIES = ['<PAD'+str(i)+'>' for i in range(MAX_PLAYERS-TEAM_SIZE)]
        for i in range(MAX_PLAYERS-TEAM_SIZE):
            doc['box_score']['PLAYER_NAME'][NULL_ENTITIES[i]] = NULL_VAL
            doc['box_score']['FIRST_NAME'][NULL_ENTITIES[i]] = NULL_VAL
            doc['box_score']['SECOND_NAME'][NULL_ENTITIES[i]] = NULL_VAL

    # add a new player named ENDBLOCK denoting the ending of block while reading
    doc['box_score']['PLAYER_NAME']['ENDBLOCK'] = END_OF_BLOCK
    doc['box_score']['FIRST_NAME']['ENDBLOCK'] = END_OF_BLOCK
    doc['box_score']['SECOND_NAME']['ENDBLOCK'] = END_OF_BLOCK

    # align PAD player to 30 players, {ATT: {<PAD0>: N/A}}
    for attr,val in doc['box_score'].items():
        # attr = 'FTA', val = {number: value, ...}
        if attr in ignore:
            continue
        if len(val) < MAX_PLAYERS:
            for i in range(MAX_PLAYERS-len(val)):
                val[NULL_ENTITIES[i]] = NULL_VAL
        # add ENDBLOCK at the end of each column
        val['ENDBLOCK'] = END_OF_BLOCK

    # rowise end of blocking adding if reading in row direction
    # Add an ENDBLOCK column as ATTRIBUTE {ENDBLOCK: {player_number: <EOB>}}
    # player_number = [p for p in doc['box_score']['PLAYER_NAME']]
    # doc['box_score']['ENDBLOCK'] = {p:END_OF_BLOCK for p in player_number}
    return doc

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