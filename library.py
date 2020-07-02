import chessModel
from collections import OrderedDict
import numpy as np
from collections import defaultdict
from enum import Enum


def convertMoves(moves):
    moveLabels = chessModel.create_uci_labels()
    labelDict = OrderedDict([(i, 0) for i in moveLabels])

    # Create function to convert one value
    def convert(move, labelDict):
        labels = labelDict.copy()
        labels[move] = 1
        d = np.array(list(labels.values())).astype(np.float16)
        return d

    # Apply this to all cells
    values = moves.apply(lambda row: convert(row['Move'], labelDict), axis=1)

    return values.values


def fenToTensor(input):
    '''
     Define valid fen components. An example:
     fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
     a = fenToTensor(fen)
    '''
    pieces_white = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
    pieces_black = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
                    'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12}

    # Split fen string into principal components
    splits = input.split()

    # Prepare tensor
    tensor = np.int8(np.zeros((12, 8, 8)))
    rownr = colnr = 0

    # If black has to move, turn the board
    if splits[1] == "b":
        pieces_white = pieces_black

    # Parse the string
    for i, c in enumerate(splits[0]):
        if c in pieces_white:
            tensor[pieces_white[c]-1, rownr, colnr] = 1
            colnr += 1
        elif c == '/':  # new row
            rownr += 1
            colnr = 0
        elif c.isdigit():
            colnr = colnr + int(c)
        else:
            raise ValueError("invalid fen string")

    return tensor


def createMoveDict():
    # Create dictionary, label as a key, numpy array as a value
    moveLabels = chessModel.create_uci_labels()
    labelDict = OrderedDict([(i, np.float16(0.0)) for i in moveLabels])

    def convertMoves(move, labels):
        labels[move] = np.float16(1.0)
        moveTensor = np.array(list(labels.values()))
        labels[move] = np.float16(0.0)
        return moveTensor

    moveDict = {i: convertMoves(i, labelDict) for i in moveLabels}
    finalDict = defaultdict(lambda: list(labelDict.values()), moveDict)

    enumDict = Enum('Move', moveLabels, start=0)

    return finalDict, enumDict
