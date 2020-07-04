import keras
import numpy as np


class SimpleFeeder:

    def __init__(self, batch_size=32, train=0.8):
        'Initialization'
        with open('data.npy', 'rb') as f:
            self.stateList3 = np.load(f)
            self.moveList3 = np.load(f)
            self.resultList3 = np.load(f)

        assert len(self.stateList3) == len(self.moveList3)
        assert len(self.stateList3) == len(self.resultList3)

        self.batch_size = batch_size
        self.size = len(self.stateList3)
        self.train_len = int(self.size * train)
        self.train_seq = self.InnerSequencer(self,
                                             self.stateList3.shape[1:],
                                             self.moveList3.shape[1:],
                                             self.resultList3.shape[1:],
                                             batch_size,
                                             0,
                                             self.train_len)
        self.val_seq = self.InnerSequencer(self,
                                           self.stateList3.shape[1:],
                                           self.moveList3.shape[1:],
                                           self.resultList3.shape[1:],
                                           batch_size,
                                           self.train_len,
                                           self.size - self.train_len)
        self.update_shuffle_map()

    def __len__(self):
        return self.size

    def update_shuffle_map(self):
        # print("!!!! update shuffle map !!!!")
        self.shuffle_map = np.random.permutation(self.size)

    def get_item(self, index):
        return self.stateList3[self.shuffle_map[index]], self.moveList3[self.shuffle_map[index]], self.resultList3[
            self.shuffle_map[index]]

    def get_train(self):
        return self.train_seq

    def get_validation(self):
        return self.val_seq

    class InnerSequencer(keras.utils.Sequence):
        def __init__(self, outer, Xdim, y1dim, y2dim, batch_size, offset, len):
            self.outer = outer
            self.batch_size = batch_size
            self.Xdim = Xdim
            self.y1dim = y1dim
            self.y2dim = y2dim
            self.len = len
            self.offset = offset

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(self.len / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch

            X = np.empty((self.batch_size, *self.Xdim))
            y1 = np.empty((self.batch_size, *self.y1dim))
            y2 = np.empty((self.batch_size, *self.y2dim))

            for i, j in enumerate(range(index * self.batch_size + self.offset,
                                        min((index + 1) * self.batch_size, self.len) + self.offset)):
                item = self.outer.get_item(j)
                X[i] = item[0]
                y1[i] = item[1]
                y2[i] = item[2]

            return X, (y1, y2)

        def on_epoch_end(self):
            self.outer.update_shuffle_map()
