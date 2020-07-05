import keras
import numpy as np


def load_file(file):
    with open(file, 'rb') as f:
        state_list = np.load(f)
        move_list = np.load(f)
        result_list = np.load(f)
    return state_list, move_list, result_list


def find_index(lens, index):
    return next(x for x, val in enumerate(lens) if index < val)


class SimpleFeeder:

    def __init__(self, files, batch_size=32, files_per_batch=2, train=0.8):
        self.files = files
        self.files_per_batch = files_per_batch
        self.sizes = []
        for i, file in enumerate(files):
            state_list, move_list, result_list = load_file(file)
            if i == 0:
                self.Xdim = state_list.shape[1:]
                self.y1dim = move_list.shape[1:]
                self.y2dim = result_list.shape[1:]
            else:
                assert self.Xdim == state_list.shape[1:]
                assert self.y1dim == move_list.shape[1:]
                assert self.y2dim == result_list.shape[1:]
            self.sizes.append(len(state_list))

        self.trains = [int(train * size) for size in self.sizes]
        self.validations = [size - int(train * size) for size in self.sizes]

        self.batch_size = batch_size
        # self.size = sum(self.sizes)
        self.train_len = sum(self.trains)
        self.validation_len = sum(self.validations)
        self.file_set_index = -1
        self.file_set = None

        self.train_seq = self.InnerSequencer(self, self.Xdim, self.y1dim, self.y2dim, batch_size, self.trains,
                                             self.train_len, self.get_train_fileset)
        self.val_seq = self.InnerSequencer(self, self.Xdim, self.y1dim, self.y2dim, batch_size, self.validations,
                                           self.validation_len, self.get_validation_fileset)

    def __len__(self):
        return self.size

    def update_shuffle_map(self):
        # print("!!!! update shuffle map !!!!")
        self.train_shuffle_maps = [np.random.permutation(size) for size in self.trains]
        self.validation_shuffle_maps = [np.random.permutation(size) + self.trains[i] for i, size in
                                        enumerate(self.validations)]

        files_shuffle = np.random.permutation(len(self.files))
        self.file_sets = [files_shuffle[i:i + self.files_per_batch] for i in
                          range(0, len(files_shuffle), self.files_per_batch)]
        self.file_sets_trains = [sum(map(lambda idx: self.trains[idx], file_set)) for file_set in self.file_sets]
        self.file_sets_validations = [sum(map(lambda idx: self.validations[idx], file_set)) for file_set in
                                      self.file_sets]

        self.file_sets_trains_cum = np.cumsum(self.file_sets_trains)
        self.file_sets_validations_cum = np.cumsum(self.file_sets_validations)

        print(self.file_sets)

    def get_train(self):
        return self.train_seq

    def get_validation(self):
        return self.val_seq

    def get_file_set(self, index, limits, shuffle_maps):
        if self.file_set_index != index:
            del self.file_set
            self.file_set = self.MultiFilesAccessor([self.files[file] for file in self.file_sets[index]],
                                                    [limits[file] for file in self.file_sets[index]],
                                                    [shuffle_maps[file] for file in self.file_sets[index]])
        return self.file_set

    def get_train_fileset(self, index):
        '''

        :param index: absolute index in the combined set
        :return:
                fileset  train fileset
                index offset for the returning fileset
        '''
        file_set_index = find_index(self.file_sets_trains_cum, index)
        return self.get_file_set(file_set_index, self.trains, self.train_shuffle_maps), self.file_sets_trains_cum[
            file_set_index - 1] if file_set_index > 0 else 0

    def get_validation_fileset(self, index):
        '''

        :param index: absolute index in the combined set
        :return:
                fileset  validation fileset
                index offset for the returning fileset
        '''
        file_set_index = find_index(self.file_sets_validations_cum, index)
        return self.get_file_set(file_set_index, self.validations, self.validation_shuffle_maps), \
               self.file_sets_validations_cum[file_set_index - 1] if file_set_index > 0 else 0

    class InnerSequencer(keras.utils.Sequence):
        def __init__(self, outer, Xdim, y1dim, y2dim, batch_size, limits, len, file_set_getter):
            self.outer = outer
            self.batch_size = batch_size
            self.Xdim = Xdim
            self.y1dim = y1dim
            self.y2dim = y2dim
            self.len = len
            self.limits = limits
            self.file_set_getter = file_set_getter

        def update_step(self, shuffle_maps):
            self.shuffle_maps = shuffle_maps

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(self.len / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'

            X = np.empty((self.batch_size, *self.Xdim))
            y1 = np.empty((self.batch_size, *self.y1dim))
            y2 = np.empty((self.batch_size, *self.y2dim))

            for i, j in enumerate(range(index * self.batch_size,
                                        min((index + 1) * self.batch_size, self.len))):
                file_set, offset = self.file_set_getter(j)
                print(i, " ", j, " ", offset, " ", j - offset)
                item = file_set[j - offset]
                X[i] = item[0]
                y1[i] = item[1]
                y2[i] = item[2]

            return X, (y1, y2)

        def on_epoch_end(self):
            self.outer.update_shuffle_map()

    class MultiFilesAccessor:
        def __init__(self, files, limits, shuffle_maps):
            assert len(files) == len(limits)
            assert len(files) == len(shuffle_maps)

            self.files = [load_file(file) for file in files]
            self.limits = limits
            self.cum_lens = np.cumsum(self.limits)
            self.shuffle_maps = shuffle_maps

        def __len__(self):
            return sum(self.limits)

        def __getitem__(self, item):
            file_index = find_index(self.cum_lens, item)
            index = self.shuffle_maps[file_index][item]
            return self.files[file_index][0][index], self.files[file_index][1][index], self.files[file_index][2][index]

    class MultiFilesAccessor2:
        def __init__(self, files):
            self.files = files
            self.data = [load_file(file.name) for file in files]
            self.train_cum_counts = np.cumsum([file.trains for file in files])
            self.validation_cum_counts = np.cumsum([file.trains for file in files])
            self.train_shuffle_map = np.random.permutation(self.train_cum_counts[-1])

        def train_len(self):
            return self.train_cum_counts[-1]

        def validation_len(self):
            return self.validation_cum_counts[-1]

        def get_train(self, idx):
            index = self.train_shuffle_map[idx]
            file_index = find_index(self.train_cum_counts, index)
            file = self.data[file_index]
            item_index = self.train_cum_counts[file_index - 1] if file_index > 0 else 0
            return file[0][item_index], file[1][item_index], file[2][item_index]

        def get_validation(self, index):
            file_index = find_index(self.validation_cum_counts, index)
            file = self.data[file_index]
            item_index = (self.validation_cum_counts[file_index - 1] if file_index > 0 else 0) + file.trains
            return file[0][item_index], file[1][item_index], file[2][item_index]

    class DataFile:
        def __init__(self, name, train=0.8):
            self.name = name
            state_list, move_list, result_list = load_file(name)

            self.len = len(state_list)
            assert len(move_list) == self.len
            assert len(result_list) == self.len

            self.trains = int(train * self.len)
            self.validations = self.len - self.trains

            self.Xdim = state_list.shape[1:]
            self.y1dim = move_list.shape[1:]
            self.y2dim = result_list.shape[1:]
