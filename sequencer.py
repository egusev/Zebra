import keras
import numpy as np


def load_file(file):
    with open(file, 'rb') as f:
        state_list = np.load(f)
        move_list = np.load(f)
        result_list = np.load(f)
    return state_list, move_list, result_list


def find_indexes(lens, index):
    file_index = next(x for x, val in enumerate(lens) if index < val)
    item_index = index - (lens[file_index - 1] if file_index > 0 else 0)
    return file_index, item_index


class SimpleFeeder:

    def __init__(self, files, batch_size=32, files_per_batch=2, train=0.8):
        self.files_per_batch = files_per_batch
        self.file_metas = [self.DataFile(file, train) for file in files]

        for i, file_meta in enumerate(self.file_metas):
            if i == 0:
                self.Xdim = file_meta.Xdim
                self.y1dim = file_meta.y1dim
                self.y2dim = file_meta.y2dim
            else:
                assert self.Xdim == file_meta.Xdim
                assert self.y1dim == file_meta.y1dim
                assert self.y2dim == file_meta.y2dim

        self.file_set_name = None
        self.file_set_names = []
        self.file_set_data = []

        self.train_seq = self.InnerSequencer(self,
                                             self.Xdim, self.y1dim, self.y2dim,
                                             batch_size,
                                             self.TrainDataAccessor())

        self.val_seq = self.InnerSequencer(self,
                                           self.Xdim, self.y1dim, self.y2dim,
                                           batch_size,
                                           self.ValidationDataAccessor())

    def get_train(self):
        self.train_seq.on_epoch_end()
        return self.train_seq

    def get_validation(self):
        self.val_seq.on_epoch_end()
        return self.val_seq

    def shuffle_files(self):
        print("!!!! shuffle files!!!!\n")
        files_shuffle = np.random.permutation(len(self.file_metas))
        'list of file sets (metas)'
        file_sets = [list(map(lambda idx: self.file_metas[idx], files_shuffle[i:i + self.files_per_batch]))
                     for i in range(0, len(files_shuffle), self.files_per_batch)]

        self.file_set_names = {''.join(map(lambda data_file: data_file.name, files)): files for files in file_sets}

        self.train_seq.update_file_sets(file_sets)
        self.val_seq.update_file_sets(file_sets)

    def get_data_file(self, file_set_name, file_index):
        if self.file_set_name != file_set_name:
            print("\nexisted file_set_name=", self.file_set_name, " required=", file_set_name, " load new files")

            del self.file_set_data
            self.file_set_name = file_set_name
            file_set = self.file_set_names[file_set_name]
            self.file_set_data = [load_file(file.name) for file in file_set]

        return self.file_set_data[file_index]

    class InnerSequencer(keras.utils.Sequence):
        def __init__(self, outer, Xdim, y1dim, y2dim, batch_size, data_accessor):
            self.outer = outer
            self.batch_size = batch_size
            self.Xdim = Xdim
            self.y1dim = y1dim
            self.y2dim = y2dim
            self.data_accessor = data_accessor
            self.file_sets = []
            'count of batches per file set'
            self.cum_counts = []

        def on_epoch_end(self):
            self.outer.shuffle_files()

        def update_file_sets(self, file_sets):
            self.file_sets = [self.outer.MultiFilesAccessor2(files, self.data_accessor, self.batch_size) for files in file_sets]
            self.cum_counts = np.cumsum(list(map(lambda fs: fs.get_batch_count(), self.file_sets)))
            print(list(map(lambda file_set: file_set.name, self.file_sets)))

        def __len__(self):
            'Denotes the number of batches per epoch'
            return self.cum_counts[-1]

        def __getitem__(self, index):
            'Generate one batch of data'
            assert index < len(self)
            print('\nbatch_index =', index)

            file_set_index, batch_number = find_indexes(self.cum_counts, index)
            file_set = self.file_sets[file_set_index]
            print("file_set_index=", file_set_index, ", name=", file_set.name)

            X = np.empty((self.batch_size, *self.Xdim))
            y1 = np.empty((self.batch_size, *self.y1dim))
            y2 = np.empty((self.batch_size, *self.y2dim))

            items_from = batch_number * self.batch_size
            items_to = min((batch_number + 1) * self.batch_size, len(file_set))
            print("items_from =", items_from, " items_to =", items_to)

            for i, j in enumerate(range(items_from, items_to)):
                file_index, item_index = file_set[j]
                if i < 1:
                    print("i =", i, " j =", j, " file_set_index =", file_set_index,
                          " file_index =", file_index, " item_index =", item_index)

                file = self.outer.get_data_file(file_set.name, file_index)
                X[i] = file[0][item_index]
                y1[i] = file[1][item_index]
                y2[i] = file[2][item_index]

            return X, (y1, y2)

    class MultiFilesAccessor:
        def __init__(self, files, data_accessor, batch_size):
            self.files = files
            self.name = ''.join(map(lambda meta: meta.name, files))
            self.data_accessor = data_accessor
            self.data = [load_file(file.name) for file in files]
            self.cum_count = np.cumsum([data_accessor.get_len(file) for file in files])
            self.shuffle_map = np.random.permutation(self.cum_count[-1])
            self.batch_size = batch_size

        def __len__(self):
            '''returns len in items'''
            return self.cum_count[-1]

        def __str__(self):
            return self.name

        def __getitem__(self, idx):
            index = self.shuffle_map[idx]
            file_index, item_index = find_indexes(self.cum_count, index)
            index_in_file = item_index + self.data_accessor.get_offset(self.files[file_index])
            return file_index, index_in_file

        def get_batch_count(self):
            # floor?
            return int(np.ceil(len(self) / self.batch_size))

    class DataFile:
        def __init__(self, name, train=0.8):
            self.name = name
            state_list, move_list, result_list = load_file(name)

            self.len = len(state_list)
            assert len(move_list) == self.len
            assert len(result_list) == self.len

            print("file=", name," len=", self.len)

            self.trains = int(train * self.len)
            self.validations = self.len - self.trains

            self.Xdim = state_list.shape[1:]
            self.y1dim = move_list.shape[1:]
            self.y2dim = result_list.shape[1:]

    class TrainDataAccessor:
        def get_len(self, data_file):
            return data_file.trains

        def get_offset(self, data_file):
            return 0

        def get_name(self):
            return "train"

    class ValidationDataAccessor:
        def get_len(self, data_file):
            return data_file.validations

        def get_offset(self, data_file):
            return data_file.trains

        def get_name(self):
            return "validation"

