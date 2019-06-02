from sklearn.model_selection import StratifiedShuffleSplit
from scipy.sparse import csr_matrix

import numpy as np
import os
from collections import Counter
import tensorflow as tf
import datetime
import inspect
import gc

# DEFAULT SETTINGS
_NOW = datetime.datetime.now()
_DATETIME = str(_NOW.year) + '-' + str(_NOW.day) + '-' + str(_NOW.month) + '-' + str(_NOW.hour) + '-' + str(_NOW.minute) + '-' + str(_NOW.second)
_LOG = 'log'
_RANDOM_SEED = 6789


def load_dataset(list_arch_os, dataset_name, make_it_imbalanced=True):
    X_full = []
    Y_full = np.array([])
    data_folder = 'datasets'
    for arch_os in list_arch_os:
        decimal_functions_path = data_folder + '/' + dataset_name + '/vocab2id_opcode-' + arch_os + '.data'
        label_path = data_folder + '/' + dataset_name + '/labels-' + arch_os + '.data'
        with open(decimal_functions_path, 'r') as f:
            X_lines = f.readlines()
        with open(label_path, 'r') as f:
            Y_lines = f.readlines()

        Y = np.array([int(number) for number in Y_lines[0].split()])
        X_full += X_lines
        Y_full = np.concatenate((Y_full, Y), axis=0)

    if dataset_name == 'NDSS18' and make_it_imbalanced:
        print('Making NDSS18 imbalanced ...')
        n_vul = np.sum(Y_full)
        n_non_vul = len(Y_full) - n_vul
        imbalanced_ratio = 1 / 50  # its means: vul:non-vul = 1:50
        n_vul_new = int(n_non_vul * imbalanced_ratio)

        imbalanced_X = []
        imbalanced_y = []
        index = 0
        for id, opcode_assembly_code in enumerate(X_full):
            if Y_full[index] == 1.0:  # vulnerable function
                if sum(imbalanced_y) < n_vul_new:
                    if opcode_assembly_code != '-----\n':
                        imbalanced_X.append(opcode_assembly_code)
                    else:  # opcode_assembly_code == '-----\n'
                        index += 1
                        imbalanced_X.append(opcode_assembly_code)  # also add '-----\n'
                        imbalanced_y.append(1)
                else:
                    if opcode_assembly_code == '-----\n':
                        index += 1
            elif Y_full[index] == 0.0:
                if opcode_assembly_code != '-----\n':
                    imbalanced_X.append(opcode_assembly_code)
                else:  # opcode_assembly_code == '-----\n':
                    index += 1
                    imbalanced_X.append(opcode_assembly_code)  # also add '-----\n'
                    imbalanced_y.append(0)
        X_full = imbalanced_X
        Y_full = np.asarray(imbalanced_y)

    # process opcodes and assembly code (note that assembly code is the instruction information of the paper)
    if dataset_name == 'NDSS18':
        X_opcode, X_assembly, sequence_length, max_length, vocab_opcode_size = NDSS18_process_opcode_assembly_code(X_full)
    else:
        max_length = 300
        X_opcode, X_assembly, sequence_length, vocab_opcode_size = six_projects_process_opcode_assembly_code(X_full, max_length)

    del X_full
    gc.collect()

    X_opcode = np.asarray(X_opcode)
    X_assembly = np.asarray(X_assembly)

    test_set_ratio = 0.1  # it means train:valid:test.txt = 8:1:1
    train_valid_index, test_index = split_by_ratio(X_opcode, Y_full, test_size=test_set_ratio)
    train_index, valid_index = split_by_ratio(X_opcode[train_valid_index],
                                              Y_full[train_valid_index],
                                              test_size=test_set_ratio / (1 - test_set_ratio))

    x_train_opcode = X_opcode[train_valid_index][train_index]
    x_valid_opcode = X_opcode[train_valid_index][valid_index]
    x_test_opcode = X_opcode[test_index]

    del X_opcode
    gc.collect()

    x_train_assembly = X_assembly[train_valid_index][train_index]
    x_valid_assembly = X_assembly[train_valid_index][valid_index]
    x_test_assembly = X_assembly[test_index]

    del X_assembly
    gc.collect()

    x_train_seq_len = sequence_length[train_valid_index][train_index]
    x_valid_seq_len = sequence_length[train_valid_index][valid_index]
    x_test_seq_len = sequence_length[test_index]

    y_train = Y_full[train_valid_index][train_index]
    y_valid = Y_full[train_valid_index][valid_index]
    y_test = Y_full[test_index]

    message = 'x_train (opcode & assembly) {}\n'.format(x_train_opcode.shape)
    message += 'y_train {}\n'.format(y_train.shape)

    message += 'x_valid (opcode & assembly) {}\n'.format(x_valid_opcode.shape)
    message += 'y_valid {}\n'.format(y_valid.shape)

    message += 'x_test (opcode & assembly) {}\n'.format(x_test_opcode.shape)
    message += 'y_test {}\n'.format(y_test.shape)

    message += 'max-length {}\n'.format(max_length)
    message += 'vocab_opcode_size {}\n'.format(vocab_opcode_size)

    print_and_write_logging_file(dir=_LOG, txt=message, running_mode=1)

    return x_train_opcode, x_train_assembly, x_train_seq_len, y_train, \
    x_valid_opcode, x_valid_assembly, x_valid_seq_len, y_valid, \
    x_test_opcode, x_test_assembly, x_test_seq_len, y_test, max_length, vocab_opcode_size


def split_by_ratio(X, y, random_seed=_RANDOM_SEED, test_size=0.1):
    shufflesplit = StratifiedShuffleSplit(n_splits=1, random_state=random_seed, test_size=test_size)
    train_index, test_index = next(shufflesplit.split(X, y, groups=y))
    return train_index, test_index


def make_batches(size, batch_size):
    # returns a list of batch indices (tuples of indices).
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def build_vocab(words):
    dictionary = dict()
    count = []
    count.extend(Counter(words).most_common())
    index = 0

    for word, occurs in count:
        dictionary[word] = index
        index += 1

    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary


def create_one_hot_vector_for_opcode(aa, dic_id_opcode, all_zeros=False):
    bb = np.zeros(len(dic_id_opcode))
    if all_zeros:
        return bb
    else:
        bb[dic_id_opcode[aa]] = 1
        return bb


def create_one_hot_vector_for_assembly(list_tuple=[], all_zeros=False):
    bb = np.zeros(256)
    if all_zeros:  # for padding
        return bb
    else:
        # count on each line of function, and assign at index the value of num_occurs
        for tuple_hex_times in list_tuple:
            decimal = int(tuple_hex_times[0])
            n_occures = tuple_hex_times[1]
            bb[decimal] = n_occures

        return bb


def convert_to_one_hot(list_function_opcode, list_function_assembly_code, dic_opcode, max_length):
    # process opcode
    function_opcode_one_hot = []
    for function_opcode in list_function_opcode:

        opcode_one_hot = []
        for opcode in function_opcode:
            one_hex = create_one_hot_vector_for_opcode(opcode, dic_opcode)
            opcode_one_hot.append(one_hex)

        while len(opcode_one_hot) < max_length:
            opcode_one_hot.append(create_one_hot_vector_for_opcode(opcode, dic_opcode, all_zeros=True))

        function_opcode_one_hot.append(csr_matrix(opcode_one_hot))

    function_opcode_one_hot = np.asarray(function_opcode_one_hot)

    # process assembly
    function_assembly_one_hot = []
    for function_assembly in list_function_assembly_code:

        assembly_one_hot = []
        list_tuple = []
        for list_hex in function_assembly:
            list_tuple.extend(Counter(list_hex).most_common())
            one_line = create_one_hot_vector_for_assembly(list_tuple)
            assembly_one_hot.append(one_line)
            list_tuple = []

        while len(assembly_one_hot) < max_length:
            assembly_one_hot.append(create_one_hot_vector_for_assembly(all_zeros=True))

        function_assembly_one_hot.append(csr_matrix(assembly_one_hot))

    function_assembly_one_hot = np.asarray(function_assembly_one_hot)

    return function_opcode_one_hot, function_assembly_one_hot


def NDSS18_process_opcode_assembly_code(raw_X):
    list_function_opcode = []
    list_function_assembly_code = []
    words_opcode = []

    list_opcode = []
    list_assembly_code = []
    max_length = -1
    length = 0
    sequence_length = np.array([]).astype(int) # actual sequence_length of each function
    for id, opcode_assembly_code in enumerate(raw_X):
        if opcode_assembly_code != '-----\n':
            opcode_assembly_code = opcode_assembly_code[:-1]
            opcode_assembly_code_split = opcode_assembly_code.split('|')
            if len(opcode_assembly_code_split) == 2: # opcode has 1 byte
                opcode = opcode_assembly_code_split[0]
                list_hex_code = opcode_assembly_code_split[1]
            else:
                opcode = ' '.join(opcode_assembly_code_split[:-1])
                list_hex_code = opcode_assembly_code_split[-1]
            list_opcode.append(opcode)
            words_opcode.append(opcode)
            list_assembly_code.append(list_hex_code.split(','))

            length += 1
        else:
            list_function_opcode.append(list_opcode)
            list_function_assembly_code.append(list_assembly_code)
            list_opcode = []
            list_assembly_code = []

            if length > max_length:
                max_length = length

            sequence_length = np.append(sequence_length, length)
            length = 0

    dictionary_index, index_dictionary = build_vocab(words_opcode)

    function_opcode_one_hot, function_assembly_one_hot = convert_to_one_hot(list_function_opcode, list_function_assembly_code, dictionary_index, max_length)
    return function_opcode_one_hot, function_assembly_one_hot, sequence_length, max_length, len(dictionary_index)


def six_projects_process_opcode_assembly_code(raw_X, max_length=190):
    list_function_opcode = []
    list_function_assembly_code = []
    words_opcode = []

    list_opcode = []
    list_assembly_code = []
    length = 0
    sequence_length = np.array([]).astype(int) # actual sequence_length of each function
    for id, opcode_assembly_code in enumerate(raw_X):
        if opcode_assembly_code != '-----\n':
            opcode_assembly_code = opcode_assembly_code[:-1]
            opcode_assembly_code_split = opcode_assembly_code.split('|')
            if len(opcode_assembly_code_split) == 2: # opcode has 1 byte
                opcode = opcode_assembly_code_split[0]
                list_hex_code = opcode_assembly_code_split[1]
            else:
                opcode = ' '.join(opcode_assembly_code_split[:-1])
                list_hex_code = opcode_assembly_code_split[-1]

            length += 1

            if length <= max_length:
                list_opcode.append(opcode)
                words_opcode.append(opcode)
                list_assembly_code.append(list_hex_code.split(','))

                length_cut_by_max_length = length

        else:
            list_function_opcode.append(list_opcode)
            list_function_assembly_code.append(list_assembly_code)
            list_opcode = []
            list_assembly_code = []

            sequence_length = np.append(sequence_length, length_cut_by_max_length)
            length = 0

    dictionary_index, index_dictionary = build_vocab(words_opcode)

    function_opcode_one_hot, function_assembly_one_hot = convert_to_one_hot(list_function_opcode, list_function_assembly_code, dictionary_index, max_length)
    return function_opcode_one_hot, function_assembly_one_hot, sequence_length, len(dictionary_index)


def get_default_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    return tf_config


def print_and_write_logging_file(dir, txt, running_mode, get_datetime_from_training=_DATETIME, show_message=True):
    if show_message:
        print(txt[:-1])

    if running_mode == 1:
        with open(os.path.join(dir, 'training_log_' + _DATETIME + '.txt'), 'a') as f:
            f.write(txt)
    elif running_mode == 0:
        with open(os.path.join(dir, 'testing_log_' + get_datetime_from_training + '.txt'), 'a') as f:
            f.write(txt)
    else:
        with open(os.path.join(dir, 'visualization_log_' + get_datetime_from_training + '.txt'), 'a') as f:
            f.write(txt)


def save_all_params(class_object):
    attributes = inspect.getmembers(class_object, lambda a: not (inspect.isroutine(a)))
    list_params = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
    message = 'List parameters '
    message += '{\n'
    for params in list_params:
        try:
            message += '\t' + str(params[0]) + ': ' + str(params[1]) + '\n'
        except:
            continue
    message += '}\n'
    if class_object.running_mode == 1:
        message += "Start training process.\n"
        message += "Start pre-processing data.\n"
    elif class_object.running_mode == 0:
        message += "Start testing process.\n"
        message += "Start pre-processing data.\n"
    message += "-----------------------------------------------------\n"

    make_dir(_LOG)
    print_and_write_logging_file(_LOG, message, class_object.running_mode)


def convert_list_sparse_to_dense(X):
    dense_matrix = []
    for one_function in X:
        dense_matrix.append(one_function.toarray())
    return np.asarray(dense_matrix)
