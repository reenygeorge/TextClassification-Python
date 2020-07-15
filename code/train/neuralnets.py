import numpy as np
import pandas as pd
import time
import os
import sys
from tqdm import tqdm
import tensorflow as tf

# insert the path to the common folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))
import constants
from constants import BASE_DIR, MISC_DIR
import preproc

def load_glovevector():
    vector_dir = BASE_DIR / "WordVectors"
    glove_file = vector_dir / "glove.6B.300d.txt"
    df = pd.read_csv(glove_file, sep=" ", quoting=3, header=None, index_col=0)
    glove = {key: val.values for key, val in df.T.items()}
    return glove

def build_embeddingmatrix(word_dict, lemma_dict):
    """ Build embedding matrix. Embedding matrix returns the word vector (300 values) for each word
        in the vocabulary.

        Args:
            word_dict (dictionary): dictionary, where keys are unique words in text_list, and values are
                                   number index to the word
            lemma_dict (dictionary): dictionary, where keys are unique words in text_list, and values are
                                     lemma of the corresponding word

        Returns:
            embedding_matrix (matrix): A 2D matrix, where each row provides the word vector (300 values)
            for the corresponding word. Embedding matrix will have as many rows as there are words in word_dict.
            Each row number of embedding matrix corresponds to word index. i.e row 1 is equivalent to word 1 in word_dict
            num_words (int): total number of words
    """

    start_time = time.time()
    glove_vector = load_glovevector()
    max_edit_dist = 2  #maximum edit distance per dictionary precalculation
    prefix_length = 7
    symspellobj = preproc.symspell_dict(max_edit_dist, prefix_length)
    embed_size = 300
    num_words = len(word_dict) + 1
    # embedding matrix is a 2D matrix, where each row is a word of 300 values (word vector has 300 values
    # for representing each word)
    embedding_matrix = np.zeros((num_words, embed_size), dtype=np.float32)
    unknown_vector = np.zeros((embed_size, ), dtype=np.float32) - 1.
    numwords_noembedding = 0
    words_noembedding = []

    for key in tqdm(word_dict):
        word = key
        embedding_vector = glove_vector.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue

        word = key.lower()
        embedding_vector = glove_vector.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue

        word = key.upper()
        embedding_vector = glove_vector.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue

        word = key.capitalize()
        embedding_vector = glove_vector.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue

        word = lemma_dict[key]
        embedding_vector = glove_vector.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue

        if(len(key) > 1):
            word = preproc.correction(symspellobj, word)
            embedding_vector = glove_vector.get(word)
            if embedding_vector is not None:
                embedding_matrix[word_dict[key]] = embedding_vector
                continue

        embedding_matrix[word_dict[key]] = unknown_vector
        words_noembedding.append(key)
        numwords_noembedding = numwords_noembedding + 1

    fileName = MISC_DIR / "NoEmbeddingWords.txt"

    with open(fileName, 'w') as f:
        f.write("\n".join(str(item.encode("utf-8")) for item in words_noembedding))
    print('numWordsNoEmbedding', numwords_noembedding)
    print("--- %s seconds ---" % (time.time() - start_time))
    return embedding_matrix, num_words

def get_embedding_layer(embedding_matrix, name, max_len, trainable = False):
    embedding_layer = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],
                                               output_dim=embedding_matrix.shape[1],
                                               input_length=max_len, weights=[embedding_matrix],
                                               trainable=trainable, name=name)
    return embedding_layer

def get_conv_pool(x_input, max_length, suffix, kernel_sizes = [1,2,3,4,5], feature_maps = 100):
    branches = []
    for n in kernel_sizes:
        branch = tf.keras.layers.Conv1D(filters=feature_maps, kernel_size=n,
                                        activation='relu', name='Conv_' + suffix + '_' + str(n))(x_input)
        branch = tf.keras.layers.MaxPooling1D(pool_size=max_length - n + 1, strides=None,
                                              padding='valid', name='MaxPooling_' + suffix + '_' + str(n))(branch)
        branch = tf.keras.layers.Flatten(name='Flatten_' + '_' + str(n))(branch)
        branches.append(branch)

    return branches

def build_CNN_pretrained_embeddings(embedding_layer, num_targets, max_length):
    #connect the input with the embedding Layer
    i = tf.keras.Input(shape=(max_length,), dtype='int32', name='main_input')
    x = embedding_layer(i)

    #generate several branches in the network, each for a different convolution + pooling operation
    #and concatenate the result of each branch into a single vector
    branches = get_conv_pool(x, max_length, 'static')
    z = tf.keras.layers.concatenate(branches, axis = 1)

    # pass the concatenated vector to the prediction layer
    o = tf.keras.layers.Dense(num_targets, activation='softmax', name='output')(z)

    model = tf.keras.Model(inputs=i, outputs=o)
    model.compile(loss={'output': 'sparse_categorical_crossentropy'}, optimizer='adam', metrics=['accuracy'])
    return model

