import os
import sys
import gc
import spacy
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

# insert the path to the common folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))
from loaddata import load_traintestdata
import preproc
from neuralnets import build_embeddingmatrix, get_embedding_layer, build_CNN_pretrained_embeddings
from constants import USE_PICKLED_FILES, MODELS_DIR


def preprocess(txt):
    """ Clean and preprocess the text

        Args:
            txt (str): text data before pre-processing

        Returns:
            txt (str): text data after pre-processing
    """
    txt = txt.lower()
    txt = preproc.replace_contractions(txt)
    txt = preproc.clean_puncts(txt)
    txt = preproc.remove_linebreaks(txt)
    txt = preproc.clean_numbers(txt)
    return (txt)

def tokenize_text(text_list, num_traindata, max_length):
    """ Tokenizes the text, returns text converted to numbers along with word and lemma dictionary

        Args:
            text_list (str): Concatenated training and test data

        Returns:
            word_dict (dict) : dictionary, where keys are unique words in text_list, and values are
                               number index to the word
            lemma_dict (dict): dictionary, where keys are unique words in text_list, and values are lemma of the
                        corresponding word
            word_sequences (list): Text_list converted to number format. Each word in text_list is replaced with its
                            corresponding index value from word_dict
    """
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'tagger'])
    nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)
    word_dict = {}
    word_index = 1
    lemma_dict = {}

    docs = nlp.pipe(text_list, n_threads=2)
    word_sequences = []
    word_dict["UNKNOWNWORD"] = word_index
    lemma_dict["UNKNOWNWORD"] = "UNKNOWNWORD"
    word_index += 1

    for doc in tqdm(docs):
        word_seq = []
        for token in doc:
            if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):
                word_dict[token.text] = word_index
                word_index += 1
                lemma_dict[token.text] = token.lemma_
            if token.pos_ is not "PUNCT":
                word_seq.append(word_dict[token.text])
        word_sequences.append(word_seq)
    del docs
    gc.collect()
    return word_sequences, word_dict, lemma_dict


def main():
    #1. Load train and test data
    train_df, test_df = load_traintestdata()
    num_traintargets = train_df['target'].nunique()
    print(train_df.head())

    #2. Pre-process the data
    train_df['text'] = train_df['text'].map(preprocess)
    test_df['text'] = test_df['text'].map(preprocess)

    if not USE_PICKLED_FILES:
        #3. Tokenize and process the text data (convert into word indices)
        train_text = train_df['text']
        test_text = test_df['text']
        y_train = train_df['target']
        y_test = test_df['target']
        text_list = pd.concat([train_text, test_text])
        num_traindata = train_df.shape[0]
        max_length = 500
        word_sequences, word_dict, lemma_dict = tokenize_text(text_list, num_traindata, max_length)

        #4. Build embedding matrix
        embedding_matrix_glove, nb_words = build_embeddingmatrix(word_dict, lemma_dict)

        train_word_sequences = word_sequences[:num_traindata]
        test_word_sequences = word_sequences[num_traindata:]

        max_length = 500

        train_word_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_word_sequences, maxlen=max_length,
                                                                       padding='post')

        test_word_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_word_sequences, maxlen=max_length,
                                                                      padding='post')
        embedding_layer = get_embedding_layer(embedding_matrix_glove, 'embedding_layer_static', max_length, trainable=True)
        model = build_CNN_pretrained_embeddings(embedding_layer, num_traintargets, max_length)
        print(type(y_train))
        target = y_train.to_numpy(dtype=int)

        model.fit(train_word_sequences, target, batch_size=512, epochs=5, verbose=2)
        #jsonFile = MODELS_DIR / "model_epoc5_tfkeras_cnn.json"
        #weightsFile = MODELS_DIR / "model_epoc5_tfkeras_cnn.h5"
        #saveModel(model, jsonFile, weightsFile)
        #loadedModel = loadModel(jsonFile, weightsFile)
        #loadedModel.compile(loss={'output': 'sparse_categorical_crossentropy'}, optimizer='adam', metrics=['accuracy'])
        #loss, accuracy = loadedModel.evaluate(test_word_sequences, y_test.to_numpy(), verbose=0)
        loss, accuracy = model.evaluate(test_word_sequences, y_test.to_numpy(), verbose=0)
        print(accuracy)
    else:
        print("use pickled files")


if __name__ == "__main__":
    main()
