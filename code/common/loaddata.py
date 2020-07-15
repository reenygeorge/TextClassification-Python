###################################################################################################
# Load required data sets
###################################################################################################

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# if all categories of text from 20 newsgroups is not required, add the required categories here,
# and give this as 'categories' option for fetch_20newsgroups invocation.
cats = ['alt.atheism', 'comp.graphics']
def load_traintestdata():
    newsgrps20_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'),
                                    shuffle=True, random_state=50)
    newsgrps20_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'),
                                          shuffle=True, random_state=50)
    train_df = pd.DataFrame([newsgrps20_train.data, newsgrps20_train.target.tolist()]).T
    test_df = pd.DataFrame([newsgrps20_test.data, newsgrps20_test.target.tolist()]).T

    train_df.columns = ['text', 'target']
    test_df.columns = ['text', 'target']

    #test_df.to_csv("testData.csv", index=False)

    # target names
    print(train_df['target'].nunique())

    #print(test_df.head())
    return train_df, test_df

#load_traintestdata()

