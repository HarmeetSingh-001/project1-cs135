import numpy as np
import pandas as pd
import os
import nltk
from nltk import pos_tag
import textwrap
import sklearn.linear_model
import sklearn.model_selection

# read in data
data_dir = 'data_readinglevel'
x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

# Function from day10 lab, use to create token of each line
def tokenize_text(raw_text):
    list_of_tokens = raw_text.split() # split method divides on whitespace by default
    for pp in range(len(list_of_tokens)):
        cur_token = list_of_tokens[pp]
        # Remove punctuation
        for punc in ['?', '!', '_', '.', ',', '"', '/']:
            cur_token = cur_token.replace(punc, "")
        # Turn to lower case
        clean_token = cur_token.lower()
        # Replace the cleaned token into the original list
        list_of_tokens[pp] = clean_token
    return list_of_tokens

# function from day1- lab, use to turn any text into a feature vector using the given dictionary
def transform_text_into_feature_vector(text, vocab_dict):
    V = len(vocab_dict.keys())
    count_V = np.zeros(V)
    for tok in tokenize_text(text):
        if tok in vocab_dict:
            vv = vocab_dict[tok]
            count_V[vv] += 1
    return count_V

# use to turn some list into an enuermated dict
def turn_list_into_dict(vocab_list):
    vocab_dict = dict()
    for vocab_id, tok in enumerate(vocab_list):
        vocab_dict[tok] = vocab_id
    return vocab_dict


# Function from day10 lab, use to count the tokens
tok_count_dict = dict()
for passage in x_train_df['text']:
    tok_list = tokenize_text(passage)
    for tok in tok_list:
        if tok in tok_count_dict:
            tok_count_dict[tok] += 1
        else:
            tok_count_dict[tok] = 1

# this will store all of our bags of words
all_bows = []

# this will store different bags of words where each index has only elements that have at least more than that value
# index 0 will be entirely full, index 1 will have elements that occur more than 1 time and so on up to 10
bow_only_greater_than_i = []
bow_full_sorted = list(sorted(tok_count_dict, key=tok_count_dict.get, reverse=True))
bow_only_greater_than_i.append(bow_full_sorted)
for i in range(1,11):
    current_bow = [w for w in bow_full_sorted if tok_count_dict[w] > i]
    bow_only_greater_than_i.append(current_bow)

# same as the above but this time we also remove parts of speech using NLTK, have to install it into your environment, and then also removing elements that occur a certain amount of times.
bow_only_greater_than_i_removed_POS = []
pos_tags = nltk.pos_tag(bow_full_sorted)
bow_full_sorted_removed_POS  = [w for w, pos_tag in pos_tags if pos_tag not in ['DT','PDT', 'WDT', 'IN','PRP', 'PRP$', 'WP', 'WP$' 'CC','TO']]
bow_only_greater_than_i_removed_POS.append(bow_full_sorted_removed_POS)
for i in range(1,11):
    current_bow = [w for w in bow_full_sorted_removed_POS if tok_count_dict[w] > i]
    bow_only_greater_than_i_removed_POS.append(current_bow)


all_bows.append(bow_only_greater_than_i)
all_bows.append(bow_only_greater_than_i_removed_POS)

# clean training labels to be simply 0 or 1
y_tr_N = []
for label in y_train_df['Coarse Label']:
    if label == "Key Stage 2-3":
        y_tr_N.append(0)
    else:
        y_tr_N.append(1)

# get number of items in our training set
N = len(x_train_df)

# go through all of our different bag of words
# train a lr model for each, compute CV on the result, store the bag number and the CV amount for the result with the best accuracy

# RESULT
# Default or POS: 0 Removed Elements: 0 Best Fold Amount: 3 Accuracy: 0.6265965101636986
# Default or POS: 0 Removed Elements: 1 Best Fold Amount: 3 Accuracy: 0.6280358121808972
# Default or POS: 0 Removed Elements: 2 Best Fold Amount: 3 Accuracy: 0.6244366829498756
# Default or POS: 0 Removed Elements: 3 Best Fold Amount: 3 Accuracy: 0.6220974529269174
# Default or POS: 0 Removed Elements: 4 Best Fold Amount: 3 Accuracy: 0.6208372623228459
# Default or POS: 0 Removed Elements: 5 Best Fold Amount: 3 Accuracy: 0.6154380828162996
# Default or POS: 0 Removed Elements: 6 Best Fold Amount: 3 Accuracy: 0.619758417167965
# Default or POS: 0 Removed Elements: 7 Best Fold Amount: 3 Accuracy: 0.6140006263071539
# Default or POS: 0 Removed Elements: 8 Best Fold Amount: 3 Accuracy: 0.6129194499842062
# Default or POS: 0 Removed Elements: 9 Best Fold Amount: 3 Accuracy: 0.6080615871291548
# Default or POS: 0 Removed Elements: 10 Best Fold Amount: 3 Accuracy: 0.6087811410057514
# Default or POS: 1 Removed Elements: 0 Best Fold Amount: 3 Accuracy: 0.6192170033844675
# Default or POS: 1 Removed Elements: 1 Best Fold Amount: 3 Accuracy: 0.6193973775136306
# Default or POS: 1 Removed Elements: 2 Best Fold Amount: 3 Accuracy: 0.6170576618306586
# Default or POS: 1 Removed Elements: 3 Best Fold Amount: 3 Accuracy: 0.6109390255795186
# Default or POS: 1 Removed Elements: 4 Best Fold Amount: 3 Accuracy: 0.6082403100142706
# Default or POS: 1 Removed Elements: 5 Best Fold Amount: 3 Accuracy: 0.6071605906713647
# Default or POS: 1 Removed Elements: 6 Best Fold Amount: 3 Accuracy: 0.6091401408880274
# Default or POS: 1 Removed Elements: 7 Best Fold Amount: 5 Accuracy: 0.6046474431615824
# Default or POS: 1 Removed Elements: 8 Best Fold Amount: 5 Accuracy: 0.6012264535806099
# Default or POS: 1 Removed Elements: 9 Best Fold Amount: 3 Accuracy: 0.5987044727733167
# Default or POS: 1 Removed Elements: 10 Best Fold Amount: 3 Accuracy: 0.6053619973758818

for i in range(0,2):
    current_bow_list = all_bows[i]
    for j in range (0,11):
        current_bow = current_bow_list[j]
        V = len(current_bow)

        # make our initial vector for x_tr
        x_tr_NV = np.zeros((N, V))
        # using vocab words, create the vector with true values based on word appearance
        for k, line in enumerate(x_train_df['text']):
            x_tr_NV[k] = transform_text_into_feature_vector(line, turn_list_into_dict(current_bow))
        
        # create lr model
        clf = sklearn.linear_model.LogisticRegression(C=1000.0, max_iter=250) 
        cv_scores = []
        # run cv using 3-5 folds, take best and print our index with our score
        for k in range(3,6):
            score = sklearn.model_selection.cross_val_score(clf, x_tr_NV, y_tr_N, cv=k, scoring='accuracy')
            cv_scores.append(score.mean())

        cv_scores = np.array(cv_scores)
        best_fold = np.argmax(cv_scores)

        print("Default or POS: " + str(i) + " Removed Elements: " + str(j) + " Best Fold Amount: " + str(best_fold + 3) + " Accuracy: " + str(cv_scores[best_fold]))







