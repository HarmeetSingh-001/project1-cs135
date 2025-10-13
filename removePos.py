import nltk
from nltk import word_tokenize, pos_tag

REMOVE_POS_TAGS = ['CC', 'CD', 'DT', 'IN','JJ','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','TO','UH','WDT','WP','WP$','WRB']

def remove_pos(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    filtered = [word for word, pos in pos_tags if pos not in REMOVE_POS_TAGS]
    return ' '.join(filtered)