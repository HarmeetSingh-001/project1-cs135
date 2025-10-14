import nltk
from nltk import word_tokenize, pos_tag

REMOVE_POS_TAGS = [
    # Conjunctions, Determiners, and Pronouns
    'CC', 'CD', 'DT', 'EX', 'IN', 'PDT', 'POS',
    'PRP', 'PRP$', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB',

    # Adjectives
    'JJ', 'JJR', 'JJS',

    # Nouns
    'NN', 'NNS', 
    # 'NNP', 
    'NNPS',

    # Adverbs
    # 'RB', 'RBR', 'RBS',

    # Verbs
    # 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',

    # Others / Special
    'FW', 'LS', 'MD', 'RP', 'SYM'
]

# all kept Best CV Accuracy: 0.682755494939792, vocab = 17

# removed adj Best CV Accuracy: 0.6989617746456277 vocab = 853 ROC = 0.645

# removed nouns Best CV Accuracy: 0.741416081234521 vocab = 4040 ROC = 0.667
#   removed NN Best CV Accuracy: 0.70708435828011 vocab = 2323
#   removed NNS Best CV Accuracy: 0.6984249802730261 vocab = 688
#   removed NNP Best CV Accuracy: 0.7394182615224492 vocab = 1066
#   removed NNPS Best CV Accuracy: 0.6831691824222339 vocab = 23

# removed adverbs Best CV Accuracy: 0.7067218803775093 vocab = 406 ROC = 
# removed verbs Best CV Accuracy: 0.7182363581807619 vocab = 1870 ROC =
# removed others Best CV Accuracy: 0.6999604873158326 vocab = 46 ROC = 

def remove_pos(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    filtered = [word for word, pos in pos_tags if pos not in REMOVE_POS_TAGS]
    return ' '.join(filtered)