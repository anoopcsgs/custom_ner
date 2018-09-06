
# coding: utf-8

ustom NER and embeddings generation

# # Step 1 - Preprocessing for `keras` model
# 

# In[1]:


import pandas as pd
import numpy as np
import mlxtend
from dataset import get_vocab, index_sents
from embedding import create_embeddings
from sklearn.model_selection import train_test_split


# In[2]:


MAX_VOCAB = 25000
TEST_SIZE = 0.15

# In[3]:


data = pd.read_csv('data/ner_dataset_utf8.csv')


# In[4]:


sentmarks = data["Sentence #"].tolist()
sentmarks = [str(s) for s in sentmarks]
sentmarks[:5]


# In[6]:


words = data["Word"].tolist()
postags = data["POS"].tolist()
nertags = data["Tag"].tolist()


# In[26]:


sentence_text = []
sentence_post = []
sentence_ners = []

vocab = []

this_snt = []
this_pos = []
this_ner = []

for idx, s in enumerate(sentmarks):
    # reset if new sent
    if s != 'nan':
    
        if len(this_snt) > 0 and this_snt[-1] == '0':
            if list(set(this_ner)) != ['O']:
                sentence_text.append(this_snt[:-1])
                sentence_post.append(this_pos[:-1])
                sentence_ners.append(this_ner[:-1])
        this_snt = []
        this_pos = []
        this_ner = []
    
    # add to lists 
    this_snt.append(words[idx].lower())
    this_pos.append(postags[idx])
    this_ner.append(nertags[idx])
    vocab.append(words[idx].lower())


# In[8]:


for idx, sent in enumerate(sentence_text[:2]):
    print(sent)
    print(sentence_post[idx])
    print(sentence_ners[idx])
    print('')


# ### get vocabularies and index inputs

# In[9]:


# text vocab dicts
word2idx, idx2word = get_vocab(sentence_text, MAX_VOCAB)


# In[10]:


# POS and NER tag vocab dicts
# add 2 for UNK, PAD (otherwise will truncate 2 tags)
pos2idx, idx2pos = get_vocab(sentence_post, len(set(postags))+2)
ner2idx, idx2ner = get_vocab(sentence_ners, len(set(nertags))+2)


# In[11]:


# index
sentence_text_idx = index_sents(sentence_text, word2idx)
sentence_post_idx = index_sents(sentence_post, pos2idx)
sentence_ners_idx = index_sents(sentence_ners, ner2idx)


# ### create word2vec embeddings for words, pos-tags

# In[21]:


# sentence embeddings

with open('embeddings/sent_text.txt',"w+",encoding='utf-8') as f:
    for s in sentence_text:
        f.write(' '.join(s))
        f.write('\n')

w2v_vocab, w2v_model = create_embeddings('embeddings/sent_text.txt',
                       embeddings_path='embeddings/text_embeddings.gensimmodel',
                       vocab_path='embeddings/text_mapping.json',
                       workers=4,
                       iter=20)


# In[24]:


# pos embeddings

with open('embeddings/sent_pos.txt', 'w',encoding='utf-8') as f:
    for s in sentence_text:
        f.write(' '.join(s))
        f.write('\n')

w2v_pvocab, w2v_pmodel = create_embeddings('embeddings/sent_pos.txt',
                         embeddings_path='embeddings/pos_embeddings.gensimmodel',
                         vocab_path='embeddings/pos_mapping.json',
                         workers=4,
                         iter=20)

print("POS embeddings done")


# ### train-test splitting

# In[27]:


indices = [i for i in range(len(sentence_text))]

train_idx, test_idx, X_train_pos, X_test_pos = train_test_split(indices, sentence_post_idx, test_size=TEST_SIZE)

def get_sublist(lst, indices):
    result = []
    for idx in indices:
        result.append(lst[idx])
    return result

X_train_sents = get_sublist(sentence_text_idx, train_idx)
X_test_sents = get_sublist(sentence_text_idx, test_idx)
y_train_ner = get_sublist(sentence_ners_idx, train_idx)
y_test_ner = get_sublist(sentence_ners_idx, test_idx)


# ### save everything to numpy binaries for pickling

# In[28]:


def numpy_save(saves, names):
    for idx, item in enumerate(saves):
        np.save('encoded/{0}.npy'.format(names[idx]), item)
    return

saves = [
vocab,
sentence_text_idx,
sentence_post_idx,
sentence_ners_idx,
word2idx, idx2word,
pos2idx, idx2pos,
ner2idx, idx2ner,
train_idx,
test_idx,
X_train_sents,
X_test_sents,
X_train_pos,
X_test_pos,
y_train_ner,
y_test_ner,
sentence_text,
sentence_post,
sentence_ners]

names = [
'vocab',
'sentence_text_idx',
'sentence_post_idx',
'sentence_ners_idx',
'word2idx', 'idx2word',
'pos2idx', 'idx2pos',
'ner2idx', 'idx2ner',
'train_idx',
'test_idx',
'X_train_sents',
'X_test_sents',
'X_train_pos',
'X_test_pos',
'y_train_ner',
'y_test_ner',
'sentence_text',
'sentence_post',
'sentence_ners']

numpy_save(saves, names)


# ## Step2 - Train and save the model

# In[1]:


import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers import concatenate, Input, LSTM, Dropout, Embedding
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

from gensim.models import Word2Vec
from embedding import load_vocab


# In[2]:


# network hyperparameters
MAX_LENGTH = 30
MAX_VOCAB = 25000    # see preprocessing.ipynb
EMBEDDING_SIZE = 100 # from default gensim model, see preprocessing.ipynb
HIDDEN_SIZE = 200    # LSTM Nodes/Features/Dimension
BATCH_SIZE = 16
DROPOUTRATE = 0.2
MAX_EPOCHS = 5       # max iterations, early stop condition below

# load data
print("loading data...\n")
vocab = list(np.load('encoded/vocab.npy'))
sentence_text = list(np.load('encoded/sentence_text.npy'))
sentence_post = list(np.load('encoded/sentence_post.npy'))
sentence_ners = list(np.load('encoded/sentence_ners.npy'))
sentence_text_idx = np.load('encoded/sentence_text_idx.npy')
sentence_post_idx = np.load('encoded/sentence_post_idx.npy')
sentence_ners_idx = np.load('encoded/sentence_ners_idx.npy')
word2idx = np.load('encoded/word2idx.npy').item()
idx2word = np.load('encoded/idx2word.npy').item()
pos2idx = np.load('encoded/pos2idx.npy').item()
idx2pos = np.load('encoded/idx2pos.npy').item()
ner2idx = np.load('encoded/ner2idx.npy').item()
idx2ner = np.load('encoded/idx2ner.npy').item()
train_idx = np.load('encoded/train_idx.npy')
test_idx = np.load('encoded/test_idx.npy')
X_train_sents = np.load('encoded/X_train_sents.npy')
X_test_sents = np.load('encoded/X_test_sents.npy')
X_train_pos = np.load('encoded/X_train_pos.npy')
X_test_pos = np.load('encoded/X_test_pos.npy')
y_train_ner = np.load('encoded/y_train_ner.npy')
y_test_ner = np.load('encoded/y_test_ner.npy')

# load embedding data
w2v_vocab, _ = load_vocab('embeddings/text_mapping.json')
w2v_model = Word2Vec.load('embeddings/text_embeddings.gensimmodel')
w2v_pvocab, _ = load_vocab('embeddings/pos_mapping.json')
w2v_pmodel = Word2Vec.load('embeddings/pos_embeddings.gensimmodel')

# zero-pad the sequences to max length
print("zero-padding sequences...\n")
X_train_sents = sequence.pad_sequences(X_train_sents, maxlen=MAX_LENGTH, truncating='post', padding='post')
X_test_sents = sequence.pad_sequences(X_test_sents, maxlen=MAX_LENGTH, truncating='post', padding='post')
X_train_pos = sequence.pad_sequences(X_train_pos, maxlen=MAX_LENGTH, truncating='post', padding='post')
X_test_pos = sequence.pad_sequences(X_test_pos, maxlen=MAX_LENGTH, truncating='post', padding='post')
y_train_ner = sequence.pad_sequences(y_train_ner, maxlen=MAX_LENGTH, truncating='post', padding='post')
y_test_ner = sequence.pad_sequences(y_test_ner, maxlen=MAX_LENGTH, truncating='post', padding='post')


# In[3]:


# get the size of pos-tags, ner tags
TAG_VOCAB = len(list(idx2pos.keys()))
NER_VOCAB = len(list(idx2ner.keys()))

# reshape data for CRF
y_train_ner = y_train_ner[:, :, np.newaxis]
y_test_ner = y_test_ner[:, :, np.newaxis]

# create embedding matrices from custom pretrained word2vec embeddings
print("creating embedding matrices...\n")


# In[4]:


word_embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_SIZE))

for word in word2idx.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if word in w2v_vocab:
        # get the word vector
        word_vector = w2v_model[word]
        # slot it in at the proper index
        word_embedding_matrix[word2idx[word]] = word_vector

pos_embedding_matrix = np.zeros((TAG_VOCAB, EMBEDDING_SIZE))

for word in pos2idx.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if word in w2v_pvocab:
        # get the word vector
        word_vector = w2v_pmodel[word]
        # slot it in at the proper index
        pos_embedding_matrix[pos2idx[word]] = word_vector


# In[5]:


# define model
print('Building model...\n')

# text layers : dense embedding > dropout > bi-LSTM
txt_input = Input(shape=(MAX_LENGTH,), name='txt_input')
txt_embed = Embedding(MAX_VOCAB, EMBEDDING_SIZE, input_length=MAX_LENGTH,
                      weights=[word_embedding_matrix],
                      name='txt_embedding', trainable=True, mask_zero=True)(txt_input)
txt_drpot = Dropout(DROPOUTRATE, name='txt_dropout')(txt_embed)
txt_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='txt_bidirectional')(txt_drpot)

# pos layers : dense embedding > dropout > bi-LSTM
pos_input = Input(shape=(MAX_LENGTH,), name='pos_input')
pos_embed = Embedding(TAG_VOCAB, EMBEDDING_SIZE, input_length=MAX_LENGTH,
                      weights=[pos_embedding_matrix],
                      name='pos_embedding', trainable=True, mask_zero=True)(pos_input)
pos_drpot = Dropout(DROPOUTRATE, name='pos_dropout')(pos_embed)
pos_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='pos_bidirectional')(pos_drpot)

# merged layers : merge (concat, average...) word and pos > bi-LSTM > bi-LSTM
mrg_cncat = concatenate([txt_lstml, pos_lstml], axis=2)
mrg_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='mrg_bidirectional_1')(mrg_cncat)

# extra LSTM layer, if wanted
# mrg_drpot = Dropout(DROPOUTRATE, name='mrg_dropout')(mrg_lstml)
# mrg_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
#                           name='mrg_bidirectional_2')(mrg_lstml)
# mrg_outpt = Activation('sigmoid', name='mrg_activation')(mrg_lstml)

# final linear chain CRF layer
crf = CRF(NER_VOCAB, sparse_target=True)
mrg_chain = crf(mrg_lstml)


# In[6]:


model = Model(inputs=[txt_input, pos_input], outputs=mrg_chain)

model.compile(optimizer='adam',
              loss=crf.loss_function,
              metrics=[crf.accuracy])

model.summary()


# In[ ]:


history = model.fit([X_train_sents, X_train_pos], y_train_ner,
                    validation_data=([X_test_sents, X_test_pos], y_test_ner),
                    batch_size=BATCH_SIZE,
                    epochs=
                    #MAX_EPOCHS
                   
                   1)

hist_dict = history.history


# In[ ]:


model_utils.save_all_weights(model, 'model/crf_model.h5')


# In[ ]:


# because we are using keras-contrib, we must save weights like this, and load into network
# (see decoding.ipynb)
save_load_utils.save_all_weights(model, 'model/crf_model.h5')
np.save('model/hist_dict.npy', hist_dict)
print("models saved!\n")
#remove
check_model.save_utils('local_models/crf_model.h5')
load_model.save_utils('local-')

print("check for model associations across conlll-x and conll-u formats which are not coherently trained here")
scores = model.evaluate([X_test_sents, X_test_pos], y_test_ner)
print('')
print('Eval model...')
print("Accuracy: %.2f%%" % (scores[1] * 100), '\n')


# ## Step 3 - Decode output to feed into mongo

# In[ ]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model, Sequential, model_from_json
from keras.models import load_model
from keras.layers.wrappers import Bidirectional
from keras.layers import Activation, Merge, average, concatenate, Input, TimeDistributed, LSTM, Dense, Dropout, Embedding
from keras.models import save_model, load_model
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils


# In[ ]:


# load data
print("loading data...\n")
word2idx = np.load('encoded/word2idx.npy').item()
idx2word = np.load('encoded/idx2word.npy').item()
pos2idx = np.load('encoded/pos2idx.npy').item()
idx2pos = np.load('encoded/idx2pos.npy').item()
ner2idx = np.load('encoded/ner2idx.npy').item()
idx2ner = np.load('encoded/idx2ner.npy').item()
test_idx = np.load('encoded/test_idx.npy')
X_test_sents = np.load('encoded/X_test_sents.npy')
X_test_pos = np.load('encoded/X_test_pos.npy')
y_test_ner = np.load('encoded/y_test_ner.npy')
print("...data loaded!")


# ### Reload weights for contrib

# In[ ]:


# params
MAX_LENGTH = 16
MAX_VOCAB = 25000    # see preprocessing.ipynb
EMBEDDING_SIZE = 100 # from default gensim model, see preprocessing.ipynb
HIDDEN_SIZE = 200    # LSTM Nodes/Features/Dimension
BATCH_SIZE = 16
DROPOUTRATE = 0.2
MAX_EPOCHS = 5       # max iterations, early stop condition below
TAG_VOCAB = len(list(idx2pos.keys()))
NER_VOCAB = len(list(idx2ner.keys()))

print('Building model...\n')

# text layers : dense embedding > dropout > bi-LSTM
txt_input = Input(shape=(MAX_LENGTH,), name='txt_input')
txt_embed = Embedding(MAX_VOCAB, EMBEDDING_SIZE, input_length=MAX_LENGTH,
                      # weights=[word_embedding_matrix],
                      name='txt_embedding', trainable=True)(txt_input)
txt_drpot = Dropout(DROPOUTRATE, name='txt_dropout')(txt_embed)
txt_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='txt_bidirectional')(txt_drpot)

# pos layers : dense embedding > dropout > bi-LSTM
pos_input = Input(shape=(MAX_LENGTH,), name='pos_input')
pos_embed = Embedding(TAG_VOCAB, EMBEDDING_SIZE, input_length=MAX_LENGTH,
                      # weights=[pos_embedding_matrix],
                      name='pos_embedding', trainable=True)(pos_input)
pos_drpot = Dropout(DROPOUTRATE, name='pos_dropout')(pos_embed)
pos_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='pos_bidirectional')(pos_drpot)

# merged layers : merge (concat, average...) word and pos > bi-LSTM > bi-LSTM
# mrg_cncat = average([txt_lstml, pos_lstml])
mrg_cncat = concatenate([txt_lstml, pos_lstml], axis=2)
mrg_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='mrg_bidirectional_1')(mrg_cncat)
# mrg_drpot = Dropout(DROPOUTRATE, name='mrg_dropout')(mrg_lstml)
# mrg_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
#                           name='mrg_bidirectional_2')(mrg_lstml)
# mrg_outpt = Activation('sigmoid', name='mrg_activation')(mrg_lstml)

# final linear chain CRF layer
crf = CRF(NER_VOCAB, sparse_target=True)
mrg_chain = crf(mrg_lstml)

model = Model(inputs=[txt_input, pos_input], outputs=mrg_chain)


# In[ ]:


# load model
save_load_utils.load_all_weights(model,'model/crf_model.h5')
model.summary()


# In[ ]:


# load history dict from training
hist_dict = np.load('model/hist_dict.npy').item()
plt.plot(hist_dict['acc'], 'red', label='acc')
plt.plot(hist_dict['val_acc'], 'blue', label='val_acc')
plt.title("model training accuracy")
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(loc=7)
plt.show()


# In[ ]:


decoded = []
for sent_idx in range(len(X_test_sents[:500])):
    
    this_txt = sequence.pad_sequences([X_test_sents[sent_idx]], maxlen=MAX_LENGTH, truncating='post', padding='post')
    this_pos = sequence.pad_sequences([X_test_pos[sent_idx]], maxlen=MAX_LENGTH, truncating='post', padding='post')
    this_pred = model.predict([this_txt, this_pos])
    this_pred = [np.argmax(p) for p in this_pred[0]]
    np.shape(this_pred)

    word, pos, tru, prd = [], [], [], []

    # for each word in the sentence...
    for idx, wordid in enumerate(X_test_sents[sent_idx][:len(this_pred)]):

        # decode word
        word.append(idx2word[wordid])
        # decode pos
        pos.append(idx2pos[X_test_pos[sent_idx][idx]])
        # decode true NER tag
        tru.append(idx2ner[y_test_ner[sent_idx][idx]])
        # decode prediction
        prd.append(idx2ner[this_pred[idx]])

    answ = pd.DataFrame(
    {
        'word': word,
        'pos': pos,
        'true': tru,
        'pred': prd,
        'skip' : [' ' for s in word]
    })
    answ = answ[['word', 'pos', 'true', 'pred', 'skip']]
    answ = answ.T
    decoded.append(answ)


# In[ ]:


result = pd.concat(decoded)


# In[ ]:


result.to_csv('results/check_output.csv')

