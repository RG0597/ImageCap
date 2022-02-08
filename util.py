#importing libraries
import numpy as np
import tensorflow
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization

from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop
def load_doc(file):
  f=open(file,'r')
  text=f.read()
  f.close()
  return text
def load_descriptions(doc):
  mapping=dict()
  for line in doc.split('\n'):
    tokens=line.split()
    if len(line)<2:
      continue
    image_id,image_desc=tokens[0],tokens[1:]
    image_id=image_id.split('.')[0]
    image_desc=' '.join(image_desc)
    if image_id not in mapping:
      mapping[image_id]=list()
    mapping[image_id].append(image_desc)
  return mapping

def clean_descriptions(descriptions):
  table=str.maketrans('','',string.punctuation)
  for key,desc_lst in descriptions.items():
    for i in range(len(desc_lst)):
      desc=desc_lst[i]
      desc=desc.split()
      desc=[w.lower() for w in desc]
      desc=[w.translate(table) for w in desc]
      desc=[word for word in desc if len(word)>1]
      desc=[word for word in desc if word.isalpha()]
      desc_lst[i]=' '.join(desc)

def to_vocab(descriptions):
  all_desc=set()
  for key in descriptions.keys():
    [all_desc.update(d.split()) for d in descriptions[key]]
  return all_desc


def save_descriptions(descriptions,filename):
  lines=list()
  for key,desc_lst in descriptions.items():
    for desc in desc_lst:
      lines.append(key+ ' ' +desc)
    data='\n'.join(lines)
    f=open(filename,'w')
    f.write(data)
    f.close()

def load_set(filename):
  doc=load_doc(filename)
  dataset=list()
  for line in doc.split('\n'):
    if len(line)<1:
      continue
    identifier=line.split('.')[0]
    dataset.append(identifier)
  return set(dataset)


def load_clean_desc(filename,dataset):
  doc=load_doc(filename)
  descriptions=dict()
  for line in doc.split('\n'):
    tokens=line.split()
    image_id, image_desc=tokens[0],tokens[1:]
    if image_id in dataset:
      if image_id not in descriptions:
        descriptions[image_id]=list()
      desc='startseq '+' '.join(image_desc)+' endseq'
      descriptions[image_id].append(desc)
  return descriptions


def preprocess(image_path):
  img=image.load_img(image_path,target_size=(299,299))
  x=image.img_to_array(img)
  x=np.expand_dims(x,axis=0)
  x=preprocess_input(x)
  return x


model=InceptionV3(weights='imagenet')
model_new=Model(model.input,model.layers[-2].output)


def encode(image,model_new):
  image=preprocess(image)
  fea_vec=model_new.predict(image)
  fea_vec=np.reshape(fea_vec,fea_vec.shape[1])
  return fea_vec





def thresh(all_train_captions):
  word_count_threshold=10
  word_counts={}
  nsents=0
  for sent in all_train_captions:
    nsents+=1
    for w in sent.split(' '):
      word_counts[w]=word_counts.get(w,0)+1
  vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
  return vocab

def wordtodict(train_descriptions):
  all_train_captions = []
  for key, val in train_descriptions.items():
    for cap in val:
      all_train_captions.append(cap)
  vocab=thresh(all_train_captions)
  ixtoword={}
  wordtoix={}
  ix=1
  for w in vocab:
    wordtoix[w]=ix
    ixtoword[ix]=w
    ix+=1
  return ixtoword,wordtoix


def to_lines(descriptions):
  all_desc=list()
  for key in descriptions.keys():
    [all_desc.append(d) for d in descriptions[key]]
  return all_desc


def max_length(descripitons):
  lines=to_lines(descripitons)
  return  max(len(d.split())for d in lines)



def data_generator(descriptions,photos,wordtoix,max_length,num_photos_per_batch,vocab_size):
  X1,X2,y=list(),list(),list()
  n=0
  while 1:
    for key,desc_list in descriptions.items():
      n+=1
      photo=photos[key+'.jpg']

      for desc in desc_list:
        seq=[wordtoix[word] for word in desc.split(' ') if word in wordtoix]
        for i in range(1,len(seq)):
          in_seq,out_seq=seq[:1],seq[i]
          in_seq=pad_sequences([in_seq],maxlen=max_length)[0]
          out_seq=to_categorical([out_seq],num_classes=vocab_size)[0]
          X1.append(photo)
          X2.append(in_seq)
          y.append(out_seq)

      if n==num_photos_per_batch:
        yield[[array(X1),array(X2)],array(y)]
        X1,X2,y=list(),list(),list()
        n=0

def glovevec():
  glove_dir='glove'
  embeddings_index={}
  f =open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")
  for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=coefs
  f.close()
  return embeddings_index
def modelfunc(vocab_size,embedding_dim,embedding_matrix,max_length):
  inputs1=Input(shape=(2048,))
  fe1=Dropout(0.5)(inputs1)
  fe2=Dense(256,activation='relu')(fe1)
  inputs2 = Input(shape=(max_length,))
  se1=Embedding(vocab_size,embedding_dim,mask_zero=True)(inputs2)
  se2 = Dropout(0.5)(se1)
  se3 = LSTM(256)(se2)
  decoder1 = add([fe2, se3])
  decoder2 = Dense(256, activation='relu')(decoder1)
  outputs = Dense(vocab_size, activation='softmax')(decoder2)
  model = Model(inputs=[inputs1, inputs2], outputs=outputs)
  model.layers[2].set_weights([embedding_matrix])
  model.layers[2].trainable = False
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  return model
