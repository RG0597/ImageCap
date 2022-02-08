import util
from util import load_doc
from util import load_descriptions
from util import clean_descriptions
from util import to_vocab
from util import save_descriptions
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from util import load_set
from util import load_clean_desc
from keras.applications.inception_v3 import InceptionV3
from util import encode
from keras.models import Model
from util import wordtodict
from util import thresh
from pickle import dump, load
from util import max_length
from util import glovevec
import numpy as np
from util import modelfunc
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.layers.merge import add
from keras import Input, layers

import glob
from util import data_generator

def train(token,train):
  doc=load_doc(token)
  desc=load_descriptions(doc)
  clean_descriptions(desc)
  #vocab=to_vocab(desc)
  save_descriptions(desc,'desc.txt')
  train=load_set(train)

  images = r'C:/Users/HP/ImageCAp/data/Images/'
  img = glob.glob(images + '*.jpg')

  train_images_file = r'C:/Users/HP/ImageCAp/data/Flickr_8k.trainImages.txt'
  train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
  train_img = []
  for i in img:
    if i[len(images):] in train_images:
      train_img.append(i)
  test_images_file = r'C:\Users\ritvic.rai\PycharmProjects\imgcap\data\Flickr8k_text\Flickr_8k.testImages.txt'
  test_images = set(open(test_images_file, 'r').read().strip().split('\n'))
  test_img = []
  for i in img:
    if i[len(images):] in test_images:
      test_img.append(i)

  model = InceptionV3(weights='imagenet')
  model_new = Model(model.input, model.layers[-2].output)
  encoding_train = {}
  for img in train_img:
     encoding_train[img[len(images):]] = encode(img,model_new)
  train_features = load(open("data/Pickle/encoded_train_images.pkl", "rb"))
  train_descriptions = load_clean_desc('desc.txt', train)


  ixtoword, wordtoix=wordtodict(train_descriptions)
  vocab_size = len(ixtoword) + 1
  max_len = max_length(train_descriptions)

  embeddings_index=glovevec()

  embedding_dim = 200
  embedding_matrix = np.zeros((vocab_size, embedding_dim))
  for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
  model=modelfunc(vocab_size,embedding_dim,embedding_matrix,max_len)

  epochs = 150
  number_pics_per_bath = 3
  steps = len(train_descriptions) // number_pics_per_bath


  for i in range(epochs):

      generator = data_generator(train_descriptions, train_features, wordtoix, max_len, number_pics_per_bath,vocab_size)
      model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
      model.save('C:/Users/HP/ImageCAp/data/model_' + str(i) + '.h5')



# filename = r"C:/Users/HP/ImageCAp/data/Flickr8k_text\Flickr8k.token.txt"
# filename2= r'C:/Users/HP/ImageCAp/data/Flickr8k_text\Flickr_8k.trainImages.txt'
# train(filename,filename2)

def imageSearch(photo,max_len,model,ixtoword,wordtoix):

  in_text = 'startseq'
  for i in range(max_len):
    sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
    sequence = pad_sequences([sequence], maxlen=max_len)
    yhat = model.predict([photo, sequence], verbose=0)
    yhat = np.argmax(yhat)
    word = ixtoword[yhat]
    in_text += ' ' + word
    if word == 'endseq':
      break
  final = in_text.split()
  final = final[1:-1]
  final = ' '.join(final)
  return final

def predict(img):
    train_descriptions = load_clean_desc('desc.txt', train)

    ixtoword, wordtoix = wordtodict(train_descriptions)
    vocab_size = len(ixtoword) + 1
    max_len = max_length(train_descriptions)
    embedding_dim = 200
    embeddings_index = glovevec()
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in wordtoix.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    model = modelfunc(vocab_size, embedding_dim, embedding_matrix, max_len)
    model.load_weights(r'C:\Users\ritvic.rai\PycharmProjects\imgcap\model_149.h5')
    images = 'data/Flicker8k_Dataset/'
    encoding_train = {}
    model = InceptionV3(weights='imagenet')
    model_new = Model(model.input, model.layers[-2].output)
    encoding_train[img[len(images):]] = encode(img, model_new)
    with open(r"C:/Users/HP/ImageCAp/data/encoded_test_images.pkl", "rb") as encoded_pickle:
        encoding_test = load(encoded_pickle)
    pic = list(encoding_test.keys())[1]
    image = encoding_test[pic].reshape((1, 2048))
    x = plt.imread(images + pic)
    plt.imshow(x)
    plt.show()
    print("Image with Caption:", imageSearch(image,max_len,model,ixtoword,wordtoix))





