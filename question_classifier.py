import string
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import gc
import configparser
import argparse
import torch, random
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
torch.manual_seed(1)
random.seed(1)


def getLines(path):
  """
  This function reads sentences from the file stored in the path. It returns a list of sentences 
  called lines.
  """
  print("Loading sentences from",path,"...")
  f = open(path)
  lines = f.read()
  lines = lines.split('\n')
  return lines

def getStopWords(path):
  """
  This function reads the stop words stored at the path. It also removes question words from 
  stop_words like 'which', 'where', etc., and returns a list of stopwords
  """
  f = open(path, "r")
  stopwords = f.read().split("\n")
  question_words = ['what',
  'which',
  'who',
  'whom','when',
  'where',
  'why',
  'how']
  stopwords = [word for word in stopwords if word not in question_words]
  return stopwords

def performPreProcessing(lines, stopwords):
  """
  This function performs pre-processing on the given lines. It changes the characters to 
  lowercase, removes punctuation from the sentences, removes stopwords, and splits all the 
  sentences into a list of words. It also separates the coarse class and fine class from the 
  sentence. It returns filtered_sentences, coarse_classes, fine_classe
  """
  print("Pre-Processing..")
  sentences = []
  fine_classes = []
  coarse_classes = []
  for line in lines:
    words = line.split()
    first_word=words[0]
    sentence = ' '.join(words[1:])
    sentences.append(sentence)
    coarse_class = first_word.split(":")[0]
    fine_class = first_word.split(":")[1]
    fine_classes.append(fine_class)
    coarse_classes.append(coarse_class)
  sentences = map(str.lower,sentences)
  without_punctuation_sentences = [line.translate(str.maketrans('', '', string.punctuation)) for line in sentences]
  without_punctuation_sentences = [line.split() for line in without_punctuation_sentences]
  filtered_sentences = []
  for words in without_punctuation_sentences:
    filtered_words = []
    for word in words:
      if(word not in stopwords):
        filtered_words.append(word)
    filtered_sentences.append(filtered_words)
  return filtered_sentences, coarse_classes, fine_classes

def getCoarseAndFineClassKeys(coarse_classes,fine_classes):
  """
  This function returns unique coarse_classes and fine_classes as coarse_classes_keys and 
  fine_classes_keys.
  """
  coarse_classes_keys = set(coarse_classes)
  fine_classes_keys = set(fine_classes)
  coarse_classes_keys = list(coarse_classes_keys)
  coarse_classes_keys.sort()
  fine_classes_keys = list(fine_classes_keys)
  fine_classes_keys.sort()
  return coarse_classes_keys, fine_classes_keys


def load_glove_model(file):
    """
    This function reads a glove model from 'file' and returns a dictionary object glove_model
    """
    glove_model = {}
    with open(file,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    return glove_model

def createEmbeddingMatrix(model):
  """
  This function takes a glove model and creates an embedding matrix. It returns the 
  embedding_matrix_tensor and the words in the glove model vocabulary
  """
  vocab = model.keys()
  words = []
  for i in vocab:
    words.append(i)
  embedding_matrix = np.zeros((len(vocab)+1, 300))
  embedding_matrix[0] = torch.tensor(np.zeros(300), dtype = torch.float32) # (For 0 embedding)
  i = 1
  for word in words:
    embedding_matrix[i] = torch.tensor(model[word], dtype = torch.float32)
    i = i+1
  embedding_matrix_tensor = torch.tensor(embedding_matrix, dtype = torch.float32)
  return words,embedding_matrix_tensor

class dataset(Dataset):
  """
  This class defines the dataset used for training and development
  """
  def __init__(self, x, y):
        # single input
        self.x = torch.tensor(x, dtype = torch.int64)
        self.y = torch.tensor(y)
        self.len = len(self.x)
  def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
  def __len__(self):
        return self.len

def getData(data_path,stop_words_path):
  """
  This function returns pre-processed sentences, separated coarse_classes and fine_classes 
  from questions stored at data_path
  """
  filtered_sentences,coarse_classes, fine_classes = performPreProcessing(getLines(data_path), getStopWords(stop_words_path))
  return filtered_sentences,coarse_classes, fine_classes

def getFeatures(filtered_sentences,coarse_classes, fine_classes,coarse_classes_keys, fine_classes_keys,words, coarse = True, max_words = 25, batch_size = 32):
  
  X = []
  Y_coarse = []
  Y_fine = []
  for sentence in filtered_sentences:
    ids = []
    for i in range(0,len(sentence)):
      if (sentence[i] in words):
        ids.append(torch.tensor(words.index(sentence[i])+1, dtype = torch.int64))
      else:
        ids.append(torch.tensor(words.index("#UNK#")+1))
    while len(ids) != max_words:
      ids.append(torch.tensor(0, dtype = torch.int64))
    X.append(ids)
  for i in coarse_classes:
    Y_coarse.append(coarse_classes_keys.index(i))
  for i in fine_classes:
    Y_fine.append(fine_classes_keys.index(i))
  if coarse:
    X, y = X, Y_coarse
  else:
    X,y= X, Y_fine
  data = dataset(X,y)
  data_loader = DataLoader(data, batch_size=batch_size)
  return data_loader

def getTrainingFeatures(train_path, stop_words_path, words,coarse, max_words, batch_size):
  """
  This function reads data from train_path and returns a train_loader, coarse_classes_keys, 
fine_classes_keys
  """
  print("----------- Training Data  -----------")
  filtered_sentences,coarse_classes, fine_classes = performPreProcessing(getLines(train_path), getStopWords(stop_words_path))
  coarse_classes_keys, fine_classes_keys = getCoarseAndFineClassKeys(coarse_classes,fine_classes)
  train_loader = getFeatures(filtered_sentences,coarse_classes, fine_classes,coarse_classes_keys, fine_classes_keys,words, coarse = coarse, max_words = max_words, batch_size = batch_size)
  return train_loader, coarse_classes_keys, fine_classes_keys

def getDevFeatures(dev_path, stop_words_path, words,coarse_classes_keys, fine_classes_keys, coarse, max_words, batch_size):
  """
  This function reads data from dev_path and returns a dev_loader
  """
  print("----------- Dev Data  -----------")
  filtered_sentences,coarse_classes, fine_classes = performPreProcessing(getLines(dev_path), getStopWords(stop_words_path))
  dev_loader = getFeatures(filtered_sentences,coarse_classes, fine_classes,coarse_classes_keys, fine_classes_keys,words, coarse = coarse, max_words = max_words, batch_size = batch_size)
  return dev_loader

def getTestingFeatures(test_path,stop_words_path,words, coarse_classes_keys, fine_classes_keys,coarse, max_words):
  """
  This function reads data from test_path and returns X_test, y_tes
  """
  print("----------- Test Data  -----------")
  filtered_sentences,coarse_classes, fine_classes = performPreProcessing(getLines(test_path), getStopWords(stop_words_path))
  X_test = []
  y_test_coarse = []
  y_test_fine = []
  for sentence in filtered_sentences:
    ids = []
    for i in range(0,len(sentence)):
      if (sentence[i] in words):
        ids.append(torch.tensor(words.index(sentence[i])+1, dtype = torch.int64))
      else:
        ids.append(torch.tensor(words.index("#UNK#")+1))
    while len(ids) != max_words:
      ids.append(torch.tensor(0, dtype = torch.int64))
    X_test.append(ids)
  for i in coarse_classes:
    y_test_coarse.append(coarse_classes_keys.index(i))
  for i in fine_classes:
    y_test_fine.append(fine_classes_keys.index(i))
  if coarse:
    return X_test, y_test_coarse
  else:
    return X_test, y_test_fine

class BOW_Model(nn.Module):
    """
    This class defines a BOW Model with three layers. It defines and sets the embedding layer 
based on the input flags. It also defines a forward function that performs forward propagation 
on the model.

    """
    def __init__(self, embedding_matrix_tensor, classes_keys, emd_dim, preTrained = True, fine_tune = False):
        super(BOW_Model, self).__init__()
        self.emd_dim = emd_dim
        if preTrained:
          self.embeddings = nn.Embedding.from_pretrained(embedding_matrix_tensor)
          if fine_tune:
            self.embeddings.weight.requires_grad = True
            print("Finetuning the Embeddings..")
          else:
            self.embeddings.weight.requires_grad = False
            print("Freezing the Embeddings..")
        else:
          print("Randomizing the Embeddings..")
          self.embeddings = nn.Embedding(embedding_matrix_tensor.shape[0], embedding_matrix_tensor.shape[1])
          random_vecs = torch.FloatTensor(np.random.rand(embedding_matrix_tensor.shape[0], embedding_matrix_tensor.shape[1]))
          self.embeddings.weight = nn.Parameter(random_vecs)

        self.seq = nn.Sequential(
            nn.Linear(emd_dim, 750),
            nn.LeakyReLU(),
            nn.Linear(750,420),
            nn.LeakyReLU(),
            nn.Linear(420, len(classes_keys)),
        )
    def forward(self, X_batch):
      features = []
      for sentence in X_batch:
        vec = torch.tensor(np.zeros(self.emd_dim), dtype = torch.float32)
        for word in sentence:
          vec += self.embeddings(word)
        vec = vec/len(sentence)
        features.append(vec)
      features1 = torch.stack(features)
      return self.seq(features1)


class BiLSTM_Model(nn.Module):
    """
    This class defines a BiLSTM Model with 1 BiLSTM layer and 1 dense layer. It defines and 
sets the embedding layer based on input flags. It also defines a forward function that 
performs forward propagation on the model.
    """
    def __init__(self, embedding_matrix_tensor, classes_keys, emd_dim, preTrained = True, fine_tune = False):
        super(BiLSTM_Model, self).__init__()
        self.emd_dim = emd_dim
        if preTrained:
          self.embeddings = nn.Embedding.from_pretrained(embedding_matrix_tensor)
          if fine_tune:
            self.embeddings.weight.requires_grad = True
            print("Finetuning the Embeddings..")
          else:
            self.embeddings.weight.requires_grad = False
            print("Freezing the Embeddings..")
        else:
          print("Randomizing the embeddings..")
          self.embeddings = nn.Embedding(embedding_matrix_tensor.shape[0], embedding_matrix_tensor.shape[1])
          random_vecs = torch.FloatTensor(np.random.rand(embedding_matrix_tensor.shape[0], embedding_matrix_tensor.shape[1]))
          self.embeddings.weight = nn.Parameter(random_vecs)
        self.bilstm = nn.LSTM(input_size = emd_dim, hidden_size = 96, bidirectional = True, batch_first = True)
        self.seq = nn.Sequential(
            nn.Linear(192, len(classes_keys)),
        )
    def forward(self, X_batch):
      features = []
      features = self.embeddings(X_batch)
      biLSTM_output, (h_n,c_n) = self.bilstm(features)
      biLSTM_output_split = biLSTM_output.view(len(X_batch), 25, 2, 96)
      out_forward = biLSTM_output_split[:, :, 0, :]
      out_backward = biLSTM_output_split[:, :, 1, :]
      final_repr = torch.cat([out_forward[:, -1, :], out_backward[:, 0, :]], dim=1)
      return self.seq(final_repr)

class EnsembleModel(nn.Module):
    def __init__(self, bilstm1, bilstm2, bow1, classes_keys):
        super().__init__()
        self.bilstm1 = bilstm1
        self.bilstm2 = bilstm2
        self.bow1 = bow1
        self.seq = nn.Linear(len(classes_keys) * 3, len(classes_keys))

    def forward(self, x):
        o1 = self.bilstm1(x)
        o2 = self.bilstm2(x)
        o3 = self.bow1(x)
        x = torch.cat((o1, o2, o3), dim=1)
        out = self.seq(x)
        return out

def getModel(classes_keys, embedding_matrix_tensor,emd_dim, preTrained = True, fine_tune = False, biLSTM = False):
  """
  This function creates a BOW or BiLSTM Model having an output layer of size of 
classes_keys based on the input biLSTM flag. It returns the model object.
  """
  if biLSTM is False:
    print("Building BOW Model..")
    model = BOW_Model(embedding_matrix_tensor, classes_keys, emd_dim, preTrained = preTrained, fine_tune = fine_tune)
  else:
    print("Building BiLSTM Model..")
    model = BiLSTM_Model(embedding_matrix_tensor, classes_keys, emd_dim, preTrained = preTrained, fine_tune = fine_tune)
  return model

def calculateF1Score(model, loss_fn, val_loader):
    """
    This function takes a trained model and calculates the loss_fn and F1 Score on val_loader. It 
displays the Dev Loss, Dev Accuracy, and Dev F1-Score.
    """
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [],[],[]
        for X, Y in val_loader:
            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())

            Y_shuffled.append(Y)
            Y_preds.append(preds.argmax(dim=-1))

        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)
        print("Valid Loss {:.3f}".format((torch.tensor(losses).mean())))
        print("Valid Acc  :",accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy()))
        print("Valid F1  :",f1_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy(), average = 'macro'))


def performTraining(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
   """
   This function takes an initialized model and trains for the given number of epochs. It also 
uses the given loss_fn and optimizer to update the weights of the model.
   """ 
   for i in range(1, epochs+1):
        losses = []
        for X, Y in tqdm(train_loader):
            Y_preds = model(X)

            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch",i)
        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        calculateF1Score(model, loss_fn, val_loader)

def trainModel(model, train_loader, test_loader, ensemble = False, epochs = 10, learning_rate = 2e-3):
  """
  This function trains a model over data from train_loader and val_loader using the given 
optimizer and loss_fn for the given number of epochs.
  """
  loss_fn = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=learning_rate)
  if ensemble:
    print("----------- Ensemble Training  -----------")
    optimizer = Adam(model.seq.parameters(), lr=learning_rate)
  performTraining(model, loss_fn, optimizer, train_loader, test_loader, epochs)

def evaluateModel(model, X_test, y_test, coarse, coarse_classes_keys, fine_classes_keys, eval_path):
  """
  This function evaluates a trained model on X_test and y_test and stores the classification 
output at eval_path. It also displays the classification report for the test split.
  """
  y_pred_test = model(torch.tensor(X_test,dtype=torch.int64))
  y_pred_test_argmax = [y1.tolist().index(max(y1)) for y1 in y_pred_test]
  accuracy = accuracy_score(y_test,y_pred_test_argmax)
  file = open(eval_path,'w')
  for y in y_pred_test_argmax:
    if coarse:
      file.write(coarse_classes_keys[y]+"\n")
    else:
      file.write(fine_classes_keys[y]+"\n")
  file.write("Accuracy: "+str(100*accuracy))
  file.close()
  print(classification_report(y_test
                            , y_pred_test_argmax))

def trainTestPipeline(train,test,model_path, eval_path,train_path,dev_path,test_path, glove_path,stop_words_path, coarse = True, preTrained = True, fine_tune = False, biLSTM = False, max_words = 25, batch_size = 32, ensemble = False, epochs = 10, lr = 2e-3, emd_dim = 300, ensemble_models= None):
  """
  This function acts as a pipeline for training and testing a given model type over the given 
dataset. It loads data from the train_path, dev_path, and test_path. It then trains a model on 
the corresponding data and stores it at model_path. It also stores the test classification 
output at eval_path.

  """
  if train:
    print("Loading Data..")
    glove_model = load_glove_model(glove_path)
    words,embedding_matrix_tensor = createEmbeddingMatrix(glove_model)
    train_loader, coarse_classes_keys, fine_classes_keys = getTrainingFeatures(train_path, stop_words_path, words,coarse, max_words, batch_size=batch_size)
    dev_loader = getDevFeatures(dev_path, stop_words_path, words,coarse_classes_keys, fine_classes_keys, coarse, max_words, batch_size=batch_size)
    print("----------- Model Training  -----------")



    if ensemble:
      if coarse:
        bilstm1 = getModel(coarse_classes_keys, embedding_matrix_tensor, emd_dim, preTrained = True, fine_tune = False, biLSTM = True)
        trainModel(bilstm1, train_loader, dev_loader)
        torch.save(bilstm1,ensemble_models[0])
        bilstm2 = getModel(coarse_classes_keys, embedding_matrix_tensor, emd_dim, preTrained = True, fine_tune = True, biLSTM = True)
        trainModel(bilstm2, train_loader, dev_loader)
        torch.save(bilstm2,ensemble_models[1])
        bow1 = getModel(coarse_classes_keys, embedding_matrix_tensor, emd_dim, preTrained = True, fine_tune = False, biLSTM = False)
        trainModel(bow1, train_loader, dev_loader)
        torch.save(bow1,ensemble_models[2])
        model = EnsembleModel(bilstm1,bilstm2,bow1,coarse_classes_keys)
      else:

        bilstm1 = getModel(fine_classes_keys, embedding_matrix_tensor, emd_dim, preTrained = True, fine_tune = False, biLSTM = True)
        trainModel(bilstm1, train_loader, dev_loader)
        bilstm2 = getModel(fine_classes_keys, embedding_matrix_tensor, emd_dim, preTrained = True, fine_tune = True, biLSTM = True)
        trainModel(bilstm2, train_loader, dev_loader)
        bow1 = getModel(fine_classes_keys, embedding_matrix_tensor, emd_dim, preTrained = True, fine_tune = False, biLSTM = False)
        trainModel(bow1, train_loader, dev_loader)
        model = EnsembleModel(bilstm1,bilstm2,bow1,fine_classes_keys)
    else:
      if coarse:
        model = getModel(coarse_classes_keys, embedding_matrix_tensor, emd_dim, preTrained = preTrained,  fine_tune = fine_tune, biLSTM = biLSTM)
      else:
        model = getModel(fine_classes_keys, embedding_matrix_tensor, emd_dim, preTrained = preTrained, fine_tune = fine_tune, biLSTM = biLSTM)
    print("Model Building:")
    print(model.children)
    print("Training Model")
    trainModel(model, train_loader, dev_loader,ensemble = ensemble, epochs = epochs, learning_rate = lr)
    print('Saving Model')
    torch.save(model,model_path)

  if test:
    print('Loading Model')
    model = torch.load(model_path)
    X_test, y_test = getTestingFeatures(test_path,stop_words_path,words, coarse_classes_keys, fine_classes_keys,coarse, max_words)
    evaluateModel(model, X_test, y_test, coarse, coarse_classes_keys, fine_classes_keys, eval_path)

def runExperiment(config_file, train = True, test = True):
  """
  This function takes a config_file and based on the train and test flags run the 
trainTestPipeline based on the details in config_file
  """
  config = configparser.ConfigParser()
  config.sections()
  config.read(config_file)
  train_path = config['PATH']['path_train']
  dev_path = config['PATH']['path_dev']
  test_path = config['PATH']['path_test']
  stop_words_path = config['PATH']['path_stop_words']
  classes = config['PATH']['classes']
  model = config['MODEL']['model']
  model_path = config['MODEL']['path_model']
  epochs = int(config['MODEL']['epoch'])
  glove_path = config['MODEL']['path_glove']
  pre_trained = config['SETTINGS']['pre_trained']
  train_embedding = config['SETTINGS']['train_embedding']
  emd_dim = int(config['SETTINGS']['word_embedding_dim'])
  batch_size = int(config['SETTINGS']['batch_size'])
  lr = float(config['SETTINGS']['lr_param'])
  eval_path = config['EVAL']['path_eval_result']
  ensemble = False
  ensemble_models = []
  if 'ENSEMBLE' in config:
      print("Ensemble")
      ensemble = True
      ensemble_models.append(config['ENSEMBLE']['bilstm1'])
      ensemble_models.append(config['ENSEMBLE']['bilstm2'])
      ensemble_models.append(config['ENSEMBLE']['bow1'])

  coarse = False
  if classes == 'coarse':
    coarse = True

  preTrained = False
  if pre_trained == 'True':
    preTrained = True

  fine_tune = False
  if train_embedding == 'True':
    fine_tune = True

  biLSTM = True
  if model == 'bow':
    biLSTM = False

  trainTestPipeline(train,test,model_path, eval_path,train_path,dev_path,test_path, glove_path,stop_words_path, coarse = coarse, preTrained = preTrained, fine_tune = fine_tune, biLSTM = biLSTM, max_words = 25, batch_size = batch_size, emd_dim = emd_dim, epochs = epochs, lr = lr,ensemble = ensemble, ensemble_models = ensemble_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
    parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')
    args = parser.parse_args()
    config = args.config
    train = args.train
    test = args.test
    runExperiment(config,train,test)
