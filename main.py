### Prohibio Health - Climate Mosquito Model
### Madhav Malhotra
### Under Guidance of Yaacov Iland, Jim Fare, and Ashu Syal
### This project takes open source research data on 
### climate and mosquito populations in Manatee County, 
### Florida. It then creates a predictive model for
### mosquito populations based on the climate data.

import tensorflow as tf 
import numpy
import csv

#Change test, train labels from 0 to 3
with open('Final-test.csv', 'r') as testInput:
  #Note: LPO = Labels Processed Output
  with open('testLPO.csv', 'w') as testOutput:
    TIReader = csv.reader(testInput); #TI = testInput
    TOWriter = csv.writer(testOutput); #TO = testOutput
    header = next(TIReader, None); #Skipping header row
    #Only keep tmax, tmin, prcp, labels
    newHeader = [header[5], header[4], header[6], header[7]];
    labelsPO = []; #PO = Processed Output
    labelsPO.append(newHeader);

    #Goes through all rows, changing text labels to 
    #integer values
    for row in TIReader:
      if row[5] == '0-100':
        row[5] = 0;
      elif row[5] == '101-500':
        row[5] = 1; 
      elif row[5] == '501-2000':
        row[5] = 2;
      elif row[5] == '2001-3500':
        row[5] = 3;

      row = [row[5], row[4], row[6], row[7]];
      labelsPO.append(row);

    TOWriter.writerows(labelsPO); #Output processed file
    testOutput.close();
  testInput.close();

with open('Final-train.csv', 'r') as trainInput:
  with open('trainLPO.csv', 'w') as trainOutput:
    TrIReader = csv.reader(trainInput); #TrI = trainInput
    TrOWriter = csv.writer(trainOutput); #TrO = trainOutput
    header = next(TrIReader, None); #Getting csv header
    #Only keep tmax, tmin, prcp, labels
    newHeader = [header[5], header[4], header[6], header[7]];
    labelsPO = [];
    labelsPO.append(newHeader); #Add header to new output

    for row in TrIReader:
      #Note: Label map: 0-100 --> 0
      if row[5] == '0-100':
        row[5] = 0;
      #Note: Label map: 101-500 --> 1
      elif row[5] == '101-500':
        row[5] = 1;
      #Note: Label map: 501-2000 --> 2 
      elif row[5] == '501-2000':
        row[5] = 2;
      #Note: Label map: 2001-3500 --> 3
      elif row[5] == '2001-3500':
        row[5] = 3;  

      row = [row[5], row[4], row[6], row[7]];
      labelsPO.append(row);
    
    #Outputting processed file
    TrOWriter.writerows(labelsPO);
    trainOutput.close();
  trainInput.close();

#Convert test, train csv to Tensorflow Dataset

#Getting the csv column headers
with open('testLPO.csv', 'r') as justGettingHeader:
  header = justGettingHeader.readline();
  CSV_COLUMNS = header.rstrip('\n').split(',');
  justGettingHeader.close();

#Label map described earlier (string to int)
LABELS = [0, 1, 2, 3];
LABEL_COLUMN = 'Monthly Mosquito Count'
FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN];

def getDataset(file_path):
  '''
  Turns CSV to TF dataset
  Input: (string) csv filepath 
  Output: (tuple) list of features, list of labels
  '''
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size = 100,
      label_name = LABEL_COLUMN,
      na_value = "?",
      num_epochs = 1,
      ignore_errors = True);
  return dataset;
print("a");

rawTrainData = getDataset('trainLPO.csv');
rawTestData = getDataset('testLPO.csv');
#Normalise tmax, tmin, prcp data

#Calculating averages for normalisation
with open('Final-processed.csv', 'r') as findingMeans:
  meansReader = csv.reader(findingMeans);
  next(meansReader, None); #Skip csv headers

  #Initialising variables for sums of data
  sumPRCP = 0; #For precipitation data (5th col)
  sumTMAX = 0; #For max. temp. data (7th col)
  sumTMIN = 0; #For min. temp. data (8th col)
  count = 0; #To track number of values for average
  
  #Note: meansReader is iterable object with all
  #rows of csv file. Each row is a list with each
  #cell value at an index in that list
  for row in meansReader: #Goes through all data
    if len(row[4]) != 0: #Checks record not empty
      sumPRCP += float(row[4]);
    if len(row[6]) != 0 and len(row[7]) != 0:
      sumTMAX += float(row[6]); #Adding to previous total
      sumTMIN += float(row[7]);  
    count += 1; #Increasing number of values each time
  
  avgPRCP = sumPRCP / count; #Calculating averages
  avgTMAX = sumTMAX / count;
  avgTMIN = sumTMIN / count;

  MEANS = { #Holds the mean values for three features
    'PRCP': avgPRCP, #Used to normalise values later
    'TMAX': avgTMAX, #Each value represented in relation 
    'TMIN': avgTMIN  #to extremes (high and low).
  };
  findingMeans.close();

def processContinuousData(data, mean):
  '''
  Normalises numerical data. 
  Input: (Tensor) - [-1, 1] data: Type of feature
  Input: (float) mean: Average of all data values
  '''
  # Normalize data
  data = tf.cast(data, tf.float32) * 1 / (2 * mean);
  #Tensor shape - no batch size specified, 1 column
  return tf.reshape(data, [-1, 1]);

#Create 1D tensor of feautures as input
def preprocess(features, labels):
  '''
  Matches normalised data to labels
  Input: (Tensor - 2D array) Features: Columns with 
  feature data - PRCP, TMAX, TMIN
  Input: (Tensor - 1D array) Labels: Column with 
  mosquito population number - Mosquito Monthly Count
  '''
  # Process all continuous features.
  for feature in MEANS.keys():
    features[feature] = processContinuousData(features[feature], MEANS[feature]);
  
  # Assemble features into a single tensor.
  features = tf.concat([features[column] for column in FEATURE_COLUMNS], 1);
  
  return features, labels;

#Shuffles and processes raw data
trainData = rawTrainData.map(preprocess).shuffle(500);
print(trainData);
testData = rawTestData.map(preprocess);

print("done preprocessing");

def getModel(inputDim, hiddenUnits = [100]):
  """Create a Keras model with layers.

  Input:
    inputDim: (int) The shape of an item in a batch. 
    labels_dim: (int) The shape of a label.
    hiddenUnits: [int] the layer sizes of the DNN (input layer first)
    learningRate: (float) the learning rate for the optimizer.

  Output:
    A Keras model.
  """
  #Input layer
  inputs = tf.keras.Input(shape = (inputDim,));
  x = inputs;
  
  #Hidden Layer
  for units in hiddenUnits:
    x = tf.keras.layers.Dense(units, activation='relu')(x);
  #Output Layer
  outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x);

  #Entire Model
  model = tf.keras.Model(inputs, outputs);
  return model;

print("start training");

#Creating optimiser
optimiser = tf.keras.optimizers.Adam(
  learning_rate = 0.0001,
  name = 'Adam'
);

model = getModel(3); #3 for input feature data categories
model.compile(
    loss = 'binary_crossentropy',
    optimizer = optimiser,
    metrics = ['accuracy']);

model.fit(trainData, epochs = 20); #Training model
print("done training")

#Testing model
testLoss, testAccuracy = model.evaluate(testData);
print('\n\nTest Loss {}, Test Accuracy {}'.format(testLoss, testAccuracy)); 

#Creating Predictions using new weights, biases
predictions = model.predict(testData);
print(predictions);
