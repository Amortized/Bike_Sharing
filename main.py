'''
    Author : Rahul M

    10/13/2014 : 9:54PM
'''

import sys;
import numpy as np;
np.set_printoptions(threshold=np.nan)
import math;
import pickle;
from sklearn.ensemble import GradientBoostingRegressor;
from sklearn.preprocessing import OneHotEncoder;
import csv;
import matplotlib.pyplot as plt;
from sklearn.ensemble.partial_dependence import plot_partial_dependence;

class IO(object):
    def __init__(self, s):
        self.s = s;

    def parse(self):
        data = self.s.rstrip().split('\n');
        content = []
        for d in data[1::]:
            content.append(d.split(','));

        return data[0].split(','), content



class Model(object):
    def __init__(self, schema, input_data, mode):
        self.schema = schema
        #Model param
        self.model_param = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 1, \
                            'learning_rate': 0.1, 'loss': 'ls', 'verbose' : 1}
        if mode == 0:
            #Train
            self.data_label ,  self.data_features = self.preprocess(input_data, mode)
        else:
            #Test - We don't have the labels.
            self.data_features = self.preprocess(input_data, mode)

    def preprocess(self, data, mode):

        if mode == 0:
          #Ensure label is not part of features
          feature_range = len(data[0]) - 1
          #Extract the label for train
          try :
            label = np.array([float(ele[11]) for ele in data])
          except :
            print "Test file needs to have mode 1"
            sys.exit(1)
        elif mode == 1:
            #There is no label for test set
            feature_range = len(data[0])

        features = []
        for ele in data:
            dp = []
            for x in range(0, feature_range):
                if x in [0,9,10]:
                    #Need to transform datetime
                    pass
                elif x in [1,2,3,4]:
                    #Categorical features
                    dp.append(int(ele[x]))
                else :
                    #Numerical features
                    dp.append(float(ele[x]))
            features.append(dp)

        #Update the schema accordingly
        self.schema = ['season1', 'season2', 'season3', 'season4', 'holiday1', 'holiday2', 'weekend1', 'weekend2', \
                       'weather1', 'weather2', 'weather3', 'weather4', 'temp', 'atemp', 'humidity', 'windspeed']


        #One-hot encoding on categorical features
        params = {'categorical_features' : np.array([0,1,2,3])}
        enc = OneHotEncoder(**params)
        features = enc.fit_transform(features).toarray()

        if mode == 0 :
           return  label, features
        else:
           return features

    def train(self):
        model = GradientBoostingRegressor(**self.model_param)
        model.fit(self.data_features, self.data_label)

        #Plot Feature Importance
        feature_importance = 100.0 * (model.feature_importances_ / model.feature_importances_.max())
        print feature_importance
        print self.schema

        #PDP
        fig, axs = plot_partial_dependence(model, self.data_features, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        return  model

    def predict(self, model):
        y_predict = model.predict(self.data_features)
        #Handle Negative Predictions
        for i in range(0, len(y_predict)):
            if y_predict[i] < 0:
                y_predict[i] = 0.0
        return y_predict;

    def evaluate(self, y_predict):
        rmsle = 0.0
        for i in range(0, len(y_predict)):
            rmsle = math.pow((math.log(y_predict[i] + 1) - math.log(self.data_label[i] + 1)), 2)
        rmsle = math.sqrt( (rmsle / float(len(y_predict))) )
        print "Root Mean Squared Log Error " +  str(rmsle);





if __name__ == '__main__':
    args = list(sys.argv[1:])
    if len(args) < 2:
        print "For Training -> python main.py 0 <train-file> <model_object>"
        print "For Test     -> python main.py 1 <test-file> <model_object> <output-file>"
        sys.exit(1)

    if int(args[0]) in [0,1]:
        #Read the file
        file_handler     = args[1];
        #Read
        with open(file_handler) as f:
          schema, content = IO(f.read()).parse();

        #Build a model object
        m = Model(schema, content, int(args[0]))
        if int(args[0]) == 0 :
          model_obj = m.train()
          y_predict = m.predict(model_obj)
          m.evaluate(y_predict)
          #Save the object"
          pickle.dump(model_obj, open(args[2], "wb"))
        else:
          #Load the Model
          model_obj = pickle.load( open(args[2], "rb") )
          y_predict = m.predict(model_obj)
          output = open(args[3], 'w')
          output.write("datetime,count\n")
          for i in range(0, len(y_predict)):
              output.write(str(content[i][0]) +"," + str(y_predict[i]) + "\n")
          output.close()



    else:
        print "Invalid Input"












