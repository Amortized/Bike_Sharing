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
import time

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
    def __init__(self, schema, input_data, mode, one_hot_encoding):
        self.schema = schema
        #Model param
        self.model_param = {'n_estimators': 150, 'max_features': 0.90, 'max_depth': 6, 'subsample': 0.9, \
            'learning_rate': 0.05, 'loss': 'ls', 'verbose' : 1, 'min_samples_leaf': 30}
        if mode in [0,2]:
            #Train/validate
            self.data_label ,  self.data_features = self.preprocess(input_data, mode, one_hot_encoding)
        else:
            #Test - We don't have the labels.
            self.data_features = self.preprocess(input_data, mode, one_hot_encoding)

    def preprocess(self, data, mode, one_hot_encoding=None):

        if mode in [0,2]:
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
                if x in [9,10]:
                    pass
                elif x == 0:
                    parse_dt = time.strptime(ele[x], "%Y-%m-%d %H:%M:%S")
                    #Create new features
                    dp.append(int(parse_dt.tm_hour)) #Time of hour
                    dp.append(int(parse_dt.tm_wday)) #Week day
                    
                    
                      
                    #Bin the time period based on day
                    if int(parse_dt.tm_hour) >= 0 and int(parse_dt.tm_hour) < 6:
                        dp.append(1)
                    elif int(parse_dt.tm_hour) >= 6 and int(parse_dt.tm_hour) <= 9:
                        dp.append(2)
                    elif int(parse_dt.tm_hour) > 10 and int(parse_dt.tm_hour) <= 16:
                        dp.append(3)
                    elif int(parse_dt.tm_hour) > 16 and int(parse_dt.tm_hour) <= 19:
                        dp.append(4)
                    else:
                        dp.append(5)
                    
                    

                    #Year
                    dp.append(int(parse_dt.tm_year))
                    
                    #Bind based on time + working_day
                    if int(ele[3]) == 1 and (int(parse_dt.tm_hour) in [7,8,17,18,19]):
                        dp.append(1) #Peak hour
                    else:
                        dp.append(0) #Non Peak hour
                        
                    #Bind based on time _ non working day
                    if int(ele[3]) == 0 and (int(parse_dt.tm_hour) in [10,11,12,13,14,15,16,17,18,19]):
                        dp.append(1)  #Peak hour
                    else:
                        dp.append(0)  #Non Peak hour   
                    
                        
                     
                elif x in [1,2,3,4]:
                    #Categorical features
                    dp.append(int(ele[x]))
                                        
                else :
                    #Numerical features
                    dp.append(float(ele[x]))
            features.append(dp)
        
        
        #Update the schema accordingly
        self.schema = ['hr1', 'hr2', 'hr3', 'hr4', 'hr5', 'hr6', \
                       'hr7', 'hr8', 'hr9', 'hr10', 'hr11', 'hr12', \
                       'hr13', 'hr14', 'hr15', 'hr16', 'hr17', 'hr18', \
                       'hr19', 'hr20', 'hr21', 'hr22', 'hr23', 'hr24', \
                       'tw1', 'tw2', 'tw3', 'tw4', 'tw5', 'tw6', 'tw7', \
                       'tb1', 'tb2', 'tb3', 'tb4', 'tb5', \
                       'y1', 'y2', \
                       'working_day_peak1', 'working_day_peak2', \
                       'non_working_day_peak1', 'non_working_day_peak2', \
                       'season1', 'season2', 'season3', 'season4', 'holiday1', 'holiday2', 'workingday1', 'workingday2', \
                       'weather1', 'weather2', 'weather3', 'weather4', 'temp', 'atemp', 'humidity', 'windspeed']


        #One-hot encoding on categorical features
        params = {'categorical_features' : np.array([0,1,2,3,4,5,6,7,8,9])}
        enc = OneHotEncoder(**params)
        
         
        if mode == 0 :
            #In Train mode, fit and dump
            enc_o = enc.fit(features)
            pickle.dump(enc_o, open(one_hot_encoding, "wb"))
        elif mode in [1,2] :
            #Validate and Test work off the same encoding as train
            enc_o = pickle.load( open(one_hot_encoding, "rb") )
            
        
        features = enc_o.transform(features).toarray()
        if mode in [0,2] :
           return  label, features
        else:
           return features

    def train(self):
        model = GradientBoostingRegressor(**self.model_param)
        model.fit(self.data_features, self.data_label)

        #Plot Feature Importance
        feature_importance = 100.0 * (model.feature_importances_ / model.feature_importances_.max())
        sorted_idx = np.argsort(feature_importance)[::-1]
    
        
        print(" Feature Importance ...")
        for idx in sorted_idx :
            print(str(feature_importance[idx]) + " : " + self.schema[idx] + " ")
        print("\n ")


        #PDP
        fig, axs = plot_partial_dependence(model, self.data_features, [12, 13, 14, 15])
        
        plt.subplots_adjust(top=0.9)
        fig.savefig('pdp.png')

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
            rmsle += math.pow((math.log(y_predict[i] + 1) - math.log(self.data_label[i] + 1)), 2)
        rmsle = math.sqrt( (rmsle / float(len(y_predict))) )
        print "Root Mean Squared Log Error " +  str(rmsle);





if __name__ == '__main__':
    args = list(sys.argv[1:])
    if len(args) < 2:
        print "For Training -> python main.py 0 <train-file> <model_object> <one-hot-encoding>"
        print "For validation -> python main.py 2 <validate-file> <model_object> <one-hot-encoding>"
        print "For Test     -> python main.py 1 <test-file> <model_object> <one-hot-encoding> <output-file>"
        sys.exit(1)

    if int(args[0]) in [0,1,2]:
        #Read the file
        file_handler     = args[1];
        #Read
        with open(file_handler) as f:
          schema, content = IO(f.read()).parse();

        #Build a model object
        m = Model(schema, content, int(args[0]), args[3])
        if int(args[0]) == 0 :
          model_obj = m.train()
          y_predict = m.predict(model_obj)
          print "Training ... "
          m.evaluate(y_predict)
          #Save the object"
          pickle.dump(model_obj, open(args[2], "wb"))      
        else:
          #Load the Model
          model_obj = pickle.load( open(args[2], "rb") )
          y_predict = m.predict(model_obj)
          if int(args[0]) == 2:
            print "Validation ..."
            m.evaluate(y_predict)
          else:    
            output = open(args[4], 'w')
            output.write("datetime,count\n")
            for i in range(0, len(y_predict)):
              output.write(str(content[i][0]) +"," + str(y_predict[i]) + "\n")
            output.close()

    else:
        print "Invalid Input"












