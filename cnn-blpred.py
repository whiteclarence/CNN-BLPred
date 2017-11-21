#!/usr/bin/python

import sys, getopt, os, gc, pickle, feps
from shutil import copyfile, rmtree
from xgboost import XGBClassifier
import tensorflow as tf
import numpy as np
import tflearn
from tflearn.layers.conv import conv_1d, global_max_pool, max_pool_1d
from sklearn.feature_selection import SelectFromModel
import pandas as pd


def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print 'python cnn-blpred.py -i <inputfile> -o <outputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -i <inputfile> -o <outputfile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   #print 'Input file is "', inputfile
   #print 'Output file is "', outputfile
   if not os.path.exists('infile'):
       os.makedirs('infile')
   if not os.path.exists('outfile'):
       os.makedirs('outfile')
   copyfile(inputfile, 'infile/'+inputfile)
   print('Extracting CKSAAP features!')
   feps.feps('infile/','outfile/')
   copyfile('outfile/'+inputfile+'_full_features.csv', inputfile.split('.')[0]+'.csv')
   copyfile('outfile/'+inputfile+'_proteinFeaturesList.txt', inputfile.split('.')[0]+'.txt')
   rmtree('infile/')
   rmtree('outfile/')
   print('Reading Features!!')
   f = open(inputfile.split('.')[0]+'.txt', 'r')
   trainPFL = f.read().splitlines()
   f.close()
   df = pd.read_csv(inputfile.split('.')[0]+'.csv',index_col=False,names = trainPFL)
   filenames = ['BL_Level_1','BL_Class_A','BL_Class_B','BL_Class_C','BL_Class_D','BL_Group_2']
   for filename in filenames:
       print('Predicting '+filename+'!')
       f1 = open('models/feature_selection/'+filename+'_XGB_FS.pkl', 'rb')
       xgb = pickle.load(f1)
       f1.close()
       f1 = open('models/feature_selection/'+filename+'_vocabulary.pkl', 'rb')
       vocabulary = pickle.load(f1)
       f1.close()
       model = SelectFromModel(xgb, prefit=True)
       df_new = model.transform(df)

       input_layer = tflearn.input_data(shape=[None, df_new.shape[1]], name='input')
       embedding_layer = tflearn.embedding(input_layer, input_dim=vocabulary, output_dim=128, validate_indices=True)
       conv_layer = conv_1d(embedding_layer, 256, 4, padding='same', activation='tanh', regularizer='L2')
       maxpool_layer = max_pool_1d(conv_layer,2)
       dropout = tflearn.dropout(maxpool_layer, 0.5)
       softmax = tflearn.fully_connected(dropout, 2, activation='softmax')
       regression = tflearn.regression(softmax, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='target')
       clf = tflearn.DNN(regression, tensorboard_verbose=3)
       clf.load('models/classification/'+filename+'/'+filename+'_model.tfl')
       predicted = clf.predict_label(df_new)[:,1]
       score = clf.predict(df_new)
       if not os.path.exists('results'):
           os.makedirs('results')
       np.savetxt('results/'+outputfile+'_'+filename+'_predict_label.csv', predicted, delimiter = ',')
       np.savetxt('results/'+outputfile+'_'+filename+'_predict_score.csv', score, delimiter = ',')
       tf.reset_default_graph()
       del vocabulary, df_new, f1, input_layer, embedding_layer, conv_layer, maxpool_layer, dropout, softmax, regression, clf, predicted, score, model, xgb
       gc.collect()
   os.remove(inputfile.split('.')[0]+'.csv')
   os.remove(inputfile.split('.')[0]+'.txt')
   

if __name__ == "__main__":
   main(sys.argv[1:])
