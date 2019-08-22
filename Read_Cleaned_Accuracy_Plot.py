####Model that is overfitting for LR and SVM
##Loads all pickles and then creates models

import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics as mt
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
#from reportlab.pdfgen import canvas
#from reportlab.lib.pagesizes import letter

#from sklearn.metrics import precision_score, recall_score, roc_curve, auc

csv.field_size_limit(100000000)
start = time.time()

# master_dict = {} #kmer are keys, values are another dictionary, where they keys are the isolate and the value is the count
# master_dict['A']['485.317']
# for res in res_file_list:
#   #for each file check
#with open('/Users/i868265/Desktop/AMR Data/'+res_file_list[0]+'.fna', 'r') as f:
#  data = list(csv.reader(f))

# wget all files
# wget --recursive --no-parent ftp://ftp.patricbrc.org/patric2/current_release/AMR_genome_sets/Neisseria/
# read all files in a directory

import os

debug = False

#sus_dir = './SUS/'
#res_dir = './RES/'
#392 = Total # of files/isolates/individual cases of gonorrhea
#178= Susceptible
#214= Resistant

sus_dir = 'C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\SUS'
res_dir = 'C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\RES'

# Typical Programming Approach
sus_files = []
for a_file in os.listdir(sus_dir):
  if a_file[-1] != 'b':
    sus_files.append(a_file)

# pythonic syntax
sus_files = [f for f in os.listdir(sus_dir) if f[-1] != 'b']
res_files = [f for f in os.listdir(res_dir) if f[-1] != 'b']

#all_isolates = ['./SUS/'+ x for x in sus_files] + ['./RES/'+ x for x in res_files]
all_isolates = ['C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\SUS\\'+ x for x in sus_files] + ['C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\RES\\'+ x for x in res_files]

total_number_isolates = len(sus_files)+len(res_files)

L = [1  if x >= len(sus_files) else 0 for x in range(len(sus_files)+len(res_files)) ]
# 1 = resistant and 0 = susceptible
if debug:
  if total_number_isolates != len(L):
    print("YOU HAVE A BUG")
    import pdb; pdb.set_trace()

master_dict = {}
# keys are kmers
# values are a dictionary where  keys are position  of isolate in all_isolates
# Value is count value of count
#k = 7 # kmer size
k_len = 10
#Create variable with isolate name and length
isolate_info = []

for idf, a_file in enumerate(all_isolates):
  with open(a_file, 'r') as f:
    data = list(csv.reader(f))
  data = [d for d in data if d]
  print(a_file)
  contigs = []
  tmp = ''
  for x in range(len(data)):
    if data[x][0][0]=='>':
      contigs.append(tmp)
      tmp = ''
    else:
      tmp += data[x][0]

  #if kmer known
  #  if kmer already in isolate
  #  else show kmer to isloate
  #else:
  #  know kmer
  #  know kmer in isolate  
  for contig in contigs:
    end_pos = len(contig)-k_len
    for x in range(end_pos):
      kmer = contig[x:x+k_len]
      if kmer in master_dict:
        if idf in master_dict[kmer]:
          master_dict[kmer][idf]+=1
        else:
          master_dict[kmer][idf] = 1
      elif 'n' not in kmer:
        master_dict[kmer] = {}
        master_dict[kmer][idf] = 1
  print(sum([len(c) for c in contigs]))
  isolate_info.append(a_file + ' ' + str(sum([len(c) for c in contigs])))
      #k-=1    
#import pdb; pdb.set_trace()

list_all_files = []
#with open('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\SUS\\485.317.fna', 'r') as f:
#  data = list(csv.reader(f))

data = [d for d in data if d]

num_str = len(data)

contigs = []
tmp = ''
for x in range(num_str):
  if data[x][0][0]=='>':
    contigs.append(tmp)
    tmp = ''
  else:
    tmp += data[x][0]
contigs.append(tmp)

contigs = [x for x in contigs if x]  

def make_kmer(k, the_strings):
    ret = {}
    for a_string in the_strings:
        end_pos = len(a_string)-k
        for x in range(end_pos):
            if a_string[x:x+k] in ret:
                ret[a_string[x:x+k]]+=1
            else:
                ret[a_string[x:x+k]]=1
    return ret

#convert plot number to string
plot_num_str = str(k_len) 
#set file name to master_dict with the length of k
master_dict_file_Name = "master_dict_k" + plot_num_str
# open the file for writing
#fileObject = open(master_dict_file_Name,'wb') 
# this writes the object a to the
# file named 'testfile'
#pickle.dump(master_dict,fileObject)   
# here we close the fileObject
#fileObject.close()

# graph of number of kmers by size in k for susceptible and resistant
# graph illustrating largest variances in size?
# table of file counts for resistant/susceptible
# range of total contigs within resistant and susceptible

zz= make_kmer(k_len, contigs)
kmer_index = list(zz.keys())
number_kmers = len(zz.keys())

M = np.zeros([len(all_isolates), len(master_dict)], dtype=np.int64)
M10 = M
for column, kmer_string in enumerate(master_dict.keys()):
    for row in master_dict[kmer_string].keys():
        #print(master_dict[kmer_string].keys())
        M[row, column] = master_dict[kmer_string][row]

#convert plot number to string
plot_num_str = str(k_len) 
#set file name to master_dict with the length of k
M_file_Name = "M" + plot_num_str
# open the file for writing
M_fileObject = open(M_file_Name,'wb') 
# this writes the object a to the
# file named 'testfile'
#pickle.dump(M,M_fileObject)   


skf = StratifiedKFold(n_splits=5, shuffle=True)

M5_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M5.pickle')
M6_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M6.pickle')
M7_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M7.pickle')
#M8_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M8.pickle')
#M9_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M9.pickle')


M5 = pickle.load(open(M5_pickle, 'rb'))
M6 = pickle.load(open(M6_pickle, 'rb'))
M7 = pickle.load(open(M7_pickle, 'rb'))
#M8 = pickle.load(open(M8_pickle, 'rb'))
#M9 = pickle.load(open(M9_pickle, 'rb'))



train_results = []
test_results = []

L = np.array(L)
L = L.astype(int)

rf_models_M5 = []
rf_models_M6 = []
rf_models_M7 = []
rf_models_M8 = []
rf_models_M9 = []
rf_models_M10 = []



#Train Random Forest model
for rf_train_index_M10, rf_test_index_M10 in skf.split(M10, L):
    rf_train_M10, rf_test_M10 = M10[rf_train_index_M10], M10[rf_test_index_M10]
    rf_train_labels_M10, rf_test_labels_M10 = L[rf_train_index_M10], L[rf_test_index_M10]
    RSEED = 50
    rf_clf_M10 = RandomForestClassifier(n_estimators=10, 
                               random_state=RSEED, 
                               max_features = 'sqrt',
                               n_jobs=-1, 
                               class_weight="balanced",
                               verbose = 1)
    # Fit on training data
    rf_clf_M10.fit(rf_train_M10, rf_train_labels_M10)
    # Class predictions
    rf_predictions_M10 = rf_clf_M10.predict(rf_test_M10)
    # Probabilities for classes
    rf_accuracy_M10 = round(mt.accuracy_score(rf_test_labels_M10, rf_predictions_M10),2)
    rf_models_M10.append((rf_accuracy_M10))
    rf_conf_M10 = mt.confusion_matrix(rf_test_labels_M10,rf_predictions_M10)
    #Plot SVM AUC Curve
    rf_fpr_M10, svm_tpr_M10, thresholds_M10 = mt.roc_curve(rf_test_labels_M10, rf_predictions_M10)
    rf_roc_auc_M10 = mt.roc_auc_score(rf_test_labels_M10, rf_predictions_M10)


svm_models_M10 = []
#SVM Model
for svm_train_index_M10, svm_test_index_M10 in skf.split(M10, L):
    svm_train_M10, svm_test_M10 = M10[svm_train_index_M10], M10[svm_test_index_M10]
    svm_train_labels_M10, svm_test_labels_M10 = L[svm_train_index_M10], L[svm_test_index_M10]
    RSEED = 50
    #svm_clf_M10 = svm.SVC(gamma='auto', C=100, kernel='linear')
    svm_clf_M10 = svm.SVC(gamma = 'auto', C = 1)
    svm_clf_M10.fit(svm_train_M10, svm_train_labels_M10)  
    svm_predictions_M10 = svm_clf_M10.predict(svm_test_M10)
    svm_accuracy_M10 = mt.accuracy_score(svm_test_labels_M10,svm_predictions_M10)
    svm_models_M10.append((svm_accuracy_M10))
    svm_conf_M10 = mt.confusion_matrix(svm_test_labels_M10,svm_predictions_M10)
    #print(f'SVM Accuracy: {svm_accuracy_M10}')
    #print(mt.classification_report(test_labels,svm_predictions_M10))
    #Plot SVM AUC Curve
    svm_fpr_M10, svm_tpr_M10, thresholds_M10 = mt.roc_curve(svm_test_labels_M10, svm_predictions_M10)
    svm_roc_auc_M10 = mt.roc_auc_score(svm_test_labels_M10, svm_predictions_M10)

#NB Model
nb_models_M10 =[]
for train_index_M10, test_index_M10 in skf.split(M, L):
    train_M10, test_M10 = M[train_index_M10], M[test_index_M10]
    train_labels_M10, test_labels_M10 = L[train_index_M10], L[test_index_M10]
    RSEED = 50
    nb_clf_M10 = GaussianNB()
    nb_clf_M10.fit(train_M10, train_labels_M10)  
    nb_predictions_M10 = nb_clf_M10.predict(test_M10)
    nb_accuracy_M10 = mt.accuracy_score(test_labels_M10,nb_predictions_M10)
    nb_models_M10.append((nb_accuracy_M10))
    nb_conf_M10 = mt.confusion_matrix(test_labels_M10,nb_predictions_M10)
    #print(f'NB Accuracy: {nb_accuracy}')    
    #print(mt.classification_report(test_labels,nb_predictions_M10))
    #Plot SVM AUC Curve
    nb_fpr_M10, nb_tpr_M10, thresholds_M10 = mt.roc_curve(test_labels_M10, nb_predictions_M10)
    nb_roc_auc_M10 = mt.roc_auc_score(test_labels_M10, nb_predictions_M10)
