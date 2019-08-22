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
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score,recall_score, roc_curve
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
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


#https://stackoverflow.com/questions/45969390/difference-between-stratifiedkfold-and-stratifiedshufflesplit-in-sklearn
#https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
RSEED = 75
#skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=RSEED)
skf = StratifiedShuffleSplit(test_size=0.8)

M5_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M5.pickle')
M6_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M6.pickle')
M7_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M7.pickle')
M8_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M8.pickle')
M9_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M9.pickle')
#M10_pickle = ('C:\\Users\\lingl\\Documents\\Data Science\\Capstone_Project\\M10.pickle')


M5 = pickle.load(open(M5_pickle, 'rb'))
M6 = pickle.load(open(M6_pickle, 'rb'))
M7 = pickle.load(open(M7_pickle, 'rb'))
M8 = pickle.load(open(M8_pickle, 'rb'))
M9 = pickle.load(open(M9_pickle, 'rb'))
#M10 = pickle.load(open(M10_pickle, 'rb'))

#train_results = []
#test_results = []

L = np.array(L)
L = L.astype(int)

##Models for k=5
svm_models_M5 = []
#SVM Model
for svm_train_index_M5, svm_test_index_M5 in skf.split(M5, L):
    svm_train_M5, svm_test_M5 = M5[svm_train_index_M5], M5[svm_test_index_M5]
    svm_train_labels_M5, svm_test_labels_M5 = L[svm_train_index_M5], L[svm_test_index_M5]
    RSEED = 50
    #svm_clf_M5 = svm.SVC(gamma='auto', C=100, kernel='linear')
    svm_clf_M5 = svm.SVC(gamma='auto', C=1, random_state = RSEED)
    svm_clf_M5.fit(svm_train_M5, svm_train_labels_M5)  
    svm_predictions_M5 = svm_clf_M5.predict(svm_test_M5)
    svm_accuracy_M5 = mt.accuracy_score(svm_test_labels_M5,svm_predictions_M5)
    svm_models_M5.append((svm_accuracy_M5))
    svm_conf_M5 = mt.confusion_matrix(svm_test_labels_M5,svm_predictions_M5)
    #print(f'SVM Accuracy: {svm_accuracy_M5}')
    #print(mt.classification_report(test_labels,svm_predictions_M5))
    #Plot SVM AUC Curve
    svm_fpr_M5, svm_tpr_M5, thresholds_M5 = mt.roc_curve(svm_test_labels_M5, svm_predictions_M5)
    svm_roc_auc_M5 = mt.roc_auc_score(svm_test_labels_M5, svm_predictions_M5)

svm_models_M6 = []
#SVM Model
for svm_train_index_M6, svm_test_index_M6 in skf.split(M6, L):
    svm_train_M6, svm_test_M6 = M6[svm_train_index_M6], M6[svm_test_index_M6]
    svm_train_labels_M6, svm_test_labels_M6 = L[svm_train_index_M6], L[svm_test_index_M6]
    RSEED = 50
    #svm_clf_M6 = svm.SVC(gamma='auto', C=100, kernel='linear')
    svm_clf_M6 = svm.SVC(gamma='auto', C=1, random_state = RSEED)
    svm_clf_M6.fit(svm_train_M6, svm_train_labels_M6)  
    svm_predictions_M6 = svm_clf_M6.predict(svm_test_M6)
    svm_accuracy_M6 = mt.accuracy_score(svm_test_labels_M6,svm_predictions_M6)
    svm_models_M6.append((svm_accuracy_M6))
    svm_conf_M6 = mt.confusion_matrix(svm_test_labels_M6,svm_predictions_M6)
    #print(f'SVM Accuracy: {svm_accuracy_M6}')
    #print(mt.classification_report(test_labels,svm_predictions_M6))
    #Plot SVM AUC Curve
    svm_fpr_M6, svm_tpr_M6, thresholds_M6 = mt.roc_curve(svm_test_labels_M6, svm_predictions_M6)
    svm_roc_auc_M6 = mt.roc_auc_score(svm_test_labels_M6, svm_predictions_M6)

svm_models_M7 = []
#SVM Model
for svm_train_index_M7, svm_test_index_M7 in skf.split(M7, L):
    svm_train_M7, svm_test_M7 = M7[svm_train_index_M7], M7[svm_test_index_M7]
    svm_train_labels_M7, svm_test_labels_M7 = L[svm_train_index_M7], L[svm_test_index_M7]
    RSEED = 50
    #svm_clf_M7 = svm.SVC(gamma='auto', C=100, kernel='linear')
    svm_clf_M7 = svm.SVC(gamma='auto', C=1, random_state = RSEED)
    svm_clf_M7.fit(svm_train_M7, svm_train_labels_M7)  
    svm_predictions_M7 = svm_clf_M7.predict(svm_test_M7)
    svm_accuracy_M7 = mt.accuracy_score(svm_test_labels_M7,svm_predictions_M7)
    svm_models_M7.append((svm_accuracy_M7))
    svm_conf_M7 = mt.confusion_matrix(svm_test_labels_M7,svm_predictions_M7)
    #print(f'SVM Accuracy: {svm_accuracy_M7}')
    #print(mt.classification_report(test_labels,svm_predictions_M7))
    #Plot SVM AUC Curve
    svm_fpr_M7, svm_tpr_M7, thresholds_M7 = mt.roc_curve(svm_test_labels_M7, svm_predictions_M7)
    svm_roc_auc_M7 = mt.roc_auc_score(svm_test_labels_M7, svm_predictions_M7)

svm_models_M8 = []
#SVM Model
for svm_train_index_M8, svm_test_index_M8 in skf.split(M8, L):
    svm_train_M8, svm_test_M8 = M8[svm_train_index_M8], M8[svm_test_index_M8]
    svm_train_labels_M8, svm_test_labels_M8 = L[svm_train_index_M8], L[svm_test_index_M8]
    RSEED = 50
    #svm_clf_M8 = svm.SVC(gamma='auto', C=100, kernel='linear')
    svm_clf_M8 = svm.SVC(gamma='auto', C=1, random_state = RSEED)
    svm_clf_M8.fit(svm_train_M8, svm_train_labels_M8)  
    svm_predictions_M8 = svm_clf_M8.predict(svm_test_M8)
    svm_accuracy_M8 = mt.accuracy_score(svm_test_labels_M8,svm_predictions_M8)
    svm_models_M8.append((svm_accuracy_M8))
    svm_conf_M8 = mt.confusion_matrix(svm_test_labels_M8,svm_predictions_M8)
    #print(f'SVM Accuracy: {svm_accuracy_M8}')
    #print(mt.classification_report(test_labels,svm_predictions_M8))
    #Plot SVM AUC Curve
    svm_fpr_M8, svm_tpr_M8, thresholds_M8 = mt.roc_curve(svm_test_labels_M8, svm_predictions_M8)
    svm_roc_auc_M8 = mt.roc_auc_score(svm_test_labels_M8, svm_predictions_M8)

svm_models_M9 = []
#SVM Model
for svm_train_index_M9, svm_test_index_M9 in skf.split(M9, L):
    svm_train_M9, svm_test_M9 = M9[svm_train_index_M9], M9[svm_test_index_M9]
    svm_train_labels_M9, svm_test_labels_M9 = L[svm_train_index_M9], L[svm_test_index_M9]
    RSEED = 50
    #svm_clf_M9 = svm.SVC(gamma='auto', C=100, kernel='linear')
    svm_clf_M9 = svm.SVC(gamma='auto', C=1, random_state = RSEED)
    svm_clf_M9.fit(svm_train_M9, svm_train_labels_M9)  
    svm_predictions_M9 = svm_clf_M9.predict(svm_test_M9)
    svm_accuracy_M9 = mt.accuracy_score(svm_test_labels_M9,svm_predictions_M9)
    svm_models_M9.append((svm_accuracy_M9))
    svm_conf_M9 = mt.confusion_matrix(svm_test_labels_M9,svm_predictions_M9)
    #print(f'SVM Accuracy: {svm_accuracy_M9}')
    #print(mt.classification_report(test_labels,svm_predictions_M9))
    #Plot SVM AUC Curve
    svm_fpr_M9, svm_tpr_M9, thresholds_M9 = mt.roc_curve(svm_test_labels_M9, svm_predictions_M9)
    svm_roc_auc_M9 = mt.roc_auc_score(svm_test_labels_M9, svm_predictions_M9)


##CHECKING THIS OUT
svm_models_M10 = []
#SVM Model
for svm_train_index_M10, svm_test_index_M10 in skf.split(M10, L):
    svm_train_M10, svm_test_M10 = M10[svm_train_index_M10], M10[svm_test_index_M10]
    svm_train_labels_M10, svm_test_labels_M10 = L[svm_train_index_M10], L[svm_test_index_M10]
    RSEED = 50
    #svm_clf_M10 = svm.SVC(gamma='auto', C=100, kernel='linear')
    svm_clf_M10 = svm.SVC(gamma='auto', C=1, random_state = RSEED)
    svm_clf_M10.fit(svm_train_M10, svm_train_labels_M10)  
    svm_predictions_M10 = svm_clf_M10.predict(svm_test_M10)
    svm_accuracy_M10 = mt.accuracy_score(svm_test_labels_M10,svm_predictions_M10)
    svm_models_M10.append((svm_accuracy_M10))
    svm_conf_M10 = mt.confusion_matrix(svm_test_labels_M10,svm_predictions_M10)
    #print(f'SVM Accuracy: {svm_accuracy_M10}')
    print(mt.classification_report(svm_test_labels_M10,svm_predictions_M10))
    #Plot SVM AUC Curve
    svm_fpr_M10, svm_tpr_M10, thresholds_M10 = mt.roc_curve(svm_test_labels_M10, svm_predictions_M10)
    svm_roc_auc_M10 = mt.roc_auc_score(svm_test_labels_M10, svm_predictions_M10)

##Random Forest Models
rf_models_M5 = []
#Train Random Forest model
for rf_train_index_M5, rf_test_index_M5 in skf.split(M5, L):
    rf_train_M5, rf_test_M5 = M5[rf_train_index_M5], M5[rf_test_index_M5]
    rf_train_labels_M5, rf_test_labels_M5 = L[rf_train_index_M5], L[rf_test_index_M5]
    RSEED = 50
    rf_clf_M5 = RandomForestClassifier(n_estimators=10, 
                               random_state=RSEED)
    # Fit on training data
    rf_clf_M5.fit(rf_train_M5, rf_train_labels_M5)
    # Class predictions
    rf_predictions_M5 = rf_clf_M5.predict(rf_test_M5)
    # Probabilities for classes
    rf_accuracy_M5 = round(mt.accuracy_score(rf_test_labels_M5, rf_predictions_M5),2)
    rf_models_M5.append((rf_accuracy_M5))
    rf_conf_M5 = mt.confusion_matrix(rf_test_labels_M5,rf_predictions_M5)
    #Plot RF AUC Curve
    rf_fpr_M5, rf_tpr_M5, thresholds_M5 = mt.roc_curve(rf_test_labels_M5, rf_predictions_M5)
    rf_roc_auc_M5 = mt.roc_auc_score(rf_test_labels_M5, rf_predictions_M5)

rf_models_M6 = []
##Models for k=6
#Train Random Forest model
for rf_train_index_M6, rf_test_index_M6 in skf.split(M6, L):
    rf_train_M6, rf_test_M6 = M6[rf_train_index_M6], M6[rf_test_index_M6]
    rf_train_labels_M6, rf_test_labels_M6 = L[rf_train_index_M6], L[rf_test_index_M6]
    RSEED = 50
    rf_clf_M6 = RandomForestClassifier(n_estimators=10, 
                               random_state=RSEED)
    # Fit on training data
    rf_clf_M6.fit(rf_train_M6, rf_train_labels_M6)
    # Class predictions
    rf_predictions_M6 = rf_clf_M6.predict(rf_test_M6)
    # Probabilities for classes
    rf_accuracy_M6 = round(mt.accuracy_score(rf_test_labels_M6, rf_predictions_M6),2)
    rf_models_M6.append((rf_accuracy_M6))
    rf_conf_M6 = mt.confusion_matrix(rf_test_labels_M6,rf_predictions_M6)
    #Plot RF AUC Curve
    rf_fpr_M6, rf_tpr_M6, thresholds_M6 = mt.roc_curve(rf_test_labels_M6, rf_predictions_M6)
    rf_roc_auc_M6 = mt.roc_auc_score(rf_test_labels_M6, rf_predictions_M6)

rf_models_M7 = []
##Models for k=7
#Train Random Forest model
for rf_train_index_M7, rf_test_index_M7 in skf.split(M7, L):
    rf_train_M7, rf_test_M7 = M7[rf_train_index_M7], M7[rf_test_index_M7]
    rf_train_labels_M7, rf_test_labels_M7 = L[rf_train_index_M7], L[rf_test_index_M7]
    RSEED = 50
    rf_clf_M7 = RandomForestClassifier(n_estimators=10, 
                               random_state=RSEED)
    # Fit on training data
    rf_clf_M7.fit(rf_train_M7, rf_train_labels_M7)
    # Class predictions
    rf_predictions_M7 = rf_clf_M7.predict(rf_test_M7)
    # Probabilities for classes
    rf_accuracy_M7 = round(mt.accuracy_score(rf_test_labels_M7, rf_predictions_M7),2)
    rf_models_M7.append((rf_accuracy_M7))
    rf_conf_M7 = mt.confusion_matrix(rf_test_labels_M7,rf_predictions_M7)
    #Plot RF AUC Curve
    rf_fpr_M7, rf_tpr_M7, thresholds_M7 = mt.roc_curve(rf_test_labels_M7, rf_predictions_M7)
    rf_roc_auc_M7 = mt.roc_auc_score(rf_test_labels_M7, rf_predictions_M7)

rf_models_M8 = []
##Models for k=8
#Train Random Forest model
for rf_train_index_M8, rf_test_index_M8 in skf.split(M8, L):
    rf_train_M8, rf_test_M8 = M8[rf_train_index_M8], M8[rf_test_index_M8]
    rf_train_labels_M8, rf_test_labels_M8 = L[rf_train_index_M8], L[rf_test_index_M8]
    RSEED = 50
    rf_clf_M8 = RandomForestClassifier(n_estimators=10, 
                               random_state=RSEED)
    # Fit on training data
    rf_clf_M8.fit(rf_train_M8, rf_train_labels_M8)
    # Class predictions
    rf_predictions_M8 = rf_clf_M8.predict(rf_test_M8)
    # Probabilities for classes
    rf_accuracy_M8 = round(mt.accuracy_score(rf_test_labels_M8, rf_predictions_M8),2)
    rf_models_M8.append((rf_accuracy_M8))
    rf_conf_M8 = mt.confusion_matrix(rf_test_labels_M8,rf_predictions_M8)
    #Plot RF AUC Curve
    rf_fpr_M8, rf_tpr_M8, thresholds_M8 = mt.roc_curve(rf_test_labels_M8, rf_predictions_M8)
    rf_roc_auc_M8 = mt.roc_auc_score(rf_test_labels_M8, rf_predictions_M8)

rf_models_M9 = []
##Models for k=9
#Train Random Forest model
for rf_train_index_M9, rf_test_index_M9 in skf.split(M9, L):
    rf_train_M9, rf_test_M9 = M9[rf_train_index_M9], M9[rf_test_index_M9]
    rf_train_labels_M9, rf_test_labels_M9 = L[rf_train_index_M9], L[rf_test_index_M9]
    RSEED = 50
    rf_clf_M9 = RandomForestClassifier(n_estimators=10, 
                               random_state=RSEED )
    # Fit on training data
    rf_clf_M9.fit(rf_train_M9, rf_train_labels_M9)
    # Class predictions
    rf_predictions_M9 = rf_clf_M9.predict(rf_test_M9)
    # Probabilities for classes
    rf_accuracy_M9 = round(mt.accuracy_score(rf_test_labels_M9, rf_predictions_M9),2)
    rf_models_M9.append((rf_accuracy_M9))
    rf_conf_M9 = mt.confusion_matrix(rf_test_labels_M9,rf_predictions_M9)
    #Plot RF AUC Curve
    rf_fpr_M9, rf_tpr_M9, thresholds_M9 = mt.roc_curve(rf_test_labels_M9, rf_predictions_M9)
    rf_roc_auc_M9 = mt.roc_auc_score(rf_test_labels_M9, rf_predictions_M9)

rf_models_M10 = []
##Models for k=10
#Train Random Forest model
for rf_train_index_M10, rf_test_index_M10 in skf.split(M10, L):
    rf_train_M10, rf_test_M10 = M10[rf_train_index_M10], M10[rf_test_index_M10]
    rf_train_labels_M10, rf_test_labels_M10 = L[rf_train_index_M10], L[rf_test_index_M10]
    RSEED = 50
    rf_clf_M10 = RandomForestClassifier(n_estimators=10, 
                               random_state=RSEED)
    # Fit on training data
    rf_clf_M10.fit(rf_train_M10, rf_train_labels_M10)
    # Class predictions
    rf_predictions_M10 = rf_clf_M10.predict(rf_test_M10)
    # Probabilities for classes
    rf_accuracy_M10 = round(mt.accuracy_score(rf_test_labels_M10, rf_predictions_M10),2)
    rf_models_M10.append((rf_accuracy_M10))
    rf_conf_M10 = mt.confusion_matrix(rf_test_labels_M10,rf_predictions_M10)
    #Plot RF AUC Curve
    rf_fpr_M10, rf_tpr_M10, thresholds_M10 = mt.roc_curve(rf_test_labels_M10, rf_predictions_M10)
    rf_roc_auc_M10 = mt.roc_auc_score(rf_test_labels_M10, rf_predictions_M10)

##NB Models
    #NB Model
nb_models_M5 =[]
for train_index_M5, test_index_M5 in skf.split(M5, L):
    train_M5, test_M5 = M5[train_index_M5], M5[test_index_M5]
    train_labels_M5, test_labels_M5 = L[train_index_M5], L[test_index_M5]
    RSEED = 50
    nb_clf_M5 = GaussianNB()
    nb_clf_M5.fit(train_M5, train_labels_M5)  
    nb_predictions_M5 = nb_clf_M5.predict(test_M5)
    nb_accuracy_M5 = mt.accuracy_score(test_labels_M5,nb_predictions_M5)
    nb_models_M5.append((nb_accuracy_M5))
    nb_conf_M5 = mt.confusion_matrix(test_labels_M5,nb_predictions_M5)
    #print(f'NB Accuracy: {nb_accuracy}')    
    #print(mt.classification_report(test_labels,nb_predictions_M5))
    #Plot SVM AUC Curve
    nb_fpr_M5, nb_tpr_M5, thresholds_M5 = mt.roc_curve(test_labels_M5, nb_predictions_M5)
    nb_roc_auc_M5 = mt.roc_auc_score(test_labels_M5, nb_predictions_M5)

#NB Model
nb_models_M6 =[]
for train_index_M6, test_index_M6 in skf.split(M6, L):
    train_M6, test_M6 = M6[train_index_M6], M6[test_index_M6]
    train_labels_M6, test_labels_M6 = L[train_index_M6], L[test_index_M6]
    RSEED = 50
    nb_clf_M6 = GaussianNB()
    nb_clf_M6.fit(train_M6, train_labels_M6)  
    nb_predictions_M6 = nb_clf_M6.predict(test_M6)
    nb_accuracy_M6 = mt.accuracy_score(test_labels_M6,nb_predictions_M6)
    nb_models_M6.append((nb_accuracy_M6))
    nb_conf_M6 = mt.confusion_matrix(test_labels_M6,nb_predictions_M6)
    #print(f'NB Accuracy: {nb_accuracy}')    
    #print(mt.classification_report(test_labels,nb_predictions_M6))
    #Plot SVM AUC Curve
    nb_fpr_M6, nb_tpr_M6, thresholds_M6 = mt.roc_curve(test_labels_M6, nb_predictions_M6)
    nb_roc_auc_M6 = mt.roc_auc_score(test_labels_M6, nb_predictions_M6)

#NB Model
nb_models_M7 =[]
for train_index_M7, test_index_M7 in skf.split(M7, L):
    train_M7, test_M7 = M7[train_index_M7], M7[test_index_M7]
    train_labels_M7, test_labels_M7 = L[train_index_M7], L[test_index_M7]
    RSEED = 50
    nb_clf_M7 = GaussianNB()
    nb_clf_M7.fit(train_M7, train_labels_M7)  
    nb_predictions_M7 = nb_clf_M7.predict(test_M7)
    nb_accuracy_M7 = mt.accuracy_score(test_labels_M7,nb_predictions_M7)
    nb_models_M7.append((nb_accuracy_M7))
    nb_conf_M7 = mt.confusion_matrix(test_labels_M7,nb_predictions_M7)
    #print(f'NB Accuracy: {nb_accuracy}')    
    #print(mt.classification_report(test_labels,nb_predictions_M7))
    #Plot SVM AUC Curve
    nb_fpr_M7, nb_tpr_M7, thresholds_M7 = mt.roc_curve(test_labels_M7, nb_predictions_M7)
    nb_roc_auc_M7 = mt.roc_auc_score(test_labels_M7, nb_predictions_M7)

#NB Model
nb_models_M8 =[]
for train_index_M8, test_index_M8 in skf.split(M8, L):
    train_M8, test_M8 = M8[train_index_M8], M8[test_index_M8]
    train_labels_M8, test_labels_M8 = L[train_index_M8], L[test_index_M8]
    RSEED = 50
    nb_clf_M8 = GaussianNB()
    nb_clf_M8.fit(train_M8, train_labels_M8)  
    nb_predictions_M8 = nb_clf_M8.predict(test_M8)
    nb_accuracy_M8 = mt.accuracy_score(test_labels_M8,nb_predictions_M8)
    nb_models_M8.append((nb_accuracy_M8))
    nb_conf_M8 = mt.confusion_matrix(test_labels_M8,nb_predictions_M8)
    #print(f'NB Accuracy: {nb_accuracy}')    
    #print(mt.classification_report(test_labels,nb_predictions_M8))
    #Plot SVM AUC Curve
    nb_fpr_M8, nb_tpr_M8, thresholds_M8 = mt.roc_curve(test_labels_M8, nb_predictions_M8)
    nb_roc_auc_M8 = mt.roc_auc_score(test_labels_M8, nb_predictions_M8)

#NB Model
nb_models_M9 =[]
for train_index_M9, test_index_M9 in skf.split(M9, L):
    train_M9, test_M9 = M9[train_index_M9], M9[test_index_M9]
    train_labels_M9, test_labels_M9 = L[train_index_M9], L[test_index_M9]
    RSEED = 50
    nb_clf_M9 = GaussianNB()
    nb_clf_M9.fit(train_M9, train_labels_M9)  
    nb_predictions_M9 = nb_clf_M9.predict(test_M9)
    nb_accuracy_M9 = mt.accuracy_score(test_labels_M9,nb_predictions_M9)
    nb_models_M9.append((nb_accuracy_M9))
    nb_conf_M9 = mt.confusion_matrix(test_labels_M9,nb_predictions_M9)
    #print(f'NB Accuracy: {nb_accuracy}')    
    #print(mt.classification_report(test_labels,nb_predictions_M9))
    #Plot SVM AUC Curve
    nb_fpr_M9, nb_tpr_M9, thresholds_M9 = mt.roc_curve(test_labels_M9, nb_predictions_M9)
    nb_roc_auc_M9 = mt.roc_auc_score(test_labels_M9, nb_predictions_M9)

#NB Model
nb_models_M10 =[]
for train_index_M10, test_index_M10 in skf.split(M10, L):
    train_M10, test_M10 = M10[train_index_M10], M10[test_index_M10]
    train_labels_M10, test_labels_M10 = L[train_index_M10], L[test_index_M10]
    RSEED = 50
    nb_clf_M10 = GaussianNB()
    nb_clf_M10.fit(train_M10, train_labels_M10)  
    nb_predictions_M10 = nb_clf_M10.predict(test_M10)
    nb_accuracy_M10 = mt.accuracy_score(test_labels_M10,nb_predictions_M10)
    nb_models_M10.append((nb_accuracy_M10))
    nb_conf_M10 = mt.confusion_matrix(test_labels_M10,nb_predictions_M10)
    #print(f'NB Accuracy: {nb_accuracy}')    
    print(mt.classification_report(test_labels_M10,nb_predictions_M10))
    #Plot SVM AUC Curve
    nb_fpr_M10, nb_tpr_M10, thresholds_M10 = mt.roc_curve(test_labels_M10, nb_predictions_M10)
    nb_roc_auc_M10 = mt.roc_auc_score(test_labels_M10, nb_predictions_M10)



##Accuracy Plot
rf_results = (rf_accuracy_M5, rf_accuracy_M6, rf_accuracy_M7, rf_accuracy_M8, rf_accuracy_M9, rf_accuracy_M10)
nb_results = (nb_accuracy_M5, nb_accuracy_M6, nb_accuracy_M7, nb_accuracy_M8, nb_accuracy_M9, nb_accuracy_M10)
svm_results = (svm_accuracy_M5, svm_accuracy_M6, svm_accuracy_M7, svm_accuracy_M8, svm_accuracy_M9, svm_accuracy_M10)
kfolds = list(range(5,11))

plt.title('Comparison of Accuracy for k-mer of k length')
#plt.xlabel = 'K-Mer Length'
#plt.ylabel = 'Accuracy'
plt.xlabel("K-Mer Length")
plt.ylabel("Accuracy")
plt.plot(kfolds, rf_results, label = 'RF')
plt.plot(kfolds, nb_results, label = 'NB')
plt.plot(kfolds, svm_results, label = 'SVM')
plt.xlim([5, 10])
plt.ylim([0.0, 1.0])
plt.legend(loc="lower right")
plt.show()

#https://matplotlib.org/1.5.0/users/pyplot_tutorial.html
kmer_size = list(range(1,11))
kmer_length = [4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
kmer_empirical = [4, 16, 64, 256, 1024, 4096, 16384, 65535, 261448, 979112]
#plt.yscale('log')
plt.plot(kmer_size, kmer_length, label = 'Theoretical', color='g')
plt.plot(kmer_size, kmer_empirical, label = 'Empirical', color='r')
plt.title('Number of k-mers by K Length')
plt.xlabel('Length of K')
plt.ylabel('Number of K-Mers')
plt.legend(loc="lower right")
plt.show()

##Build Time Plot
model_build_time = [10.36, 11, 13.33, 19.5, 22.06, 377]
kmer_size_time = list(range(5,11))
plt.xlabel('Length of K')
plt.ylabel('Time to Build Model (minutes)')
plt.plot(kmer_size_time, model_build_time, color='g')
plt.show()

##ROC - AUC Plots
plt.figure()
plt.plot(rf_fpr_M10, rf_tpr_M10, label='k=10, (auc = %0.2f)' % rf_roc_auc_M10)
plt.plot(rf_fpr_M9, rf_tpr_M9, label='k=9, (auc = %0.2f)' % rf_roc_auc_M9)
plt.plot(rf_fpr_M8, rf_tpr_M8, label='k=8, (auc = %0.2f)' % rf_roc_auc_M8)
plt.plot(rf_fpr_M7, rf_tpr_M7, label='k=7, (auc = %0.2f)' % rf_roc_auc_M7)
plt.plot(rf_fpr_M6, rf_tpr_M6, label='k=6, (auc = %0.2f)' % rf_roc_auc_M6)
plt.plot(rf_fpr_M5, rf_tpr_M5, label='k=5, (auc = %0.2f)' % rf_roc_auc_M5)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(nb_fpr_M10, nb_tpr_M10, label='k=10, (auc = %0.2f)' % nb_roc_auc_M10)
plt.plot(nb_fpr_M9, nb_tpr_M9, label='k=9, (auc = %0.2f)' % nb_roc_auc_M9)
plt.plot(nb_fpr_M8, nb_tpr_M8, label='k=8, (auc = %0.2f)' % nb_roc_auc_M8)
plt.plot(nb_fpr_M7, nb_tpr_M7, label='k=7, (auc = %0.2f)' % nb_roc_auc_M7)
plt.plot(nb_fpr_M6, nb_tpr_M6, label='k=6, (auc = %0.2f)' % nb_roc_auc_M6)
plt.plot(nb_fpr_M5, nb_tpr_M5, label='k=5, (auc = %0.2f)' % nb_roc_auc_M5)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes ROC Curve')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(svm_fpr_M10, svm_tpr_M10, label='k=10, (auc = %0.2f)' % svm_roc_auc_M10)
plt.plot(svm_fpr_M9, svm_tpr_M9, label='k=9, (auc = %0.2f)' % svm_roc_auc_M9)
plt.plot(svm_fpr_M8, svm_tpr_M8, label='k=8, (auc = %0.2f)' % svm_roc_auc_M8)
plt.plot(svm_fpr_M7, svm_tpr_M7, label='k=7, (auc = %0.2f)' % svm_roc_auc_M7)
plt.plot(svm_fpr_M6, svm_tpr_M6, label='k=6, (auc = %0.2f)' % svm_roc_auc_M6)
plt.plot(svm_fpr_M5, svm_tpr_M5, label='k=5, (auc = %0.2f)' % svm_roc_auc_M5)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Support Vector Machine ROC Curve')
plt.legend(loc="lower right")
plt.show()


#Classification Report
print(mt.classification_report(rf_test_labels_M10,rf_predictions_M10))

#Top 25 Features for RF K=10
importances = rf_clf_M10.feature_importances_
indices = np.argsort(importances)[::-1]
top_k = 25
new_indices = indices[:top_k]

#for f in range(top_k):
#        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#        plt.barh(indices[f], importances[indices[f]], color='b', align='center')
#        plt.figure(1)
#        plt.title('Top 25 Features')
#        plt.barh(range(len(new_indices)), importances[new_indices], color='b', align='center')
#        plt.xlabel('Relative Importance')
#        #Feature Importance Plot
#        #https://matplotlib.org/3.1.0/tutorials/text/text_intro.html
#        fig = plt.figure()
#        #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
#        ax = fig.add_subplot(111)
#        fig.subplots_adjust(top=.85)
#        ax.set_title('Feature Importance by Feature')
#        ax.set_xlabel('Feature')
#        ax.set_ylabel('Feature Importance Score')
#        ax.plot(sorted(rf_clf_M10.feature_importances_))
#        plt.show()

#Identify the most important feature
rf_max_feature = np.where(rf_clf_M10.feature_importances_==max(rf_clf_M10.feature_importances_))[0][0]
##Plot Random Forest Feature with Maximum value
plt.plot(M10[:,rf_max_feature])
plt.xlabel('Isolate #')
plt.ylabel('Frequency')
plt.title('Frequency of k-mer within Max Feature')
plt.show()
#Draw the average line to illustrate separation

plt.hist(M10[:,rf_max_feature],40)
plt.xlabel('Isolate #')
plt.ylabel('Frequency')
plt.show()

plt.scatter(range(len(all_isolates)),M10[:,rf_max_feature])
plt.xlabel('Isolate #')
plt.ylabel('Frequency')
plt.title('Frequency of k-mer within Feature '+ str(rf_max_feature))
plt.show()

#To Do:
#Pickle matrix and labels and rerun for k 5-10+
#Gauassian Mixture Model (GMM) and Kmeans

##KMM
means = KMeans(n_clusters=2)
feature_kmeans = M10[:,rf_max_feature].reshape(-1, 1)
#means.fit(M[:,2015])
means.fit(feature_kmeans)
means.labels_
labels_kmeans = means.predict(feature_kmeans)
plt.scatter(range(392),feature_kmeans, c=labels_kmeans)
plt.show()
#plt.scatter(train[:, 0], train[:, 1], c=labels_kmeans, s=40);
##plt.scatter(train[M], c=labels_kmeans);
#plt.scatter(M[:,2015], c=labels_kmeans)

##GMM
feature_gmm = M10[:,rf_max_feature].reshape(-1, 1)
#Calculate accuracy on purple appears to be highly accurate
#Light green - 50/50
#Teal = Mostly negative - highly accurate
#Yellow 

gmm = GaussianMixture(n_components=2).fit(feature_gmm)
gmm_labels = gmm.predict(feature_gmm)
plt.scatter(range(392),feature_gmm, c=gmm_labels, s=40);
plt.xlabel('Isolate #')
plt.ylabel('Frequency')
plt.title('Frequency within Each Isolate of Feature '+ str(rf_max_feature) + ' for k=10')
plt.vlines(178.5, 0, 8, colors='r')
#plt.scatter(range(len(all_isolates)),M[:,2015], c=gmm_labels)
plt.show()

#Through 178 is susceptible and rest are resistant
#Classification of max feature with 0 or 1 occurence
np.where(M10[:,rf_max_feature]==1) or np.where(M10[:,rf_max_feature]==0)

#Classification of max feature with 6 or 7 occurences
np.where(M10[:,rf_max_feature]>=6)

##GMM & Kmeans new code
#https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f
#Might take one feature and do a k-means clustering to point out the clusters.  1287.

svm_cm_M10 = (mt.classification_report(svm_test_labels_M10,svm_predictions_M10))
rf_cm_M10 = (mt.classification_report(rf_test_labels_M10,rf_predictions_M10))
nb_cm_M10 = (mt.classification_report(test_labels_M10,nb_predictions_M10))

#https://heartbeat.fritz.ai/analyzing-machine-learning-models-with-yellowbrick-37795733f3ee
from yellowbrick.classifier import ClassificationReport
classes = ['Susceptible', 'Resistant']

visualizer = ClassificationReport(svm_clf_M10, classes = classes)
visualizer.fit(svm_train_M10, svm_train_labels_M10)  
visualizer.score(svm_test_M10, svm_test_labels_M10)  
g = visualizer.poof()

visualizer = ClassificationReport(rf_clf_M10, classes = classes)
visualizer.fit(rf_train_M10, rf_train_labels_M10)  
visualizer.score(rf_test_M10, rf_test_labels_M10)  
g = visualizer.poof()

visualizer = ClassificationReport(nb_clf_M10, classes = classes)
visualizer.fit(train_M10, train_labels_M10)  
visualizer.score(test_M10, test_labels_M10)  
g = visualizer.poof()

#plt.scatter(range(392), M8[:,65530])
#plt.scatter(range(392), M5[:,1023])
#plt.show()

#Looking at a couple of plots that might explain SVM accuracy increase
#https://www.dummies.com/programming/big-data/data-science/how-to-visualize-the-classifier-in-an-svm-supervised-learning-model/
#Plot PCA Plot for k = 5
pca_M5 = PCA(n_components=2).fit(svm_train_M5)
pca_M5_2d = pca_M5.transform(svm_train_M5)
import pylab as pl
for i in range(0, pca_M5_2d.shape[0]):
    if svm_train_labels_M5[i] == 0:
        c1 = pl.scatter(pca_M5_2d[i,0],pca_M5_2d[i,1],c='r',    marker='+')
    elif svm_train_labels_M5[i] == 1:
        c2 = pl.scatter(pca_M5_2d[i,0],pca_M5_2d[i,1],c='g',    marker='o')
pl.legend([c1, c2], ['Susceptible', 'Resistant'])
pl.title('Susceptible and Resistant for k = 5')
pl.show()

import warnings
warnings.simplefilter('ignore')
svmClassifier_M5 = svm.LinearSVC(random_state=111).fit(pca_M5_2d, svm_train_labels_M5)
M5_x_min, M5_x_max = pca_M5_2d[:, 0].min() - 1,   pca_M5_2d[:,0].max() + 1
M5_y_min, M5_y_max = pca_M5_2d[:, 1].min() - 1,   pca_M5_2d[:, 1].max() + 1
xx_M5, yy_M5 = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(M5_y_min, M5_y_max, .01))
Z = svmClassifier_M5.predict(np.c_[xx_M5.ravel(),  yy_M5.ravel()])
Z = Z.reshape(xx_M5.shape)
pl.contour(xx_M5, yy_M5, Z)
pl.title('Support Vector Machine Decision Surface')
pl.axis('off')
pl.show()






pca_M10 = PCA(n_components=2).fit(svm_train_M10)
pca_M10_2d = pca_M10.transform(svm_train_M10)
import pylab as pl
for i in range(0, pca_M10_2d.shape[0]):
    if svm_train_labels_M10[i] == 0:
        c1 = pl.scatter(pca_M10_2d[i,0],pca_M10_2d[i,1],c='r',    marker='+')
    elif svm_train_labels_M10[i] == 1:
        c2 = pl.scatter(pca_M10_2d[i,0],pca_M10_2d[i,1],c='g',    marker='o')
pl.legend([c1, c2], ['Susceptible', 'Resistant'])
pl.title('Susceptible and Resistant for k = 10')
pl.show()

import warnings
warnings.simplefilter('ignore')
svmClassifier_M10 = svm.LinearSVC(random_state=111).fit(pca_M10_2d, svm_train_labels_M10)
M10_x_min, M10_x_max = pca_M10_2d[:, 0].min() - 1,   pca_M10_2d[:,0].max() + 1
M10_y_min, M10_y_max = pca_M10_2d[:, 1].min() - 1,   pca_M10_2d[:, 1].max() + 1
#xx_M10, yy_M10 = np.meshgrid(np.arange(M10_x_min, M10_x_max, .01),   np.arange(M10_y_min, M10_y_max, .01))
xx_M10, yy_M10 = np.meshgrid(np.arange(M10_x_min, M10_x_max, .5),   np.arange(M10_y_min, M10_y_max, .5))

Z = svmClassifier_M10.predict(np.c_[xx_M10.ravel(),  yy_M10.ravel()])
Z = Z.reshape(xx_M10.shape)
pl.contour(xx_M10, yy_M10, Z)
pl.title('Support Vector Machine Decision Surface')
pl.axis('off')
pl.show()





#https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
plt.scatter(pca_M10_2d[:, 0], pca_M10_2d[:, 1], c=pca_M10_2d[:, 1], s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
#Z = clf.decision_function(xy).reshape(XX.shape)
Z = svmClassifier_M10.predict(np.c_[xx_M10.ravel(),  yy_M10.ravel()])
Z = Z.reshape(xx_M10.shape)
# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(svmClassifier_M10.support_vectors_[:, 0], svmClassifier_M10.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()




svm_models_M10 = []
#SVM Model
for svm_train_index_M10, svm_test_index_M10 in skf.split(M10, L):
    svm_train_M10, svm_test_M10 = M10[svm_train_index_M10], M10[svm_test_index_M10]
    svm_train_labels_M10, svm_test_labels_M10 = L[svm_train_index_M10], L[svm_test_index_M10]
    RSEED = 50
    #svm_clf_M10 = svm.SVC(gamma='auto', C=100, kernel='linear')
    svm_clf_M10 = svm.SVC(gamma='auto', random_state = RSEED)
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

svm_roc_auc_M10
svm_accuracy_M10
