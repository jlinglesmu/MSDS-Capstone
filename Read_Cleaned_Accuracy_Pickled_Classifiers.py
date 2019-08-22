##Automatic Model Selection for 5-10 for NB, SVM, and RF

import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mt
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from sklearn.cluster import KMeans
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

kmer_size = list(range(1,21))

zz= make_kmer(k_len, contigs)
kmer_index = list(zz.keys())
number_kmers = len(zz.keys())

M = np.zeros([len(all_isolates), len(master_dict)], dtype=np.int64)

for column, kmer_string in enumerate(master_dict.keys()):
    for row in master_dict[kmer_string].keys():
        #print(master_dict[kmer_string].keys())
        M[row, column] = master_dict[kmer_string][row]

#convert plot number to string
plot_num_str = str(k_len) 
#set file name to master_dict with the length of k
#M_file_Name = "M" + plot_num_str
# open the file for writing
#M_fileObject = open(M_file_Name,'wb') 
# this writes the object a to the
# file named 'testfile'
#pickle.dump(M,M_fileObject)   

L = np.array(L)
L = L.astype(int)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

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
M10 = M

#M5 Models
for train_index, test_index in skf.split(M5, L):
    train_M5, test_M5 = M5[train_index], M5[test_index]
    train_labels, test_labels = L[train_index], L[test_index]
models_M5 = []
models_M5.append(('LR', LogisticRegression()))
models_M5.append(('RF', RandomForestClassifier()))
#models_M5.append(('KNN', KNeighborsClassifier()))
models_M5.append(('NB', GaussianNB()))
#models_M5.append(('LDA', LinearDiscriminantAnalysis()))
models_M5.append(('SVM', svm.SVC()))
# evaluate each model in turn
results_M5 = []
names_M5 = []
seed = 7
scoring = 'accuracy'
for name_M5, model in models_M5:
	kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
	cv_results_M5 = model_selection.cross_val_score(model, train_M5, train_labels, cv=kfold, scoring=scoring)
	results_M5.append(cv_results_M5)
	names_M5.append(name_M5)
	msg = "%s: %f (%f)" % (name_M5, cv_results_M5.mean(), cv_results_M5.std())
	print(msg)

#M6 Models
for train_index, test_index in skf.split(M6, L):
    train_M6, test_M6 = M6[train_index], M6[test_index]
    train_labels, test_labels = L[train_index], L[test_index]
models_M6 = []
models_M6.append(('LR', LogisticRegression()))
models_M6.append(('RF', RandomForestClassifier()))
#models_M6.append(('KNN', KNeighborsClassifier()))
models_M6.append(('NB', GaussianNB()))
#models_M6.append(('LDA', LinearDiscriminantAnalysis()))
models_M6.append(('SVM', svm.SVC()))
# evaluate each model in turn
results_M6 = []
names_M6 = []
seed = 7
scoring = 'accuracy'
for name_M6, model in models_M6:
	kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
	cv_results_M6 = model_selection.cross_val_score(model, train_M6, train_labels, cv=kfold, scoring=scoring)
	results_M6.append(cv_results_M6)
	names_M6.append(name_M6)
	msg = "%s: %f (%f)" % (name_M6, cv_results_M6.mean(), cv_results_M6.std())
	print(msg)

#M7 Models
for train_index, test_index in skf.split(M7, L):
    train_M7, test_M7 = M7[train_index], M7[test_index]
    train_labels, test_labels = L[train_index], L[test_index]
models_M7 = []
models_M7.append(('LR', LogisticRegression()))
models_M7.append(('RF', RandomForestClassifier()))
#models_M7.append(('KNN', KNeighborsClassifier()))
models_M7.append(('NB', GaussianNB()))
#models_M7.append(('LDA', LinearDiscriminantAnalysis()))
models_M7.append(('SVM', svm.SVC()))
# evaluate each model in turn
results_M7 = []
names_M7 = []
seed = 7
scoring = 'accuracy'
for name_M7, model in models_M7:
	kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
	cv_results_M7 = model_selection.cross_val_score(model, train_M7, train_labels, cv=kfold, scoring=scoring)
	results_M7.append(cv_results_M7)
	names_M7.append(name_M7)
	msg = "%s: %f (%f)" % (name_M7, cv_results_M7.mean(), cv_results_M7.std())
	print(msg)

#M8 Models
for train_index, test_index in skf.split(M8, L):
    train_M8, test_M8 = M8[train_index], M8[test_index]
    train_labels, test_labels = L[train_index], L[test_index]
models_M8 = []
models_M8.append(('LR', LogisticRegression()))
models_M8.append(('RF', RandomForestClassifier()))
#models_M8.append(('KNN', KNeighborsClassifier()))
models_M8.append(('NB', GaussianNB()))
#models_M8.append(('LDA', LinearDiscriminantAnalysis()))
models_M8.append(('SVM', svm.SVC()))
# evaluate each model in turn
results_M8 = []
names_M8 = []
seed = 7
scoring = 'accuracy'
for name_M8, model in models_M8:
	kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
	cv_results_M8 = model_selection.cross_val_score(model, train_M8, train_labels, cv=kfold, scoring=scoring)
	results_M8.append(cv_results_M8)
	names_M8.append(name_M8)
	msg = "%s: %f (%f)" % (name_M8, cv_results_M8.mean(), cv_results_M8.std())
	print(msg)

#M9 Models
for train_index, test_index in skf.split(M9, L):
    train_M9, test_M9 = M9[train_index], M9[test_index]
    train_labels, test_labels = L[train_index], L[test_index]
models_M9 = []
models_M9.append(('LR', LogisticRegression()))
models_M9.append(('RF', RandomForestClassifier()))
#models_M9.append(('KNN', KNeighborsClassifier()))
models_M9.append(('NB', GaussianNB()))
#models_M9.append(('LDA', LinearDiscriminantAnalysis()))
models_M9.append(('SVM', svm.SVC()))
# evaluate each model in turn
results_M9 = []
names_M9 = []
seed = 7
scoring = 'accuracy'
for name_M9, model in models_M9:
	kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
	cv_results_M9 = model_selection.cross_val_score(model, train_M9, train_labels, cv=kfold, scoring=scoring)
	results_M9.append(cv_results_M9)
	names_M9.append(name_M9)
	msg = "%s: %f (%f)" % (name_M9, cv_results_M9.mean(), cv_results_M9.std())
	print(msg)
    
#M10 Models
for train_index, test_index in skf.split(M10, L):
    train_M10, test_M10 = M10[train_index], M10[test_index]
    train_labels, test_labels = L[train_index], L[test_index]
models_M10 = []
models_M10.append(('LR', LogisticRegression()))
models_M10.append(('RF', RandomForestClassifier()))
#models_M10.append(('KNN', KNeighborsClassifier()))
models_M10.append(('NB', GaussianNB()))
#models_M10.append(('LDA', LinearDiscriminantAnalysis()))
models_M10.append(('SVM', svm.SVC()))
# evaluate each model in turn
results_M10 = []
names_M10 = []
seed = 7
scoring = 'accuracy'
for name_M10, model in models_M10:
	kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
	cv_results_M10 = model_selection.cross_val_score(model, train_M10, train_labels, cv=kfold, scoring=scoring)
	results_M10.append(cv_results_M10)
	names_M10.append(name_M10)
	msg = "%s: %f (%f)" % (name_M10, cv_results_M10.mean(), cv_results_M10.std())
	print(msg)

    
#https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison For k-mer of k length 10')
ax = fig.add_subplot(111)
plt.boxplot(results_M10)
ax.set_xticklabels(names_M10)
plt.xlabel('Accuracy')
plt.ylabel('Classifier')
plt.show()



lr_results = [results_M5[0].mean(), results_M6[0].mean(), results_M7[0].mean(), results_M8[0].mean(), results_M9[0].mean(), results_M10[0].mean()]
rf_results = [results_M5[1].mean(), results_M6[1].mean(), results_M7[1].mean(), results_M8[1].mean(), results_M9[1].mean(), results_M10[1].mean()]
#knn_results = [results_M5[2].mean(), results_M6[2].mean(), results_M7[2].mean(), results_M8[2].mean(), results_M9[2].mean(), results_M10[2].mean()]
nb_results = [results_M5[2].mean(), results_M6[2].mean(), results_M7[2].mean(), results_M8[2].mean(), results_M9[2].mean(), results_M10[2].mean()]
#lda_results = [results_M5[4].mean(), results_M6[4].mean(), results_M7[4].mean(), results_M8[4].mean(), results_M9[4].mean(), results_M10[4].mean()]
svm_results = [results_M5[3].mean(), results_M6[3].mean(), results_M7[3].mean(), results_M8[3].mean(), results_M9[3].mean(), results_M10[3].mean()]
#Line Plot
kfolds = list(range(5,11))

plt.title('Comparison of Accuracy for k-mer of k length 10')
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



#AUC Code
rf_predictions = rf_clf.predict(test)
# Probabilities for classes
rf_accuracy = round(mt.accuracy_score(test_labels, rf_predictions),2)
rf_conf = mt.confusion_matrix(test_labels,rf_predictions)
print(f'RF Model Accuracy: {rf_accuracy}')
print(mt.classification_report(test_labels,rf_predictions))
#Top 25 Features
#importances = rf_clf.feature_importances_
#indices = np.argsort(importances)[::-1]
#top_k = 25

fpr, tpr, thresholds = mt.roc_curve(test_labels, rf_predictions)
roc_auc = mt.roc_auc_score(test_labels, rf_predictions)
filename = 'roc_auc_curve' + plot_num_str + '.png' # concatenate the filename prefix and the plot_num_str
plt.figure()
plt.plot(fpr, tpr, label='Random Forest ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - %s' % filename.split('/')[-1])
plt.legend(loc="lower right")
plt.show()
plt.savefig(filename, bbox_inches = 'tight')

###Comparison Plot
plt.figure(figsize=(8,6))

#for M10 in models_M10:
#    M10['model'].probability = True
#    probas = M10['model'].fit(M10['roc_train'], M10['roc_train_class']).predict_proba(M10['roc_test'])
#    fpr, tpr, thresholds = mt.roc_curve(M10['roc_test_class'], probas[:, 1])
#    roc_auc  = mt.auc(fpr, tpr)
#    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (M9['label'], roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.show()

##Placeholder for Classifier Plot
plt.title('Accuracy by Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('k-mer size')
plt.ylabel('Accuracy Score')
plt.show()

#https://matplotlib.org/1.5.0/users/pyplot_tutorial.html
kmer_size = list(range(1,21))
kmer_length = [4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824, 4294967296, 17179869184, 68719476736, 274877906944, 1099511627776]
plt.yscale('log')
plt.plot(kmer_size, kmer_length, color='g')
plt.title('Number of Possible K-mers by K Length')
plt.xlabel('Length of K')
plt.ylabel('Log of Possible K-Mers')
plt.show()

model_build_time = [10.36, 11, 13.33, 19.5, 22.06, 377]
kmer_size_time = list(range(5,11))
plt.xlabel('Length of K')
plt.ylabel('Time to Build Model (minutes)')
plt.plot(kmer_size_time, model_build_time, color='g')
plt.show()

#5 = 622.2274
#6 = 660.334
#7 =  & 799.4
#8 = 1169.13
#9 = 1323.937
#10 = 22608.248