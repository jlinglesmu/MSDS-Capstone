#Version with NB, SVM, RF for only 1 length

import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics as mt
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
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
fileObject = open(master_dict_file_Name,'wb') 
# this writes the object a to the
# file named 'testfile'
pickle.dump(master_dict,fileObject)   
# here we close the fileObject
fileObject.close()

# graph of number of kmers by size in k for susceptible and resistant
# graph illustrating largest variances in size?
# table of file counts for resistant/susceptible
# range of total contigs within resistant and susceptible

kmer_size = list(range(1,21))
    
kmer_length = []
for j in range(1, 21):
    k = make_kmer(j, contigs)
    kmer_length.append(len(k.keys()))
    print(kmer_length, ' ' , kmer_size)

#https://matplotlib.org/3.1.0/tutorials/text/text_intro.html
fig = plt.figure()
#fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=.85)
ax.set_title('Number of Possible Contigs by K-mer Length')
ax.set_xlabel('Number of Possible Contigs')
ax.set_ylabel('K-Mer Length')
ax.plot(kmer_length, kmer_size)
plt.show()


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
M_file_Name = "M" + plot_num_str
# open the file for writing
M_fileObject = open(M_file_Name,'wb') 
# this writes the object a to the
# file named 'testfile'
pickle.dump(M,M_fileObject)   


skf = StratifiedKFold(n_splits=5, shuffle=True)
train_results = []
test_results = []

L = np.array(L)
L = L.astype(int)


#Train Random Forest model
for train_index, test_index in skf.split(M, L):
    train, test = M[train_index], M[test_index]
    train_labels, test_labels = L[train_index], L[test_index]
    RSEED = 50
    rf_clf = RandomForestClassifier(n_estimators=10, 
                               random_state=RSEED, 
                               max_features = 'sqrt',
                               n_jobs=-1, 
                               class_weight="balanced",
                               verbose = 1)
    # Fit on training data
    rf_clf.fit(train, train_labels)
    # Class predictions
    rf_predictions = rf_clf.predict(test)
    # Probabilities for classes
    rf_accuracy = round(mt.accuracy_score(test_labels, rf_predictions),2)
    rf_conf = mt.confusion_matrix(test_labels,rf_predictions)
    print(f'RF Model Accuracy: {rf_accuracy}')
    print(mt.classification_report(test_labels,rf_predictions))
    #Top 25 Features
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_k = 25
    new_indices = indices[:top_k]
    # Print the feature ranking
    #canvas = canvas.Canvas('rf_model.pdf', pagesize=letter)
    #width, height = letter
    print("Feature ranking:")
    for f in range(top_k):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #canvas.save()
    #Plot RF AUC Curve
    rf_fpr, rf_tpr, thresholds = mt.roc_curve(test_labels, rf_predictions)
    rf_roc_auc = mt.roc_auc_score(test_labels, rf_predictions)
    filename = 'rf_roc_auc_curve' + plot_num_str + '.png' # concatenate the filename prefix and the plot_num_str
    plt.figure()
    plt.plot(rf_fpr, rf_tpr, label='RF ROC curve (area = %0.2f)' % rf_roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - %s' % filename.split('/')[-1])
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(filename, bbox_inches = 'tight')


# Plot the feature importances of the forest
importances = rf_clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
for f in range(top_k):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        plt.barh(indices[f], importances[indices[f]], color='b', align='center')
        plt.figure(1)
        plt.title('Top 25 Features')
        plt.barh(range(len(new_indices)), importances[new_indices], color='b', align='center')
        plt.xlabel('Relative Importance')
        #Feature Importance Plot
        #https://matplotlib.org/3.1.0/tutorials/text/text_intro.html
        fig = plt.figure()
        #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=.85)
        ax.set_title('Feature Importance by Feature')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Feature Importance Score')
        ax.plot(sorted(rf_clf.feature_importances_))
        plt.show()



#SVM Model
for train_index, test_index in skf.split(M, L):
    train, test = M[train_index], M[test_index]
    train_labels, test_labels = L[train_index], L[test_index]
    RSEED = 50
    svm_clf = svm.SVC(gamma='auto', C=100, kernel='linear')
    svm_clf.fit(train, train_labels)  
    svm_predictions = svm_clf.predict(test)
    svm_accuracy = mt.accuracy_score(test_labels,svm_predictions)
    svm_conf = mt.confusion_matrix(test_labels,svm_predictions)
    print(f'SVM Accuracy: {svm_accuracy}')
    print(mt.classification_report(test_labels,svm_predictions))
    #Plot SVM AUC Curve
    svm_fpr, svm_tpr, thresholds = mt.roc_curve(test_labels, svm_predictions)
    svm_roc_auc = mt.roc_auc_score(test_labels, svm_predictions)
    filename = 'rf_roc_auc_curve' + plot_num_str + '.png' # concatenate the filename prefix and the plot_num_str
    plt.figure()
    plt.plot(svm_fpr, svm_tpr, label='SVM ROC curve (area = %0.2f)' % svm_roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - %s' % filename.split('/')[-1])
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(filename, bbox_inches = 'tight')

#NB Model
for train_index, test_index in skf.split(M, L):
    train, test = M[train_index], M[test_index]
    train_labels, test_labels = L[train_index], L[test_index]
    RSEED = 50
    nb_clf= GaussianNB()
    nb_clf.fit(train, train_labels)  
    nb_predictions = nb_clf.predict(test)
    nb_accuracy = mt.accuracy_score(test_labels,nb_predictions)
    nb_conf = mt.confusion_matrix(test_labels,nb_predictions)
    print(f'NB Accuracy: {nb_accuracy}')    
    print(mt.classification_report(test_labels,nb_predictions))
    #Plot SVM AUC Curve
    nb_fpr, nb_tpr, thresholds = mt.roc_curve(test_labels, nb_predictions)
    nb_roc_auc = mt.roc_auc_score(test_labels, nb_predictions)
    filename = 'nb_roc_auc_curve' + plot_num_str + '.png' # concatenate the filename prefix and the plot_num_str
    plt.figure()
    plt.plot(nb_fpr, nb_tpr, label='NB ROC curve (area = %0.2f)' % nb_roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - %s' % filename.split('/')[-1])
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(filename, bbox_inches = 'tight')

end = time.time()
print(end-start)
#5 = 622.2274
#6 = 660.334
#7 =  & 799.4
#8 = 1169.13
#9 = 1323.937
#10 = 22608.248


#Identify the most important feature
rf_max_feature = np.where(rf_clf.feature_importances_==max(rf_clf.feature_importances_))[0][0]
##Plot Random Forest Feature with Maximum value
plt.plot(M[:,rf_max_feature])
plt.xlabel('Isolate #')
plt.ylabel('Frequency')
plt.title('Frequency of Contig within Isolate '+ str(rf_max_feature))
plt.show()
##Repeated gene--duplication of DNA
#Draw the average line to illustrate separation

plt.hist(M[:,rf_max_feature],40)
plt.xlabel('Isolate #')
plt.ylabel('Frequency')
plt.show()

plt.scatter(range(len(all_isolates)),M[:,rf_max_feature])
plt.xlabel('Isolate #')
plt.ylabel('Frequency')
plt.title('Frequency of Contig within Isolate '+ str(rf_max_feature))
plt.show()

#To Do:
#Pickle matrix and labels and rerun for k 5-10+
#Gauassian Mixture Model (GMM) and Kmeans

##KMM
means = KMeans(n_clusters=2)
feature_kmeans = M[:,rf_max_feature].reshape(-1, 1)
#means.fit(M[:,2015])
means.fit(feature_kmeans)
means.labels_
labels_kmeans = means.predict(feature_kmeans)
plt.scatter(range(392),feature_kmeans, c=labels_kmeans)
plt.show()
#plt.scatter(train[:, 0], train[:, 1], c=labels_kmeans, s=40);
##plt.scatter(train[M], c=labels_kmeans);
#plt.scatter(M[:,2015], c=labels_kmeans)
#
##GMM
feature_gmm = M[:,rf_max_feature].reshape(-1, 1)
#Calculate accuracy on purple appears to be highly accurate
#Light green - 50/50
#Teal = Mostly negative - highly accurate
#Yellow 
gmm = GaussianMixture(n_components=5).fit(feature_gmm)
gmm_labels = gmm.predict(feature_gmm)
plt.scatter(range(392),feature_gmm, c=gmm_labels, s=40);
#plt.scatter(range(len(all_isolates)),M[:,2015], c=gmm_labels)
plt.show()


##GMM & Kmeans new code
#https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f
#Might take one feature and do a k-means clustering to point out the clusters.  1287.



#AUC Code
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