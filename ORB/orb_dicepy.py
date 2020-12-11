# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import cv2
import numpy as np
import os
import pandas as pd
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
import glob
import os
import gc
import pickle
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
#       print(os.path.join(dirname, filename))



train_files = glob.glob('dice-d4-d6-d8-d10-d12-d20/dice/train/**/*.JPG', recursive=True)
test_files = glob.glob('dice-d4-d6-d8-d10-d12-d20/dice/valid/**/*.JPG', recursive=True)



print(len(train_files))
print(len(test_files))



#def read_image(path):
    #label = os.path.basename(os.path.dirname(path))
    
    #image = cv2.imread(path)
    #image = cv2.resize(image, (DIMS[:2]))
    
    #return image #/ 255.0

#def randomize_dataset(X, labels):
    # randomize dataset
#    ids = list(range(len(X)))
#    np.random.shuffle(ids)
    
#    for (i, j) in enumerate(ids):
#        X[i], X[j] = X[j], X[i]
#        labels[i], labels[j] = labels[j], labels[i]
        
#    return (X, labels)
        
def read_dataset(files):    
    
    #X = np.zeros((len(train_files), *DIMS), dtype=np.float32)
    n = len(files)
    X = [None] * n 
    i = 0  
    for i in range(n):
        image = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        X[i] = image
        i = i + 1 
    #X = np.array(X,dtype=object) #runs out of RAM
    X = np.array(X)
    labels = [os.path.basename(os.path.dirname(path)) for path in files]
    
    #X, labels = randomize_dataset(X, labels)
    
    classes  = set([label for label in labels])
    id2class = dict([(i, c) for (i, c) in enumerate(classes)])
    class2id = dict([(c, i) for (i, c) in enumerate(classes)])
    
    Y = np.zeros((n, len(classes)), dtype=np.bool)
    
    del i
    for (i, clazz) in enumerate(labels):
        Y[i, class2id[clazz]] = 1
     
    return (X, Y, classes, id2class, class2id)



X_train, Y_train, classes, id2class, class2id = read_dataset(train_files)


X_test, Y_test, _, _, _ = read_dataset(test_files)



del read_dataset
gc.collect()


print("X_train.shape")
print(X_train.shape)
print("Y_train.shape")
print(Y_train.shape)
print("X_test.shape")
print(X_test.shape)
print("Y_test.shape")
print(Y_test.shape)
print("classes")
print(classes)
print("id2class")
print(id2class)
print("class2id")
print(class2id)







orb = cv2.ORB_create()
n = X_train.shape[0]
features = []




num_labels = Y_train.shape[1]
k = num_labels * 10
kmeans_path = "kmeans.pkl"
print("Starting Kmeans")
if os.path.isfile(kmeans_path):
    kmeans = pickle.load(open(kmeans_path, "rb"))
else:
    for i in range(n):
        img = X_train[i]
        kp, des = orb.detectAndCompute(img, None)
        for d in des:
            features.append(d)
    kmeans = MiniBatchKMeans(n_clusters=k, verbose=1, n_init = 3, max_iter = 500).fit(features)
    pickle.dump(kmeans, open(kmeans_path, "wb"))

print("Finishing Kmeans")
gc.collect()

kmeans.verbose = False
histo_list = [None] * n 

"""
mlp_path = "mlp.pkl"
print("Starting MLP")
if os.path.isfile(mlp_path):
    mlp = pickle.load(open(mlp_path, "rb"))
else:
    for i in range(n):
        img = X_train[i]
        kp, des = orb.detectAndCompute(img, None)
        histo = np.zeros(k)
        nkp = np.size(kp)
        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1/nkp
        histo_list[i] = histo
    X = np.array(histo_list)
    mlp = MLPClassifier(verbose=True, max_iter=600000)
    mlp.fit(X, Y_train)
    pickle.dump(mlp, open(mlp_path, "wb"))
print("Finished MLP")
"""
for i in range(n):
    img = X_train[i]
    kp, des = orb.detectAndCompute(img, None)
    histo = np.zeros(k)
    nkp = np.size(kp)
    for d in des:
        idx = kmeans.predict([d])
        histo[idx] += 1/nkp
    histo_list[i] = histo
X = np.array(histo_list)
mlp = MLPClassifier(verbose=True, max_iter=600000)
mlp.fit(X, Y_train)
print("Finished MLP")
gc.collect()


result_file = open("orb.csv", "w", newline='')
result_file_obj = csv.writer(result_file)
#result_file_obj.writerow(np.append("id", classes, "True class"))




n_test = X_test.shape[0]
for i in range(n_test):
    img = X_test[i]
    kp, des = orb.detectAndCompute(img, None)
    x = np.zeros(k)
    nkp = np.size(kp)
    for d in des:
        idx = kmeans.predict([d])
        x[idx] += 1/nkp

    res = mlp.predict_proba([x])
    row = []
    row.append(i)

    for e in res[0]:
        row.append(e)
    for e_t in Y_test[i]:
        row.append(int(e_t))
    result_file_obj.writerow(row)
result_file.close()


