import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

path = "orb.csv"
columns = ["pred_d6", "pred_d20", "pred_d4", "pred_d12", "pred_d8", "pred_d10", "d6", "d20", "d4", "d12", "d8", "d10"]
pred_columns = ["pred_d6", "pred_d20", "pred_d4", "pred_d12", "pred_d8", "pred_d10"]
true_columns = ["d6", "d20", "d4", "d12", "d8", "d10"]
df = pd.read_csv(path, names = columns)
print(df.shape)
#largest_pred = df[pred_columns].idxmax(axis=1)
largest_pred = df[pred_columns].max(axis=1)
#df["high_score"] = largest_pred
#df[pred_columns] = df[pred_columns].apply(, axis=1)
df[pred_columns] = df[pred_columns].where(df[pred_columns].eq(largest_pred, axis = 0),int(0))
df[pred_columns] = df[pred_columns].where(~df[pred_columns].eq(largest_pred, axis = 0), int(1))
df = df.astype("int32")

pred_names = df[pred_columns].idxmax(axis=1)

#print(pred_names.str[5:])
y_pred =  pred_names.str[5:]
y_true =  df[true_columns].idxmax(axis=1)
#print(y_pred)
#print(y_true)
matrix = confusion_matrix(y_true, y_pred)
#print(matrix)
#print(df.columns.tolist())
#conf_plot = ConfusionMatrixDisplay(matrix).plot()
#sns.heatmap(matrix, annot=True)
#plt.show()

figsize=(10,10)
cols_ordered = ['d4', 'd6', 'd8','d10', 'd12', 'd20']
#cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
cm = confusion_matrix(y_true, y_pred, labels=cols_ordered)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
            annot[i, j] = ''
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)
#cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
cm = pd.DataFrame(cm, index=cols_ordered, columns=cols_ordered)
#print(cm.columns.tolist())

#cm = cm[cols_ordered]
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=figsize)
sns_plot=sns.heatmap(cm, cmap= 'Blues', annot=annot, fmt='', ax=ax)
#sns_plot.figure.savefig("orb_confusion.png")
#fig.savefig("orb_confusion.png",dpi=100)

#plt.savefig("orb_confusion.png")
print("METRICS:")
print("Accuracy:")
#acc = accuracy_score(y_true, y_pred, normalize=False)
#print(acc)
#print("Normalize Accuracy:")
acc_norm = accuracy_score(y_true, y_pred)
print(acc_norm)
print("ROC:")
roc = roc_auc_score(df[true_columns], df[pred_columns])
print(roc)
plt.show()