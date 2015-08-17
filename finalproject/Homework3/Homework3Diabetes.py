
# coding: utf-8

# ###This is a csv file with 768 observations and 7 features. The label is a dummy variable 1=diabetes 0=not diabetes
# 
# ###This file does not have the header row (I handtyped them in. Is there a better way?)
# 
# ###I think that BMI, Glucose, and Age may be related to having diabetes

# In[1]:

#Import data into a dataframe and inspect basic structure of the data.
#no missing values, so imputation is not needed.

import numpy as np
import pandas as pd
import urllib
from bokeh.plotting import figure,output_notebook,show,VBox,HBox,gridplot 

filename = urllib.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data')
header = ['Pregnancy', 'Glucose', 'Hg', 'Triceps', 'Insulin', 'BMI', 'Pedigree', 'Age', 'Diabetes']

df = pd.read_csv(filename, index_col=False, header=None, names=header)

df.info()

output_notebook()
df.describe()


# In[2]:

len(df.columns)


# In[3]:

#Visually inspect all the features to see if there are outliers
#There are outliers in a few of the features, so may consider removing them and re-run the models

p = []

p1 = figure(plot_width=250, plot_height=250, title='Pregancy', x_axis_label='Diabetes', y_axis_label='Pregnancy')
p1.circle(df.Diabetes, df.Pregnancy, size=10, alpha=.5)

p2 = figure(plot_width=250, plot_height=250, title='Insulin', x_axis_label='Diabetes', y_axis_label='Insulin')
p2.circle(df.Diabetes, df.Insulin, size=10, alpha=.5)

p3 = figure(plot_width=250, plot_height=250, title='Triceps', x_axis_label='Diabetes', y_axis_label='Triceps')
p3.circle(df.Diabetes, df.Triceps, size=10, alpha=.5)

p4 = figure(plot_width=250, plot_height=250, title='Glucose', x_axis_label='Diabetes', y_axis_label='Glucose')
p4.circle(df.Diabetes, df.Glucose, size=10, alpha=.5)

p5 = figure(plot_width=250, plot_height=250, title='Hg', x_axis_label='Diabetes', y_axis_label='Hg')
p5.circle(df.Diabetes, df.Hg, size=10, alpha=.5)

p6 = figure(plot_width=250, plot_height=250, title='Pedigree', x_axis_label='Diabetes', y_axis_label='Pedigree')
p6.circle(df.Diabetes, df.Pedigree, size=10, alpha=.5)

p7 = figure(plot_width=250, plot_height=250, title='BMI', x_axis_label='Diabetes', y_axis_label='BMI')
p7.circle(df.Diabetes, df.BMI, size=10, alpha=.5)

p8 = figure(plot_width=250, plot_height=250, title='Age', x_axis_label='Diabetes', y_axis_label='Age')
p8.circle(df.Diabetes, df.Age, size=10, alpha=.5)

p.append(p1)
p.append(p2)
p.append(p3)
p.append(p4)
p.append(p5)
p.append(p6)
p.append(p7)
p.append(p8)

#print p


gplots = np.array(p).reshape(2,4)
print gplots

a = gridplot(gplots.tolist())
show(a)



# In[4]:

############################################     KNN    ########################################################
######This approach uses df indices to access data and label values######


from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

outcome = 'Diabetes'
features = ['Pregnancy', 'Glucose', 'Hg', 'Triceps', 'Insulin', 'BMI', 'Pedigree', 'Age']
data = df[features]
label = df[outcome]

#label.info()

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=.2, random_state=0)

# X_train = list(X_train.values.flatten())
# Y_train = list(Y_train.values.flatten())


train_ind = X_train.index.values
test_ind = X_test.index.values

diab_knn = KNeighborsClassifier(3).fit(data.loc[train_ind], label.loc[train_ind])

pred = diab_knn.predict(data.loc[test_ind])

correct = 0 

for a, b in zip(Y_test, pred):
    if a == b:
        correct +=1
    else:
        pass

print 'Correct number:', correct    
print 'accuracy:', float(correct)/len(Y_test)





# In[5]:

############################################     KNN    ########################################################
######This approach converts df to np arrays for classification#######


from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

outcome = 'Diabetes'
features = ['Pregnancy', 'Glucose', 'Hg', 'Triceps', 'Insulin', 'BMI', 'Pedigree', 'Age']
data = df[features]
label = df[outcome]

print outcome

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=.2, random_state=0)

X_train = X_train.values
Y_train = Y_train.values

X_test = X_test.values
Y_test = Y_test.values


diab_knn = KNeighborsClassifier(3).fit(X_train, Y_train)

pred = diab_knn.predict(X_test)

correct = 0 

for a, b in zip(Y_test, pred):
    if a == b:
        correct +=1
    else:
        pass

print 'Correct number:', correct    
print 'accuracy:', float(correct)/len(Y_test)


#########Get the four permutations of predicted vs. observed values############

TN = 0
FN = 0
TP = 0
FP = 0
for a, b in zip(Y_test, pred):
    if a == b == 0:
        TN +=1
    elif a <> b & b == 0:
        FN +=1
    elif a == b == 1:
         TP +=1
    elif a <> b & b == 1:
         FP +=1
print "True Negative:", TN
print "False Negative:", FN
print "True Positive:", TP
print "False Positive", FP
    

# compare = pd.DataFrame(zip(Y_test, pred))
# zip(Y_test, pred)
# compare


# In[6]:

from sklearn import metrics

print metrics.classification_report(Y_test, pred)

######what is F1-score???


# In[77]:

##########Cross Validation of KNN################

from sklearn import cross_validation

data_arr = data.values
label_arr = label.values

data_arr
label_arr

fold = []
knn_accuracy = []

for i in range(1, 51):
    clf_knn = KNeighborsClassifier(i)

    scores = cross_validation.cross_val_score(clf_knn, data_arr, label_arr, cv=5)
    scores_avg = scores.mean()
    fold.append(i)
    knn_accuracy.append(scores_avg)
    
knn_chart = figure(width=500, height=500, title = 'KNN accuracy as a function of folds', x_axis_label='Folds', y_axis_label='Accuracy')  
knn_chart.circle(fold, knn_accuracy, alpha=.5, size=10, color='green')

show(knn_chart)
                   



     
    




# In[28]:

#######Naive Bayes no cross validation###########
from sklearn.naive_bayes import MultinomialNB
clf_nb = MultinomialNB()
clf_nb.fit(X_train, Y_train)
pred_nb = clf_nb.predict(X_test)

clf_nb.score(X_test, Y_test)


# In[39]:

#######Naive Bayes and cross validation###########

from sklearn.naive_bayes import MultinomialNB
clf_nb = MultinomialNB()

scores2 = cross_validation.cross_val_score(clf_nb, data_arr, label_arr, cv=5)

nb_accuracy = scores2.mean()

print ("Naive Bayes Accuracy: %0.2f" %(nb_accuracy))


# In[60]:

#####Compute final accuracy scores for two algorithems. Separating KNN into k>=8 and k<8########

knn_acc_df = pd.DataFrame(zip(knn_accuracy, fold), columns=['accuracy','fold'])

knn_accuracy_k_lt8 = knn_acc_df[(knn_acc_df.fold<8)].accuracy.mean()
knn_accuracy_k_ge8 = knn_acc_df[(knn_acc_df.fold>=8)].accuracy.mean()
print knn_accuracy_k_lt8
print knn_accuracy_k_ge8


# In[71]:

knn_nb = {'accuracy': [knn_accuracy_k_lt8, knn_accuracy_k_ge8, nb_accuracy]}
knn_nb


# In[76]:

########Final Bar Chart Comparison for the Prediction Accuracy between KNN and NB##########

from bokeh.charts import Bar, show

final_result = Bar(knn_nb, title = 'Prediction Accuracy: KNN vs. Naive Bayes', 
                   cat=['KNN Accuracy K<8',
                           'KNN Accuracy K>=8',
                           'Naive Bayes Accuracy'],
                   xlabel = 'Algorithms', ylabel = 'Prediction Accuracy')

show(final_result)

