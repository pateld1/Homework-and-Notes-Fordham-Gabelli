
# coding: utf-8

# # CISC 6930 Assignment 2
# ### Completed by Darshan Patel

# In[1]:

# Import packages 
import pandas as pd
import numpy as np
import time


# ### Question 1:  Implement the KNN classifier

# Accept two data files: a **spam_train.csv** file and a **spam_test.csv** file. Both files contain examples of e-mail messages, with each example having a class label of either "1" (spam) or "0" (no-spam). Each example has 57 (numeric) features that characterize the message. The classifier should examine each example in the **spam_test** set and classify it as one of the two classes. The classification will be based on an **unweighted** vote of its $k$ nearest examples in the **spam_train** set. Measure all distance using regular Euclidean distance: 
# $$ d(x,y) = \sqrt{ \sum_i (x_i - y_i)^2 } $$ 

# In[2]:

# Mark starting time
print('Marking down starting time')
start = time.time()


# In[3]:

# Read in csv files
spam_train = pd.read_csv('spam_train.csv')
spam_test = pd.read_csv('spam_test.csv')

# Separate the features from the class/labels for 
# both the training and testing set
train_features = spam_train.iloc[:,:-1]
train_class = spam_train['class']
test_features = spam_test.iloc[:,1:-1]
test_labels = spam_test['Label']

# Normalize the features 
n_train_feat = (train_features - train_features.mean()) / train_features.std()
n_test_feat = (test_features - test_features.mean()) / test_features.std()


# In[4]:

# List of specific k values to use
k = [1,5,11,21,41,61,81,101,201,401]


# In[5]:

# Calculate the Euclidean distance between two messages 
def euclidean_distance(X, Y):
    return np.sqrt(np.sum((X - Y)**2))


# In[6]:

# Get the indices of the 401 closest neighbors of a certain message 
def getClosestNeighbors(train, test, test_value):
    
    train = np.array(train)
    test = np.array(test)
    
    n = train.shape[0]
    dist = []
    
    for i in range(n):
        d = euclidean_distance(train[i], test[test_value])
        dist.append([d, i])
        
    dist = sorted(dist)[:401]
    
    index = []
    for i in dist:
        index.append(i[1])
        
    return index


# In[7]:

# Calculate the mode of a list of 0s and 1s using the average
def mode(v):
    
    if np.mean(v) > 0.5: mode = 1
    else: mode = 0
        
    return mode


# In[8]:

# Perform the KNN algorithm on a data set 
# and get the accuracy for each k value used 
def KNN_classifier(train_f, test_f):
    
    train_f = np.array(train_f)
    test_f = np.array(test_f)
    
    accuracies = []
    n = test_f.shape[0]
    counts = np.zeros(10)
    
    for test_point in range(n):

        indices = getClosestNeighbors(train_f, test_f, test_point) 
        spam_or_not = []
        for j in indices:
            spam_or_not.append(train_class[j]) 

        for a in range(len(k)):
            m = mode(spam_or_not[:(k[a])])
            if m == test_labels[test_point]:
                counts[a] += 1
                
    for c in counts:
        accuracies.append(100 * c/n)
        
    return accuracies


# (a) Report **test** accuracies when $k = 1,5,11,21,41,61,81,101,201,401$ **without** normalizing the features. 

# In[9]:

# Get the test accuracies when performing the KNN algorithm using 
# regular features and print them out respectively 
accuracies = KNN_classifier(train_features, test_features)
print('Without normalizing the features, ')
for a in range(len(accuracies)):
    print('test accuracy for k =', k[a], ':', accuracies[a], '%')
print('\n')


# (b) Report **test** accuracies when $k = 1,5,11,21,41,61,81,101,201,401$ **with z-score normalization** applied to the features. 

# In[10]:

# Get the test accuracies when performing the KNN algorithm using
# normalized features and print them out respectively 
accuracies_normalized = KNN_classifier(n_train_feat, n_test_feat)
print('With z-score normalization applied to the features, ')
for a_n in range(len(accuracies_normalized)):
    print('test accuracy for k =', k[a_n], ':', accuracies_normalized[a_n], '%')
print('\n')


# (c) In the previous case, generate an output of KNN predicted labels for the first $50$ instances (i.e. $t1-t50$) when $k = 1,5,11,21,41,61,81,101,201,401$ (in this order). For example, if $t5$ is classified as class 'spam' when $k=1,5,11,21,41,61$ and classified as class 'no-spam' when $k=81,101,201,401$, then the output line for $t5$ should be: $$ t5 ~ \textbf{spam, spam, spam, spam, spam, spam, no, no, no, no} $$ 

# In[11]:

# Prints the output of a certain number of instances of whether it is 
# spam or not depending on the k value 
def print_output(instances): 

    test = np.array(n_test_feat.iloc[:instances,])
    train = np.array(n_train_feat)

    for row in range(instances):

        cn = getClosestNeighbors(train, test, row)

        spam_or_not = []

        for closest in cn:
            spam_or_not.append(train_class[closest])

        instance = []

        for val in k:

            classified = mode(spam_or_not[:val])
            if classified == 1:
                instance.append('spam')
            else:
                instance.append('not')

        print(spam_test.iloc[row, 0], instance[1], 
              instance[2], instance[3], instance[4], 
              instance[5], instance[6], instance[7], 
              instance[8], instance[9])


# In[12]:

# Print the output for the first 50 instances
print('Output of KNN predicted labels for the first 50 instances when \n', 
      'k = 1, 5, 11, 21, 41, 61, 81, 101, 201, and 401 respectively.')
print_output(50)


# In[13]:

# Mark end time
end = time.time()


# In[14]:

# Print the elapsed time of the entire program
print("Elapsed Time:", end - start, "s")


# (d) What can you conclude by comparing the KNN performance in (a) and (b)? 

# **Answer:** By normalizing the features, the KNN algorithm was able to be $10\%$ to $15\%$ more accurate with classifing whether the message is spam or not. 

# (e) Describe a method to select the optimal $k$ for the KNN algorithm. 

# **Answer:** To select the optimal $k$ from a list of $k$ values, split the test data into $k$ subsets randomly and perform the KNN algorithm on each subset using its specific $k$ value. Then for each $k$ value, calculate the individual performance metrics and select the optimal $k$ with the highest performance metric. 
