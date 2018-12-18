print("CISC 6930 Assignment 3")
print("Darshan Patel")

print("Importing scipy, numpy, pandas and time.")

# Import packages
from scipy.io import arff 
import numpy as np
import pandas as pd
import time 

print("Read in data.")
# Load data
data, metadata = arff.loadarff("veh-prime.arff")

# Create dataframe of the data
x = pd.DataFrame(data)

print("Filter Method initiating.")

# Reclassify the class labels 
x['CLASS'] = np.where(x['CLASS'] == b'noncar', 0, 1)

# A function to calculate the Pearson Correlation Coefficient 
# between two variables
def pearson(x,y):
    
    n = len(x)
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    mean_x = 0
    mean_y = 0
    
    for i in range(n):
        
        sum_sq_x += x[i]**2
        sum_sq_y += y[i]**2
        sum_coproduct += x[i] * y[i]
        mean_x += x[i]
        mean_y += y[i]
    
    mean_x = mean_x / n
    mean_y = mean_y / n
    pop_sd_x = np.sqrt((sum_sq_x / n) - (mean_x**2))
    pop_sd_y = np.sqrt((sum_sq_y / n) - (mean_y**2))
    cov_x_y = (sum_coproduct / n) - (mean_x * mean_y)
    correlation = cov_x_y / (pop_sd_x * pop_sd_y)
    
    return correlation

# (a) List the features from highest $|r|$ to lowest, along with their $|r|$ values. Why would one be interested in the absolute value of $r$ rather than the raw value? 

# Separate the features from the label
X = x.iloc[:,:-1]
Y = x.iloc[:,-1]

# Create list to store PCC values
r = list()

# For each feature, calculate its |r| values with respect to the class label
# and append it to the above list
for i in range(36):
    r.append(np.abs(pearson(x.iloc[:,i], Y)))

# Sort the list of PCCs in descending order
r_sorted = pd.DataFrame(sorted(r, reverse = True))

# Obtain the appropriate feature numbers whose |r| values are 
# in descending order 
features_rank = pd.DataFrame(np.argsort(r)[::-1])

# Create a dataframe that shows the feature number and its respective 
# |r| value
df = pd.concat((features_rank, r_sorted),axis = 1)

# Rename columns
df.columns = ["Feature Number", "|r| score"]

# Keep the list of sorted feature numbers in terms of |r| score for
# later use 
high_low_r = df.iloc[:,0]

# Show the dataframe
for i in range(len(r)):
    print("Feature", df.iloc[i,0], "'s |r| value is", round(df.iloc[i,1],3))


print("Answer to Question 3a: The parity of the Pearson correlation coefficient tells whether the data is positively or negatively correlated. In our case, we are not looking at if the correlation is positive or negative but rather just how much two column of data are correlated to each other from 0 being no correlation to 1 being a perfect correlation.")

# (b) Select the features that have the highest $m$ values of $|r|$ and run LOOCV on the dataset restricted to only those $m$ features. Which value of $m$ gives the highest LOOCV classification accuracy and what is the value of this optimal accuracy? 

# A function to run LOOCV on a dataset, with features and class labels
# where features to use is given
def KNN_LOOCV(df, feats, k = 7):
    
    # Make the dataset an array and store its dimensions in variables
    #df = np.array(df)
    rows = df.shape[0]
    cols = df.shape[1]
    
    # Instantiate a matrix to store distances for each data value
    # its n - 1 neighbors
    dist = np.zeros((rows, rows - 1))
    
    # Create an empty list for keeping track of LOOCV 
    # classification accuracies
    accuracies = list()
    
    # Iterate over all features given
    for i in feats:
        
        # Initialize number of correct classifications to 0
        count = 0
        
        for j in range(rows):
            
            test_index = df.index.isin([j])
            train = np.array(df[~test_index])
            test = np.array(df[test_index])
            
            # Create Xtrain and Ytrain from the training set
            Xtrain, Ytrain = np.split(train, [cols - 1], axis = 1)
            Xtrain = Xtrain[:,i]
            
            # Create Xtest and Ytest from the test set
            Xtest, Ytest = np.split(test, [cols - 1], axis = 1)
            Xtest = Xtest[:,i]
            
            # Calculate the Euclidean distance between Xtrain and Xtest
            dist[j] = np.add(dist[j], np.sqrt((Xtrain - Xtest)**2))
            
            # Obtain the closest k neighbors by sorting the distances
            # and getting the top indexes 
            indices = np.argsort(dist[j])[:k]
            
            # Get the majority vote of the k neighbors
            mode = 1 if np.mean(Ytrain[indices]) > 0.5 else 0

            # If the majority vote is accurate, correct 
            # classification increases by 1
            if mode == Ytest:
                count = count + 1
        
        # Calculate the accuracy of adding the feature 
        accuracy = 100 * count / rows
        print("Accuracy of adding feature", i, "is:", round(accuracy, 3), '%')
        
        # Store accuracy 
        accuracies.append((100 * count) / rows)
        
    # Return the list of accuracies
    return accuracies

start = time.time()
# Run the KNN algorithm with LOOCV on the dataset using the 
# ranked feature based on |r|
accuracies = KNN_LOOCV(x, high_low_r)
end = time.time()
print("Time to run Filter Method:", end - start)

# Create dataframe to show m, feature number added and aggregating accuracy
df = pd.DataFrame({"m": list(range(1, len(high_low_r) + 1)), 
                  "Feature # Added": high_low_r,
                  "Aggregating Accuracy": accuracies})

# Show the above dataframe
df[["m", "Feature # Added", "Aggregating Accuracy"]]


print("Answer to Question 3b: The highest LOOCV classification accuracy is when m = 20 and the LOOCV accuracy is 95.153664%")

# ## Question 4: Wrapper Method
# 
# Starting with the empty set of features, use a greedy approach to add the single feature that improves performance by the largest amount when added to the feature set. This is called Sequential Forward Selection. Define perfomance as the LOOCV classification accuracy of the KNN classifier using only the features in the selection set (including the candidate feature). Stop adding features only when there is no candidate that when added to the selection set increases the LOOCV accuracy. 

print("Wrapper Method initiating.")
# A function to run LOOCV on a dataset, with features and class labels
# where features to use and calculated Euclidean distances is given
def KNN_LOOCV_wrapper(df, feat, prior_distances, k = 7):
    
    # Make the dataset an array and store its dimensions in variables
    rows = df.shape[0]
    cols = df.shape[1]
    
    # Instantiate a matrix to store distances for each 
    # data value and its (n - 1) neighbors
    distances = np.zeros((rows, rows - 1))
    
    # Create an empty list for keeping track of LOOCV 
    # classification accuracies
    accuracies = list()
        
    # Initialize number of correct classifications to 0
    count = 0

    for j in range(rows):

        test_index = df.index.isin([j])
        train = np.array(df[~test_index])
        test = np.array(df[test_index])

        # Create Xtrain and Ytrain from the training set
        Xtrain, Ytrain = np.split(train, [cols - 1], axis = 1)
        Xtrain = Xtrain[:,feat]

        # Create Xtest and Ytest from the test set
        Xtest, Ytest = np.split(test, [cols - 1], axis = 1)
        Xtest = Xtest[:,feat]

        # Calculate the Euclidean distance between Xtrain and Xtest
        distances[j] = np.add(prior_distances[j],
                              np.sqrt((Xtrain - Xtest)**2))

        # Obtain the closest k neighbors by sorting the distances
        # and getting the top indexes 
        indices = np.argsort(distances[j])[:k]

        # Get the majority vote of the k neighbors
        mode = 1 if np.mean(Ytrain[indices]) > 0.5 else 0

        # If the majority vote is accurate, correct 
        # classification increases by 1
        if mode == Ytest:
            count = count + 1

        # Calculate the accuracy of adding the feature 
        accuracies.append((100 * count) / rows)
        
    # Return the last accuracy and distances 
    return accuracies[-1], distances

# A function to execute sequential forward selection on a dataset
def sfs(df): 
    
    dist = np.zeros((df.shape[0], df.shape[0] - 1))
    
    # Store number of features in a variable
    num_features = df.shape[1] - 1
    
    # Store original list of features to be worked with
    features = list(range(0, num_features))
    
    # Instantiate a vector to store the features that improve 
    # performance by the greatest amount
    best_feat = []
    
    # Instantiate a vector to store aggregating LOOCV 
    # classification accuracies
    max_accuracy = [0]
    
    # Create a variable to store previous accuracy value for testing
    prev_acc = 0
    
    # Run KNN LOOCV over the range of possible features 
    for i in range(num_features):
        
        # Print the iteration number and the 
        # current feature selection set
        print("Iteration", i, "'s Best Feature List:", best_feat)
        
        # Instantiate an empty vector to store LOOCV accuracy, 
        # feature number and distance array for all features added to
        # the best feature list
        acc_feat_dist = []
        
        # Run KNN with LOOCV over all possible features that can be added 
        for j in features:
            a, temp_dist = KNN_LOOCV_wrapper(df, j, dist)
            acc_feat_dist.append((a, j, temp_dist))
            
        # Store the maximum accuracy obtained and its feature number
        new_acc = max(acc_feat_dist)[0]
        best_feat_num = max(acc_feat_dist)[1]
        aggregate_dist = max(acc_feat_dist)[2]
        
        # If the maximum accuracy is not greater than the previous accuracy 
        # value, stop the sequential forward selection process, 
        # else add the maximum accuracy and best feature number to its 
        # respective lists. Remove the new feature from the list of features
        # to test on, update the distances and store the new acccuracy value
        # in the previous variable
        if new_acc <= prev_acc:
            break
        else:
            print("Accuracy of adding feature", best_feat_num, "is:",
                  round(new_acc, 3), '%')
            max_accuracy.append(new_acc)
            best_feat.append(best_feat_num)
            features.remove(best_feat_num)
            dist = aggregate_dist
            prev_acc = new_acc
    
    # Remove the initial accuracy test value
    del max_accuracy[0]
    
    # Return the feature selection set and its LOOCV accuracy list
    return(best_feat, max_accuracy)


# (a) Show the set of selected features at each step as it grows from size zero to its final size (increasing in size by exactly one feature at each step).

print("Answer to Question 4a:")
start = time.time()
# Run the above method on the dataset
test_feat, test_acc = sfs(x)
end = time.time()
print("Time to Run Wrapper Method:", end - start)


# Print the LOOCV accuracies using a dataframe
#print("LOOCV Accuracies:")
df = pd.DataFrame({"Feature Added": test_feat,
                   "LOOCV Accuracy": test_acc})
df

# (b) What is the LOOCV accuracy over the final set of selected features? 

print("Answer to Question 4b:")

# Run KNN LOOCV on the final set of selected features
print("LOOCV accuracy over the final set of selected features: ",
      round(KNN_LOOCV(x, test_feat)[-1],3), '%')

