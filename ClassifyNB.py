from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    clf = GaussianNB()
    ### fit the classifier on the training features and labels
    
    clf.fit(features_train, labels_train)
    ### return the fit classifier
    return clf
    ### your code goes here!
    
def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()
    ### fit the classifier on the training features and labels
    
    clf.fit(features_train, labels_train)

    ### fit the classifier on the training features and labels

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    ### Calcula precision 0.884 = acierta 88.4%
    accuracy = clf.score(features_test, labels_test)

    return accuracy    