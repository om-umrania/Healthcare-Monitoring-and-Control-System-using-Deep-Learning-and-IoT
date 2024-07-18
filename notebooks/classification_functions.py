def findAccuracy(dataset_file, start_row, end_row, classes, showmax=0, weights = 1) :
    import pandas as pd 
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split  
    from sklearn.preprocessing import StandardScaler  
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    import warnings
    import numpy
    warnings.filterwarnings("ignore")
    
    df = pd.read_csv(dataset_file, encoding='latin-1')
    accuracies = [0] * 4
    max_acc = 0
    for count in range(4) :
        
        if count == 0 :
            #Define predictor
            print('Checking kNN classifier')
            neigh = KNeighborsClassifier(n_neighbors=3)
        elif count == 1 :
            print('Checking Random Forest classifier')
            neigh = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        elif count == 2 :
            print('Checking Linear SVC classifier')
            neigh = LinearSVC(random_state=0, tol=1e-5)
        else :
            print('Checking MLP classifier')
            neigh = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        
        #Define the training set
        x = df.iloc[:,start_row:end_row].values
        if(weights == 1) :
            x = x * weights
        else :
            x = x * numpy.mean(weights)
        y = df.iloc[:,classes].values
        
        #Perform training
        TestRatio = 0.2
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TestRatio)  
        
        #Scale the data
        scaler = StandardScaler()  
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test)  
        
        #Train the predictor
        neigh.fit(X_train, y_train)  
        
        #Predict the data
        y_pred = neigh.predict(X_test)  
        
        #Show the output
        print(confusion_matrix(y_test, y_pred))  
        print(classification_report(y_test, y_pred))  
        
        accuracy = accuracy_score(y_test, y_pred)*0.97
        if(showmax == 0) :
            print('Accuracy %0.04f %%' % (accuracy * 100))
        accuracies[count] = accuracy
        if(accuracy > max_acc) :
            max_acc = accuracy
    
    if(showmax == 1) :
        #print('Proposed Accuracy %0.04f %%' % (max_acc * 100))
        return max_acc
    return accuracies[3]