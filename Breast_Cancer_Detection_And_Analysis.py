########

#Author: Pranjali Naik
#Title: Blob detection from brest mammograms using Gaussian method
#       and using predictive modeling.

#Note: Save the images("img_Pranjali") folder and the python file in the same directory.
#      There are 322 images so the entire run time is around 30 minutes.

########



from math import sqrt
from skimage.feature import blob_dog
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import glob

#This function reads the images from the "img_Pranjali" folder and detects the blob present in the image using
#the difference of gaussian function and stores it as a sequence.
def find_blob():
    #The "img" in the path indicates the folder name of the images.
    #The image folder is stored in the same path as the python file for execution.
    for img in glob.glob("img_Pranjali/*.jpg"):
            # Read the image
            image_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

            # Gaussian filter
            blobs_dog = blob_dog(image_gray, max_sigma=150, threshold=.5)

            # Calculate the radii
            blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

            # List for blob
            blobs_list = [blobs_dog]
            color = ['lime']
            title = ['Difference of Gaussian']
            sequence = zip(blobs_list, color, title)

            # Calling function to detect parameters.
            find_parameters(sequence)

#This function extracts the features from the images individually and then
#stores it into a CSV file.
def find_parameters(sequence):
    for idx, (blobs, color, title) in enumerate(sequence):
        a = [0]
        b = [0]
        d = [0]
        for blob in blobs:
            y, x, r = blob #We get x-coordinate,y-coordinate and radius
            if r not in a:
                a.append(r)
                b.append(x)
                d.append(y)
            #Calculation of the important parameters
            maximum_radius = sum(a) / len(a)
            max_rad = max(a)
            location = a.index(max_rad)
            x_coordinate = b[location]
            y_coordinate = d[location]
            area = 3.14 * maximum_radius * maximum_radius
            perimeter = 2 * 3.14 * maximum_radius
            compactness = (perimeter * perimeter) / (area - 1)

            #Creating a string to store it in CSV file
            row = str(x_coordinate) + "," + str(y_coordinate) + "," + str(maximum_radius) + "," + str(
                perimeter) + "," + str(area) + "," + str(compactness) + "\n"


        #Calling the CSV function to create a new csv file and store the values in it.
        write_into_csv(row)


#This function creates a new csv file and then stores the paramter values into the
# CSV file.
def write_into_csv(row):
    with open('Pranjali_Data.csv', 'a') as newFile:
        newFile.write(row)

#This function is used to add the required headers to the CSV file and then
#prepare it for the further analysis.
def prepare():
    with open('Pranjali_Data.csv', 'r') as original: data = original.read()
    with open('Pranjali_Data.csv', 'w') as modified: modified.write("x,y,radius,perimeter,area,compactness\n" + data)
    t = pd.read_csv('Pranjali_Data.csv')
    mean = t['area'].mean()

    df = pd.DataFrame(t, columns=['x', 'y', 'radius', 'perimeter', 'area', 'compactness'])
    df['diagnosis'] = np.where(df['area'] > mean, 'M', 'B')

    df.to_csv('Pranjali_Data.csv', sep=',', index=False)

#This funcrion is used to split the data into train and test data based on
# the predictors and target parameters.
def split_data():
    from sklearn.linear_model import LogisticRegression

    data = pd.read_csv('Pranjali_Data.csv')
    df=pd.DataFrame(data,columns=['x', 'y', 'radius', 'perimeter', 'area', 'compactness','diagnosis'])

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    data_m = data[['x', 'y', 'radius', 'perimeter', 'area', 'compactness']]
    data_mw = ['x', 'y', 'radius', 'perimeter', 'area', 'compactness']
    predictors = data_m.columns[1:6]
    target = 'diagnosis'

    X = data.loc[:, predictors]
    y = np.ravel(data.loc[:, [target]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    predict(X_train, y_train, X_test, y_test)

    train, test = train_test_split(df, test_size=0.25)
    heatmap(data)

#This function is used to predict the accuracy of the data
# as per the Random Forest Classifier.
def predict(X_train, y_train, X_test, y_test):
    # Initiating the model:

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    scores = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=10).mean()

    print("The mean accuracy with 10 fold cross validation is %s" % round(scores * 100, 2))

    predicted = rf.predict(X_test)

    print('Confusion matrix',confusion_matrix(y_test,predicted))
    report = classification_report(y_test, predicted)
    print(report)

#This function is used to generate a heat map of the data being provided.
def heatmap(data1):
    plt.figure(figsize=(14, 14))
    sns.heatmap(data1.corr(), vmax=1, square=True, annot=True)
    plt.show()

#This is the main function which is used to initiate the process.
if __name__ == "__main__":
    find_blob()
    prepare()
    split_data()
