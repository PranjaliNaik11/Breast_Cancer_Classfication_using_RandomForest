# Breast-Cancer-Detection-using-Blob-Detection

#### **Data**: 
The data consists of mamogram images from 322 patients.

#### **Methodology**:
1) The images are scanned individually to detecet the fetaures of any suspicious mass.
2) The x co-ordinate, y co-ordinate and radius of this mass is extracted and stored in a csv file.
3) The mammogram images with area of the mass > mean(area) is labelled as Malignant and others as Benign.
4) The resulting dataset is then used to design a Machine Learning model to predict future mammogram images based on all available features.

#### **Algorithm used:**
#### Random Forest Classifier algorithm is used to train a machine learning model to classify the breast cancer features data.
