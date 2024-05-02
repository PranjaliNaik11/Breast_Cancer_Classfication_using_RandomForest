# Breast-Cancer-Detection-using-Blob-Detection
We import a dataset of mammogram images to detect the presence of a blob present on the image. We extract the basic features from the image. We perform Random Forest Classification to find how well the presence of breast cancer is detected.
### **Data**: The data consists of mamogram images from 322 patients.

### **Methodology**:
#### 1) The images are scanned individually to detecet the fetaures of any suspicious mass.
#### 2) The x co-ordinate, y co-ordinate and radius of this mass is extracted and stored in a csv file.
#### 3) The mammogram images with area of the mass > mean(area) is labelled as Malignant and others as Benign.
#### 4) The resulting dataset is then used to design a ML model to predict future mammogram images based on all available features.

### **Algorithm used:**:
#### Random Forest Classifier algorithm is used to train a machine learning model to classify the breast cancer features data.
