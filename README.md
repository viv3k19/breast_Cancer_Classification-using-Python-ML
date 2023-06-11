# Breast Cancer Classification
This repository contains code for classifying breast cancer using various machine learning algorithms. The code utilizes a dataset that includes several features extracted from breast tumor images, such as radius, texture, perimeter, area, smoothness, compactness, concavity, and more.

The main objective of this code is to accurately predict the diagnosis of breast tumors as either malignant (cancerous) or benign (non-cancerous) based on the provided features. By leveraging machine learning techniques, the code aims to assist in early detection and accurate diagnosis of breast cancer, which is crucial for effective treatment and improved patient outcomes.

The code follows a standardized workflow, starting with data preprocessing steps like dropping unnecessary columns and encoding the target variable. The dataset is then split into training and testing sets, and the features are scaled using standardization to ensure optimal model performance.

Several popular machine learning algorithms are implemented and evaluated, including Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and Support Vector Classifier. Each algorithm is trained on the training data and tested on the unseen testing data to measure its accuracy in classifying breast tumors.

The results of the classification accuracies achieved by each algorithm are displayed, providing insights into their performance. These accuracies indicate the reliability and effectiveness of the implemented algorithms in classifying breast tumors based on the provided features.

By sharing this code repository, we aim to contribute to the field of breast cancer research and assist healthcare professionals in making informed decisions regarding tumor diagnosis. The code can be used as a foundation for further research and development of more advanced and accurate breast cancer classification models.

Please refer to the repository for the complete code implementation, detailed instructions, and the necessary dataset to get started with breast cancer classification using machine learning algorithms.

Note: It is important to consult with medical professionals and domain experts when interpreting the results of the classification models and making clinical decisions.

This repository contains code for classifying breast cancer using different machine learning algorithms. The dataset used for training and testing the models is available in `data.csv`.

## Dataset

The dataset contains the following columns:

- Diagnosis
- Radius Mean
- Texture Mean
- Perimeter Mean
- Area Mean
- Smoothness Mean
- Compactness Mean
- Concavity Mean
- Concave Points Mean
- Symmetry Mean
- Fractal Dimension Mean
- Radius SE
- Texture SE
- Perimeter SE
- Area SE
- Smoothness SE
- Compactness SE
- Concavity SE
- Concave Points SE
- Symmetry SE
- Fractal Dimension SE
- Radius Worst
- Texture Worst
- Perimeter Worst
- Area Worst
- Smoothness Worst
- Compactness Worst
- Concavity Worst
- Concave Points Worst
- Symmetry Worst
- Fractal Dimension Worst

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these libraries using the following command:

```shell
pip install numpy pandas matplotlib seaborn scikit-learn
```
# Usage
1. Clone the repository and navigate to the project directory.
```shell
git clone https://github.com/viv3k19/breast_Cancer_Classification-using-Python-ML
```
2. Run the breast_cancer_classification.py file.
```shell
python breast_cancer_prediction.py
```

## The code will perform the following steps:

* Read the dataset from data.csv.
* Preprocess the data by dropping unnecessary columns and encoding the target variable.
* Split the data into training and testing sets.
* Scale the features using standardization.
* Train and evaluate different machine learning algorithms, including Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and Support Vector Classifier.
* Display the accuracy of each algorithm.

## Results
The accuracy results of the different algorithms are as follows:

* Logistic Regression Method: 0.970760
* Decision Tree Classifier Method: 0.923977
* Random Forest Classifier Method: 0.964912
* Support Vector Classifier Method: 0.959064

## Analytics
Distributions

![visualization](https://github.com/viv3k19/breast_Cancer_Classification-using-Python-ML/assets/82309435/47f498d2-beb8-413f-a473-03ed6f610457)

Values

![visualization (1)](https://github.com/viv3k19/breast_Cancer_Classification-using-Python-ML/assets/82309435/8a8d9ad1-d5ed-4c98-80b6-0bd821f2c010)

Categorical distributions

![visualization (2)](https://github.com/viv3k19/breast_Cancer_Classification-using-Python-ML/assets/82309435/3770a39b-2769-424a-a088-f854af1a4a77)

Swarm plots

![visualization (3)](https://github.com/viv3k19/breast_Cancer_Classification-using-Python-ML/assets/82309435/e214392a-5f93-466f-8892-6670fe0c2bfc)

# Project Creator
* Vivek Malam - Feel free to contact me at viv3k.19@gmail.com for any questions or feedback.
