# Machine Learning Projects Repository

Hey there! This is my personal repo with a bunch of machine learning projects I've been working on. You'll find all sorts of projects here, like sentiment analysis on movie reviews, predicting insurance purchases, linear regression models for vehicle weights, evaluating classification models with custom metrics, and classifying music genres based on audio features.

Each project has its own notebook with detailed explanations, code, and results. I've tried to make everything as clear and easy to follow as possible, but feel free to reach out if you have any questions or suggestions!

Here's what's inside:

- [Project 1: Sentiment Analysis of Movie Reviews Using Machine Learning](https://github.com/Juanfra21/DS_Projects/blob/main/movie_reviews_sentiment_analysis.ipynb)
- [Project 2: Predicting Insurance Product Purchases: KNN and SVM Model Comparison](https://github.com/Juanfra21/DS_Projects/blob/main/KNN_SVM_insurance_prediction.ipynb)
- [Project 3: Cross-Validated Linear Regression Model for Predicting Vehicle Curb Weight](https://github.com/Juanfra21/DS_Projects/blob/main/predicting_curb_weight.ipynb)
- [Project 4: Evaluating Classification Models: Custom Metrics and Scikit-Learn Functions](https://github.com/Juanfra21/DS_Projects/blob/main/classification_models_metrics.ipynb)
- [Project 5: Music Genre Classification from Audio Features](https://github.com/Juanfra21/DS_Projects/blob/main/music_genre_classification.ipynb)

Hope you find something interesting or useful in here! Let me know what you think or if you have any cool ideas for new projects.

---

## Project 1: Sentiment Analysis of Movie Reviews Using Machine Learning

For this project, I created a sentiment analysis model using machine learning to classify movie reviews as either positive or negative. Here I preprocess the text data, extract relevant features, train a machine learning model on it, and then evaluate how well the model performs. The goal is to build a model that can accurately determine if a movie review conveys a positive or negative sentiment just from analyzing the text.

**Key Steps:**

1. Data acquisition and preprocessing
2. Feature extraction using natural language processing (NLP) techniques
3. Model training and evaluation
4. Interpretation of results and discussion of challenges

## Project 2: Predicting Insurance Product Purchases: KNN and SVM Model Comparison

For this project, I dive into classification models to predict if insurance customers are likely to buy additional products from the company. Instead of using the typical binary logistic regression, I use two other techniques: K-Nearest Neighbors (KNN) and Support Vector Machines (SVM). The goal is to find the model that can make the most accurate predictions on whether a customer will make an additional purchase or not.

**Key Steps:**

1. Exploratory data analysis
2. Data preparation and feature selection
3. Model training and cross-validation
4. Performance evaluation and comparison

## Project 3: Cross-Validated Linear Regression Model for Predicting Vehicle Curb Weight

In this project, I work with data from the UC Irvine machine learning archive to build a linear regression model that can predict the curb weight of passenger vehicles. Here, I use cross-validation to ensure our model is robust and generalizable. The key variables that are considered are things like the vehicle's height, width, length, wheelbase, engine size, horsepower, peak RPM, and city mileage. The goal is to find the combination of these variables that best predicts a vehicle's curb weight.

**Key Steps:**

1. Data preprocessing and feature selection
2. Linear regression analysis
3. Cross-validation techniques
4. Model performance evaluation

## Project 4: Evaluating Classification Models: Custom Metrics and Scikit-Learn Functions

For this project, I evaluate the performance of classification models using a combination of custom Python code functions and pre-built functions from the popular scikit-learn library. The dataset consists of around 180 labeled binary observations. The goal is to get hands-on experience with different techniques for assessing how well the classification models are performing, both by writing evaluation functions and using the tools provided by scikit-learn.

**Key Steps:**

1. Computation of confusion matrix using Pandas
2. Development of custom Python functions for accuracy, precision, sensitivity, specificity, and F1 Score
3. Construction of ROC curves and computation of AUC
4. Comparison of custom implementations with scikit-learn functions

## Project 5: Music Genre Classification from Audio Features

In this project, I explore music genre classification using audio features as input data. I work with a large dataset of 114,000 Spotify tracks spanning 114 different genres, which were consolidated into 56 distinct classes through hierarchical clustering.

To tackle this multiclass classification problem, I developed four different models: a neural network, an XGBoost model, a K-Nearest Neighbors (KNN) classifier, and an ensemble model composed of four weak learners. The goal is for each of these models to predict the probability of a given song belonging to each of the 56 genres based solely on its audio features.

**Key Steps:**

1. Data preprocessing and feature extraction
2. Model development: neural network, XGBoost, KNN, and ensemble
3. Evaluation using Top-3 categorical accuracy
4. Comparison of model performance

## Contact

For any questions or inquiries, feel free to contact me at juanfraleonhardt@gmail.com
