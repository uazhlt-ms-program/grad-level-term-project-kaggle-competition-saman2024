[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/ml3o3YLw)
# Task

See https://uazhlt-ms-program.github.io/ling-539-course-blog/assignments/class-competition

Competition: https://www.kaggle.com/competitions/ling-539-sp-2024-class-competition

Invitation URL: https://www.kaggle.com/t/33320feca02e42bd924c3473e3a8c2fd

# Notes
- **At least one of your submitted solutions must use one or more of the classification algorithms covered in this course**
- You are not obligated to use Python
- You may delete or alter any files in this repository
- You are free to add dependencies
  - Ensure that your code can be installed/used on another machine running Linux or MacOS (consider containerizing your project with Docker or an equivalent technology)
  
  
  ---
title: "Efficient Movie classifier"
slug: "/meihami/class-competition"
date: 2024-05-01
author: Saman Meihami
description: " logistic regression classifier trained on text data represented using TF-IDF"
tags:#Logistic_regression, #TF-IDF
  - class competition
---

## Task

My task is to classify documents for fictional Marvin, determining whether they are movie reviews and if so, whether they are positive or negative. The labels are: 0 for not a movie review, 1 for positive, and 2 for negative. With a dataset of 70,317 training samples and 17,580 testing samples, I aimed to improve performance without resorting to deep learning or transformer models.

## Approach

I implemented a text classification pipeline model using TF-IDF vectorization and logistic regression. Initially, the training dataset was loaded and preprocessed to handle missing values in the text column. The pipeline was then defined, comprising a TfidfVectorizer to convert text data into numerical TF-IDF representations and a LogisticRegression classifier for label prediction. After training the pipeline on the training dataset, the test data was loaded and processed similarly to the training data. Predictions were made using the trained pipeline, and the results were saved in a submission file for further analysis. 



## Results

This approach yielded promising results, with an accuracy of 92.5 percentachieved on the development dataset, which accounted for 20 percent of the training data. When evaluated on the test data submitted to the Kaggle website, the model achieved an accuracy score of 91.9 percent. These results demonstrate the effectiveness of the TF-IDF vectorization and logistic regression classifier in accurately classifying text data. Additionally, the modular structure of the pipeline facilitated experimentation with different vectorization and classification techniques, allowing for further optimization and improvement of the model's performance.