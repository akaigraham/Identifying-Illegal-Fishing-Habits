# Capstone Project - Predicting Fishing Habits Using AIS and World Ocean Database Database
## README Outline
Within this [`README.md`](/README.md) you will find:
1. Introduction
2. Overview of Repository Contents
3. Project Objectives
4. Overview of the Process
5. Findings & Recommendations   
6. Conclusion / Summary

## Introduction
Build a classifier to identify whether a vessel is fishing.  Ultimate goal is to create a classifier that can be used by policymakers, regulators, and other stakeholders involved with preservation of ocean resources to identify vessels that are fishing.  

![`Vessels Plotted by Label`](/readme_imgs/map.png)

## Repository Contents
1. [`README.md`](/README.md)
2. [`notebook.ipynb`](/notebook.ipynb)
3. [`/datasets`](/datasets)
4. [`/tide_api_call.ipynb`](/tide_api_call.ipynb)

## Project Objectives
Build a classifier to predict whether a vessel is engaged in fishing activity in context of providing policymakers, regulators, and other ocean resource stakeholders a tool to identify vessels that are fishing.  Follow CRISP-DM machine learning process to explore dataset, prepare data for modeling, modeling, and post-model evaluation. Main performance metrics focused on were accuracy and recall as our dataset is not very sensitive to producing false positives.

## Overview of the Process:
Following CRISP-DM, the process outlined within [`notebook.ipynb`](/notebook.ipynb) follows 5 key steps, including:
1. Business Understanding: Outlines facts and requirements of the project.
2. Data Understanding: focused on unpacking data available and leveraged throughout classification tasks. Section will focus on the distribution of our data, and highlighting relationships between target and predictors.
3. Data Preparation: further preprocessing of our data to prepare for modeling.  Includes separating validation sets, handling missing values, and encoding certain columns
4. Modeling: this section iteratively trains a number of machine learning models, specifically using Decision Trees, Random Forests, XGBoost, and Neural Networks.
5. Evaluation: Final / optimal model is selected and final performance metrics of final model are discussed and evaluated.  Focused primarily on accuracy and recall as performance metrics.

## Findings & Recommendations
The best performing model identified was a tuned random forest model.  Final model scores:
- Train Accuracy: 93.8%
- Test Accuracy: 92.9%
- Train Recall: 97.3%
- Test Recall: 96.4%
