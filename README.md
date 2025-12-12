# Using a Neural Net to Forecast Favorita's KPIs

Author: Christopher Kuzemka

# Project Summary

This project's code was originally submitted for Fordham's Data Mining class in Fall of 2024. The goal of this project was to deploy a SARIMAX model to forecast sales for Favorita. The original project's outcome provided a working SARIMAX model (INSERT SCORING METRIC) that did not scale and was only able to be used on the Grocery I category for Store \#45. 

Building off of the old project, our goal is to scale out a more robust and better performant model to beat our SARIMAX model in its own categorical prediction and across other categories. 

PROJECT RESULTS

# Scope Limits

- technical considerations on scope
- discuss of previous Kaggle competition winners and their relationship to our approach. 

## Technical Changes 

# About the Dataset

## Data Dictionary (Feature Engineered KPIs also)

## Context

- what it is in general
- about the data

## Relevant Cleaning

- the steps of cleaning

Model Fix
- revision of model encoding and cleaning:
    - it was discovered that there may be a simulated economic crash happening based on an error towards the end that changes nulls to 0
        - will ffill and see what that looks like. Will also observe what the issue looked like previously when the error was sustained
    - model complexity could be reduced due to the isolation towards one store meaning the location should remain the same

# Modeling

## Overall Considerations

- LSTM priority (pros, cons, reasons why chosen) (RNN)
- TCM secondary (pros, cons, reasons why chosen) (CNN)
- ensemble SARIMAX modeling (pros and cons)

## Model Prep

## Chosen Models and Performance Analysis


# Bibliography (Relevant Links and Work)

[Mean Absolute Scaled Error (MASE) in Forecasting](https://medium.com/@ashishdce/mean-absolute-scaled-error-mase-in-forecasting-8f3aecc21968)