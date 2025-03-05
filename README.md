# Twitter Stance Analysis: 2020 US Presidential Election

UniPi Data Mining Project

## Introduction

The 2020 US Presidential Election was marked by intense competition between **Donald Trump** (Republican) and **Joe Biden** (Democrat). During this period, social media platforms, especially **Twitter**, became a hub for political discourse. This project focuses on **mining and analyzing Twitter users' stance** in relation to both candidates, leveraging natural language processing (NLP) and machine learning techniques to detect favorability trends over time.

## Features

- **Data Collection**: Scrapes tweets and replies mentioning the candidates.
- **Data Cleaning & Filtering**: Processes raw text, removes noise, and extracts meaningful content.
- **Stance Classification**: Labels tweets as **In Favor** or **Not in Favor** using machine learning models.
- **Trend Analysis**: Tracks users' stance shifts over key election events.
- **Adaptive Models**: Compares static, incremental, and sliding window models for improved accuracy.

## Dataset

The dataset consists of multiple CSV files containing:
- **Joe Biden’s tweets & replies**
- **Donald Trump’s tweets & replies**
- Timeframe: **September 1, 2020 – January 8, 2021**

Data was collected using the **Twint API** and preprocessed to remove links, images, and non-text elements.

## Models Used

- **Logistic Regression**
- **Support Vector Machines (SVM)** *(final chosen model)*
- **Multinomial Naïve Bayes**
- **Passive Aggressive Classifier**

We implemented **three learning strategies**:
1. **Static Model**: Trained once in September and applied throughout.
2. **Incremental Model**: Updated with new training data after each major event.
3. **Sliding Window Model**: Maintains a fixed training set by replacing old samples with new ones.

