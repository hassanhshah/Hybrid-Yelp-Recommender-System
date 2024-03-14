# Hybrid Recommendation Model for Yelp Ratings

## Description

This project implements a hybrid recommendation system combining item-based collaborative filtering (CF) with a model-based approach using XGBoost. The goal is to predict Yelp ratings of businesses based on user reviews. The hybrid model aims to leverage the strengths of both approaches to provide accurate rating predictions.

## Methodology

The recommendation system consists of two main components:

- **Item-based Collaborative Filtering (CF):** This component calculates similarity scores between items (businesses) using Pearson similarity. It addresses cases such as new users or businesses not present in the training dataset through default rating mechanisms.
- **Model-based Approach:** Utilizes XGBoost, a decision tree-based ensemble Machine Learning algorithm, to predict ratings. It considers various features from users and businesses, such as average stars, review counts, and additional attributes like "HasTV", "NoiseLevel", etc.
  
The final rating prediction is a weighted average of the outputs from these two components.

## Error Distribution and Performance

### Error Distribution:
- >=0 and <1: 102657
- >=1 and <2: 32454
- >=2 and <3: 6135
- >=3 and <4: 797
- >=4: 1

### RMSE:
- 0.9761563231043705

### Execution Time:
- 306.16 seconds

## Datasets

The Yelp Open Dataset is utilized, comprising reviews, businesses, users, and additional metadata across several metropolitan areas.

## Requirements

- Python 3.x
- Apache Spark
- NumPy
- scikit-learn
- XGBoost

## Running the Code

To execute the hybrid recommendation system, use the following command:

```console
spark-submit HybridRecommendationSystem.py <folder_path> <test_file_name> <output_file_name>
```

### Parameters:

- `folder_path`: Path to the dataset folder containing Yelp data (JSON files).
- `test_file_name`: Path to the testing file (e.g., yelp_val.csv).
- `output_file_name`: Path to save the prediction results (CSV format).

## Output Format

The output is a CSV file with the header "user_id, business_id, prediction", listing the predicted ratings for each user-business pair in the test dataset.
