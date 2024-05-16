# Hybrid Recommender System for Yelp Ratings

## Description

This project develops a hybrid recommendation system to predict Yelp business ratings based on user reviews, integrating item-based collaborative filtering (CF) and a model-based approach using XGBoost within the Apache Spark environment. This combination aims for higher accuracy by leveraging the strengths of both approaches to provide accurate rating predictions.  

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

## Dataset

The Yelp Open Dataset is used in this project, providing a rich set of data including reviews, businesses, users, and more, suitable for developing and testing recommendation systems. To obtain the dataset:

1. Go to the Yelp Open Dataset page at https://www.yelp.com/dataset.
2. Follow the instructions for accessing the data. You may need to agree to certain terms of use and create an account if you haven't already.
3. Download the dataset files. The project specifically requires yelp_academic_dataset_business.json, yelp_academic_dataset_review.json, yelp_academic_dataset_user.json, yelp_academic_dataset_checkin.json, and yelp_academic_dataset_tip.json.
4. Place the downloaded files in a directory accessible to the script, as specified by the <folder_path> argument when running the recommendation system.

## Requirements

- Python 3.x
- Apache Spark
- NumPy
- scikit-learn
- XGBoost

## Running the Code

To execute the hybrid recommendation system, use the following command:

```console
spark-submit HybridYelpRecommender.py <folder_path> <test_file_name> <output_file_name>
```

### Parameters:

- `folder_path`: Path to the dataset folder containing Yelp data (JSON files).
- `test_file_name`: Path to the testing file (e.g., yelp_val.csv).
- `output_file_name`: Path to save the prediction results (CSV format).

## Output Format

The output is a CSV file with the header "user_id, business_id, prediction", listing the predicted ratings for each user-business pair in the test dataset.
