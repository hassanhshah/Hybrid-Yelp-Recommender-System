"""
Method Description:
    This hybrid recommendation model predicts Yelp Ratings of Businesses by User's Reviews.

Error Distribution:
    >=0 and <1: 102657
    >=1 and <2: 32454
    >=2 and <3: 6135
    >=3 and <4: 797
    # >=4: 1

RMSE:
    0.9761563231043705
    
Execution Time:
    306.16
"""

import json
import numpy as np
import sys
import time
from math import sqrt
from operator import add
from pyspark import SparkContext
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def get_HasTV(data):
    if data:
        if "HasTV" in data.keys():
            if data["HasTV"] == 'True':
                return 1
            elif data["HasTV"] == 'False':
                return -1
    return 0
    
def get_NoiseLevel(data):
    if data:
        if "NoiseLevel" in data.keys():
            if data["NoiseLevel"] == 'quiet':
                return 1
            elif data["NoiseLevel"] == 'average':
                return 2
            elif data["NoiseLevel"] == 'loud':
                return 3
            elif data["NoiseLevel"] == 'very_loud':
                return 4
    return 2

def get_RestaurantsGoodForGroups(data):
    if data:
        if "RestaurantsGoodForGroups" in data.keys():
            if data["RestaurantsGoodForGroups"] == 'True':
                return 1
            elif data["RestaurantsGoodForGroups"] == 'False':
                return -1
    return 0

def get_RestaurantsPriceRange2(data):
    if data:
        if "RestaurantsPriceRange2" in data.keys():
            if data["RestaurantsPriceRange2"] == '1':
                return 1
            elif data["RestaurantsPriceRange2"] == '2':
                return 2
            elif data["RestaurantsPriceRange2"] == '3':
                return 3
            elif data["RestaurantsPriceRange2"] == '4':
                return 4
    return 2

def get_WiFi(data):
    if data:
        if "WiFi" in data.keys():
            if data["WiFi"] == 'free':
                return 1
            elif data["WiFi"] == 'no':
                return -1
    return 0

def abs_dif(label, predictions):
    result = {'0-1': 0, '1-2':0, '2-3':0, '3-4':0, '>4':0}
    for i in range(0, len(label)):
        abs_diff = abs(label[i]-predictions[i])
        if abs_diff <= 1:
            result['0-1'] += 1
        elif abs_diff <= 2:
            result['1-2'] += 1
        elif abs_diff <= 3:
            result['2-3'] += 1
        elif abs_diff <= 4:
            result['3-4'] += 1
        else:
            result['>4'] += 1
    return result

# Item-based approach
def item_based_pred(bus, user, user_bus_dict, bus_user_dict, user_avg_dict, bus_avg_dict, bus_user_rate_dict, weight_dict):
    # Checking if the business has ever been rated by the user, if not return 3 as default rating
    if bus not in bus_user_dict:
        return user_avg_dict[user]
    if user not in user_bus_dict:
        return 3.0 # default rating
    
    weight_list = [] # will hold the different weights
    for bus_id in user_bus_dict[user]: # for businesses the user has rated
        temp = tuple(sorted((bus_id, bus)))
        weight = weight_dict.get(temp)
        if weight is None: # i.e. if corrated
            user_inter = bus_user_dict[bus] & bus_user_dict[bus_id] # intersection
            num_corrated = len(user_inter) # however many businesses in common
            if num_corrated <= 1: # only one in common
                weight = (5.0 - abs(bus_avg_dict[bus] - bus_avg_dict[bus_id])) / 5
            elif num_corrated == 2: # if two in common
                user_inter = list(user_inter)
                weight_1 = (5.0 - abs(float(bus_user_rate_dict[bus][user_inter[0]]) - float(bus_user_rate_dict[bus_id][user_inter[0]]))) / 5
                weight_2 = (5.0 - abs(float(bus_user_rate_dict[bus][user_inter[1]]) - float(bus_user_rate_dict[bus_id][user_inter[1]]))) / 5
                weight = (weight_1 + weight_2) / 2
            else: # if more in common
                rating_1 = [float(bus_user_rate_dict[bus][user]) for user in user_inter] 
                rating_2 = [float(bus_user_rate_dict[bus_id][user]) for user in user_inter]
                avg_1, avg_2 = sum(rating_1) / num_corrated, sum(rating_2) / num_corrated # averages
                norm_1, norm_2 = [x - avg_1 for x in rating_1], [x - avg_2 for x in rating_2] # ratings normalized
                num, den = sum(x * y for x, y in zip(norm_1, norm_2)), sum(x ** 2 for x in norm_1) ** 0.5 * sum(x ** 2 for x in norm_2) ** 0.5 
                weight = num / den if den != 0 else 0 # final calculation of the weight as long as den != 0
            weight_dict[temp] = weight
        weight_list.append((weight, float(bus_user_rate_dict[bus_id][user]))) # add weight

    weight_rate_neighbors = sorted(weight_list, key=lambda x: -x[0])[:10] # 10 closest neighbors to compute weight
    num, den = sum(w * r for w, r in weight_rate_neighbors), sum(abs(w) for w, _ in weight_rate_neighbors)

    return num / den if den != 0 else 3

# Model-based approach
def model_based_pred(folder_path, val_path):
    # Reading in train csv
    train_rdd = sc.textFile(folder_path + '/yelp_train.csv')
    train_header = train_rdd.first()
    train_rdd = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))
    
    # Reading in test csv
    test_rdd = sc.textFile(test_file_name)
    test_header = test_rdd.first()
    test_rdd = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))
    
    # Reading in user.json to get average stars given, user review count, and number of fans for prediction
    user_dict = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], (float(x['average_stars']), float(x['review_count']), float(x['fans']), float(x['useful']), float(x['funny']), float(x['cool']), float(x['compliment_hot'])+float(x['compliment_more'])+float(x['compliment_profile'])+float(x['compliment_cute'])+float(x['compliment_list'])+float(x['compliment_note'])+float(x['compliment_plain'])+float(x['compliment_cool'])+float(x['compliment_funny'])+float(x['compliment_writer'])+float(x['compliment_photos'])))).collectAsMap()
    
    # Reading in business.json to get average stars, and total review count for prediction
    bus_dict = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (float(x['stars']), float(x['review_count']), float(get_HasTV(x['attributes'])), float(get_NoiseLevel(x['attributes'])), float(get_RestaurantsGoodForGroups(x['attributes'])), float(get_RestaurantsPriceRange2(x['attributes'])), float(get_WiFi(x['attributes'])), (float(x["longitude"])+180)/360 if x["longitude"] is not None else 0.5, (float(x["latitude"])+90)/180 if x["latitude"] is not None else 0.5, float(x['is_open'])))).collectAsMap()
    
    # Additional features
    checkin_dict = sc.textFile(folder_path + '/checkin.json').map(lambda x: json.loads(x)).map(lambda x:(x['business_id'], len(x["time"]))).reduceByKey(add).collectAsMap()
        
    photo_dict = sc.textFile(folder_path + '/photo.json').map(lambda x: json.loads(x)).map(lambda x:(x['business_id'],1)).reduceByKey(add).collectAsMap()

    tip_dict = sc.textFile(folder_path + '/tip.json').map(lambda x: json.loads(x)).map(lambda x:((x['business_id'],x['user_id']), x["likes"])).reduceByKey(add).collectAsMap()

    # Generating training data
    x_train = []
    y_train = []
    
    train_data = train_rdd.collect()

    x_train = [[user_dict.get(user, (None, None, None, None, None))[0],
                user_dict.get(user, (None, None, None, None, None))[1],
                user_dict.get(user, (None, None, None, None, None))[2],
                user_dict.get(user, (None, None, None, None, None))[3],
                user_dict.get(user, (None, None, None, None, None))[4],
                user_dict.get(user, (None, None, None, None, None))[5],
                user_dict.get(user, (None, None, None, None, None))[6],
                bus_dict.get(bus, (None, None, None, None, None, None, None))[0],
                bus_dict.get(bus, (None, None, None, None, None, None, None))[1],
                bus_dict.get(bus, (None, None, None, None, None, None, None))[2],
                bus_dict.get(bus, (None, None, None, None, None, None, None))[3],
                bus_dict.get(bus, (None, None, None, None, None, None, None))[4],
                bus_dict.get(bus, (None, None, None, None, None, None, None))[5],
                bus_dict.get(bus, (None, None, None, None, None, None, None))[6],
                bus_dict.get(bus, (None, None, None, None, None, None, None))[7],
                bus_dict.get(bus, (None, None, None, None, None, None, None))[8],
                bus_dict.get(bus, (None, None, None, None, None, None, None))[9],
                checkin_dict.get(bus, None),
                photo_dict.get(bus, None),
                tip_dict.get((bus, user), None)
               ] for user, bus, _ in train_data]
    y_train = [rating for _, _, rating in train_data]
    
    x_train = np.array(x_train).astype(float)
    y_train = np.array(y_train).astype(float)

    
    # Generating test data 
    x_test = []
    y_test = []
    
    test_data = test_rdd.collect()
    
    # Extract the actual ratings (Y_test) from the test data
    y_test = [float(rating) for _, _, rating in test_data]
    y_test = np.array(y_test).astype(float)
    
    user_bus_list = []
    for row in test_data:
        user = row[0]
        bus = row[1]
        user_bus_list.append((user, bus))
        if user in user_dict.keys():
            user_avg_str = user_dict[user][0]
            user_rev_cnt = user_dict[user][1]
            user_fan = user_dict[user][2]
            user_useful = user_dict[user][3]
            user_funny = user_dict[user][4]
            user_cool = user_dict[user][5]
            user_comp = user_dict[user][6]
        else:
            user_avg_str = None
            user_rev_cnt = None
            user_fan = None
            user_useful = None
            user_funny = None
            user_cool = None
            user_comp = None
        if bus in bus_dict.keys():
            bus_avg_str = bus_dict[bus][0]
            bus_rev_cnt = bus_dict[bus][1]
            bus_noise_level = bus_dict[bus][2]
            bus_wifi = bus_dict[bus][3]
            bus_has_tv = bus_dict[bus][4]
            bus_good_for_groups = bus_dict[bus][5]
            bus_price_range = bus_dict[bus][6]
            bus_long = bus_dict[bus][7]
            bus_lat = bus_dict[bus][8]
            bus_open = bus_dict[bus][9]
        else:
            bus_avg_str = None
            bus_rev_cnt = None
            bus_noise_level = None
            bus_wifi = None
            bus_has_tv = None
            bus_good_for_groups = None
            bus_price_range = None
            bus_long = None
            bus_lat = None
            bus_open = None
        if bus in checkin_dict.keys():
            checkin_time = checkin_dict[bus]
        else:
            checkin_time = None
            
        if bus in photo_dict.keys():
            photo_num = photo_dict[bus]
        else:
            photo_num = None

        if (bus, user) in tip_dict.keys():
            tip_likes = tip_dict[(bus, user)]
        else:
            tip_likes = None

        x_test.append([user_avg_str, user_rev_cnt, user_fan, user_useful, user_funny, user_cool, user_comp,
                   bus_avg_str, bus_rev_cnt, bus_noise_level, bus_wifi, bus_has_tv, bus_good_for_groups, bus_price_range,
                   bus_long, bus_lat, bus_open, checkin_time, photo_num, tip_likes])

    x_test = np.array(x_test).astype(float)
        
    # Fitting xgb model
    xgb = XGBRegressor(eval_metric=['rmse'], learning_rate=0.07, max_depth=7, n_estimators=700, reg_lambda=0.7, reg_alpha=0.7, n_jobs=-1)
    xgb.fit(x_train, y_train)
    y_pred = xgb.predict(x_test)
    
    return user_bus_list, y_pred

# Executes from here, using base of task 2_1
if __name__ == "__main__":
    start_time = time.time()  # For duration

    # Command-line arguments
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]
    
    # Initializing Spark
    sc = SparkContext('local[*]', 'competition')
    sc.setLogLevel("ERROR")
    
    # Reading in train csv
    train_rdd = sc.textFile(folder_path + '/yelp_train.csv')
    train_header = train_rdd.first()
    train_rdd = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(',')).map(lambda x: (x[1], x[0], x[2]))
    
    # Reading in test csv
    test_rdd_start = sc.textFile(test_file_name)
    test_header = test_rdd_start.first()
    test_rdd = test_rdd_start.filter(lambda x: x != test_header).map(lambda x: x.split(',')).map(lambda x: (x[1], x[0]))
    
    test_data = test_rdd_start.filter(lambda x: x != test_header).map(lambda x: x.split(',')).map(lambda x: (x[1], x[0], x[2])).collect()
    y_test = [float(rating) for _, _, rating in test_data]
    y_test = np.array(y_test).astype(float)
    
    # Business and user dictionaries
    bus_user_dict = train_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()
    user_bus_dict = train_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()

    # Business and user average dictionaries
    bus_avg_dict = train_rdd.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(list).map(lambda x: (x[0], sum(x[1]) / len(x[1]))).collectAsMap()
    user_avg_dict = train_rdd.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(list).map(lambda x: (x[0], sum(x[1]) / len(x[1]))).collectAsMap()

    # Dictionary of businesses and the users who rated them and what rating they gave
    bus_user_rate_sets = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(set)
    bus_user_rate_dict = {}
    for bus, user_rate_set in bus_user_rate_sets.collect():
        temp = {}
        for user_rate in user_rate_set:
            temp[user_rate[0]] = user_rate[1]
        bus_user_rate_dict[bus] = temp
    weight_dict = {} # Used to hold weights later on
    
    # Item based predictions
    item_based_result = []
    for row in test_rdd.collect():
        item_pred = item_based_pred(row[0], row[1], user_bus_dict, bus_user_dict, user_avg_dict, bus_avg_dict, bus_user_rate_dict, weight_dict)
        item_based_result.append(item_pred)
    
    # Model based predictions
    user_bus_list, model_based_result = model_based_pred(folder_path, test_file_name)
    
    # Getting weight from both models
    y_pred = []
    alpha = 0.07
    result = "user_id, business_id, prediction\n"
    for row in range(0, len(model_based_result)):
        weighted_pred = float(alpha) * float(item_based_result[row]) + (1 - float(alpha)) * float(model_based_result[row])
        y_pred.append(weighted_pred)
        result += user_bus_list[row][0] + "," + user_bus_list[row][1] + "," + str(weighted_pred) + "\n"
    with open(output_file_name, "w") as csvfile:
        csvfile.writelines(result)
    
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)
    
    abs_dif_list = abs_dif(y_test, y_pred)
    print(abs_dif_list)
    
    duration = time.time() - start_time
    print("Duration: {0:.2f}".format(duration))
