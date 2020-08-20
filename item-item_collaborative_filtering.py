import json
import numpy as np
from pyspark import SparkConf
from pyspark.context import SparkContext 
import sys
#sc = SparkContext
#sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

import time
start = time.time()

file = sys.argv[1]
#file = '/user/gramajayam/Software_5.json'

asin_prods_given = sys.argv[2]
asin_prods_given = asin_prods_given.strip('[]').replace('\'', '').replace(' ', '').split(',')

#asin_prods_given = ['B00EZPXYP4', 'B00CTTEKJW']

conf = SparkConf().setAppName('Q2')
sc = SparkContext(conf=conf)

reviews = sc.textFile(file)

def read_reviews(input_review):
    input_review = json.loads(input_review)
    return((input_review['asin'], input_review['reviewerID']), (int(input_review['overall']), input_review['unixReviewTime']))

rdd = reviews.map(read_reviews).groupByKey()
print("Reading Reviews")
#rdd = rdd.take(5)

def step1(key_value): #Filter to only one rating per user per item by taking their most recent rating
    for second in key_value[1]:
        rating = second[0]
        break
    return (key_value[0][0], (key_value[0][1], rating)) #(asin, (ID, rating))    

rdd = rdd.map(step1).groupByKey()
print("Filtering only one rating per user per item by taking their most recent rating ")
#rdd = rdd.take(5)

def step2(key_value): #Filter to items associated with at least 25 distinct users
    final = []
    if len(key_value[1]) >= 25:
        for second in key_value[1]:
            final.append(((second[0]), (key_value[0], second[1]))) #(ID, (asin, rating))
    return final

rdd = rdd.flatMap(step2).groupByKey() 
print("Filtering items associated with at least 25 distinct users")
#rdd = rdd.take(5)

def step3(key_value): #Filter to users associated with at least 5 distinct items
    final = []
    if len(key_value[1]) >= 5:
        for second in key_value[1]:
            final.append(((second[0]), (key_value[0], second[1]))) #(asin, (ID, rating))
            #final.append((val[0], x[0], val[1]))
    return final 

rdd = rdd.flatMap(step3).groupByKey()
print("Filtering users associated with at least 5 distinct items")
#rdd_temp = rdd.take(5)

def step4(key_value): #accumulate
    user_ID_records = []
    overall_rat_records = []
    for second in key_value[1]:
        user_ID_records.append(second[0])
        overall_rat_records.append(second[1])
    return (key_value[0], (user_ID_records, overall_rat_records))

rdd1 = rdd.map(step4)
#rdd1 = rdd1.take(5)

def norm_calc(key_value):
    ratings = key_value[1][1] - np.mean(key_value[1][1])
    norm = np.linalg.norm(ratings)
    return (key_value[0], (key_value[1][0], ratings, norm))

asin_prods = rdd1.filter(lambda x:x[0] in asin_prods_given)
asin_prods1 = asin_prods.map(norm_calc).collect()

print("Collecting products")

global_bc_products = dict()
global_norm_products = dict()

for given in asin_prods1:
    global_bc_products[given[0]] = {}
    #mean = np.mean(given[1][1])
    global_norm_products[given[0]] = given[1][2]
    for num in range(len(given[1][0])):
    #ip_item, id, rate = item
    #global_bc_products[ip_item][id] = rate
        global_bc_products[given[0]][given[1][0][num]] = given[1][1][num] 
        
global_bc_products = sc.broadcast(global_bc_products)
global_norm_products = sc.broadcast(global_norm_products)
global_norm_products.value #dict[prod][ID] =  rating

print("Broadcasting products")

def similarity(review_data): #(item, ([users_records], [ratings_records]))
    given =  global_bc_products.value #.items()) #dict[ip_item][user] =  rating
    #asin, id, rate = review_data
    final = []
    for prod in given:
        common = 0
        if prod == review_data[0]:
            continue
        for ID in review_data[1][0]:
            if ID in given[prod]:
                common += 1
                if common == 2:
                    break
                   
        if common >= 2:
            rat_records = np.array(review_data[1][1]) - np.mean(review_data[1][1])
            #ratings_mean = np.mean(ratings)#np.sum(ratings)/np.count_nonzero(ratings)           
            
            cos_inter1 = []
            cos_inter2 = []
            for num in range(len(review_data[1][0])):
                try:
                    cos_inter1.append(given[prod][review_data[1][0][num]])
                    cos_inter2.append(rat_records[num])
                except:
                    continue
            cos_sim = np.dot(cos_inter1, cos_inter2)/(global_norm_products.value[prod]*np.linalg.norm(rat_records))
            if cos_sim > 0:
                #final.append((prod, com_cos_inter1, len(cos_inter1), ratings, len(cos_inter2)))
                final.append((prod, (review_data[0], cos_sim))) #, review_data[1][0], review_data[1][1], sim)))
                #final.append((prod, (len(cos_inter1), len(cos_inter2), len(com_cos_inter1), len(ratings))))
    return final         

temp_rdd1 = rdd1.flatMap(similarity).groupByKey()

print("Calculating Cosine Similarity")

def neigh_find(review_data): #Sorting and collecting top 50 neighbours
    val = [y for y in review_data[1]]
    return (review_data[0], sorted(val, key = lambda review_data: review_data[1], reverse = True))[0:50]

temp_rdd2 = temp_rdd1.map(neigh_find).collect()

print("Sorting and collecting top 50 neighbours")

global_neigh = dict()
for key_pair in temp_rdd2:
    for prod, cosine in key_pair[1]:
        try:
            global_neigh[prod].append((key_pair[0], cosine))
        except:
            global_neigh[prod] = [(key_pair[0], cosine)]
global_neigh = sc.broadcast(global_neigh)
len(global_neigh.value)

def users_pred(review_data): #(item, (ID, sim))
    final = []
    #value = sorted(x[1], key = lambda z : z[1]) #(target, sim)
    
    for (prod, cosine) in global_neigh.value[review_data[0]]:
        for num in range(len(review_data[1][0])):
            final.append(((review_data[1][0][num], prod), (review_data[1][1][num], cosine)))
    return final        

answer = rdd1.filter(lambda x: x[0] in global_neigh.value).flatMap(users_pred).groupByKey()

def recom_pred(review_data):
    denominator = 0
    numerator = 0
    for pairs in review_data[1]:
        numerator += pairs[0]*pairs[1]
        denominator += pairs[1]
    rating = numerator/denominator
    return (review_data[0][1], (review_data[0][0], rating))

answer = answer.filter(lambda x: len(x[1]) >= 2 and x[0][0] not in global_bc_products.value[x[0][1]])
answer = answer.map(recom_pred).groupByKey().collect()

print("Collecting final answer")

print(answer[0][0], len(answer[0][1]))
for i in list(answer[0][1]):
    print(i)
print("\n")    
print(answer[1][0], len(answer[1][1]))
for i in list(answer[1][1]):
    print(i)
    
print("Total Time Taken:", (time.time() - start)/60, "minutes")   
sc.stop()
