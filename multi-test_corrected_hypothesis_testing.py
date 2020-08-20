import json
import re
import sys
import numpy as np
from scipy.stats import t
from pyspark import SparkConf
from pyspark.context import SparkContext
#sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
#sc = SparkContext

file = sys.argv[1]
#file = '/user/gramajayam/Software_5.json'

conf = SparkConf().setAppName('Q1')
sc = SparkContext(conf=conf)

import time
start = time.time()

reviews = sc.textFile(file)

def splitter(input_review):
    review = json.loads(input_review)
    if 'reviewText' in review:
        reviewText = review['reviewText'].lower()
        return re.findall(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', reviewText)
    else:
        return []

rdd1 = reviews.flatMap(splitter).map(lambda x : (x, 1))
rdd2 = rdd1.reduceByKey(lambda x,y : x+y).sortBy(lambda x : -x[1])
temp1 = rdd2.take(1000)

print("Sorting 1000 words done")

global_words_1000 = []
for word in temp1:
    global_words_1000.append(word[0])
global_words_1000 = sc.broadcast(global_words_1000)

print("Broadcasting 1000 words done")

def words_1000_count(input_review):
    review = json.loads(input_review)
    overall = review['overall']
    final = []
    verified = review['verified']
    #verified = 1 if input_review['verified'] == True else 0
    if 'reviewText' in review:
        reviewText = review['reviewText'].lower()
        words = re.findall(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', reviewText)
        word_length = len(words)
        word_length = word_length if word_length else 0.01    
        for word in global_words_1000.value:
            freq = words.count(word)/word_length
            final.append((word, [freq, int(verified), overall]))
    else:
        for word in global_words_1000.value:
            final.append((word, [0.0,  int(verified), overall]))
    return final

temp2 = reviews.flatMap(words_1000_count)

print("Calculating 1000 words frequency done")

def hypo_test(x_mat, y_vec, second_mom_sum):
    #betas = np.dot(xt_x_inv_x, y) Doing the same set operations using pseudo-inverse
    betas = np.dot(np.linalg.pinv(x_mat), y_vec)
    y_vec_pred = np.dot(x_mat, betas.transpose())
    residual_sum = np.sum(np.square(y_vec - y_vec_pred))
    N = len(x_mat)
    m = len(x_mat[0])
    deg_freedom = N - (m + 1)
    denom = residual_sum/deg_freedom
    b0 = betas[0]
    t_stat = b0/(np.sqrt(denom/(second_mom_sum)))
    p_calc = t.sf(np.abs(t_stat), deg_freedom)*2000
                       
    return (b0, p_calc)                       

def words_1000_lr(key_value):
    word, values = key_value                     
    x_mat_val = []
    y_vec = []
    for score in values:
        x_mat_val.append([score[0], score[1]])
        y_vec.append(score[2])
    x_mat_val = np.array(x_mat_val)
    y_vec = np.array(y_vec)    
    x_mat_val  = (x_mat_val - x_mat_val.mean(axis = 0))/x_mat_val.std(axis = 0) #normalizing
    y_vec = (y_vec - y_vec.mean())/y_vec.std() #normalizing
    x_mat_val = np.append(x_mat_val, np.zeros([len(x_mat_val), 1]), 1)
    second_mom_sum = np.square(x_mat_val[:,0]).sum()    
    without_cont = hypo_test(x_mat_val[:,[0,2]], y_vec, second_mom_sum)
    with_cont = hypo_test(x_mat_val, y_vec, second_mom_sum)
    
    return (word, without_cont, with_cont)

temp3 = temp2.groupByKey().map(words_1000_lr) #hypo testing
list_elements = temp3.collect()
list_elements_cont = list_elements

print("Linear Regression for 1000 words done")

list_elements.sort(reverse = True, key = lambda x: x[1][0])
print("Top 20 Words without control")
top20 = list_elements[0:20]
for row in top20:
    print(row[0], row[1])
print("\n")
print("Bottom 20 Words without control")
bottom20 = list_elements[-20:]
bottom20.reverse()
for row in bottom20:
    print("Word:", row[0],",Beta value:", row[1][0], ",p-value:", row[1][1])

print("\n")

#temp4 = temp2.groupByKey().map(words_1000_lr_control)
#list_elements_cont = temp4.collect()

list_elements_cont.sort(reverse = True, key = lambda x: x[2][0])

print("Top 20 Words with control")
top20_cont = list_elements_cont[0:20]
for row in top20_cont:
    print(row[0], row[2])
print("\n")
print("Bottom 20 Words with control")
bottom20_cont = list_elements_cont[-20:]
bottom20_cont.reverse()
for row in bottom20_cont:
    print("Word:", row[0],",Beta value:", row[1][0], ",p-value:", row[1][1])

print("Total Time Taken:", (time.time() - start)/60, "minutes")
sc.stop()