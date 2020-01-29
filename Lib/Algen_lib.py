from random import randint, random
import copy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize 
import numpy as np
from sklearn.naive_bayes import MultinomialNB

def feature_extraction(data):
    i_temp = list()
    for i in data:
        i = word_tokenize(i.lower())
        i_temp += i
    return sorted(set(i_temp))

def weighting(data_, metode = "tfidf", feature=None):
    feature = sorted(set(feature))
#     print(feature)
    if metode=="tfidf":
        vectorizer = TfidfVectorizer(vocabulary = feature)
        model = vectorizer.fit(data_) 
        X = model.transform(data_)
#         fitur = vectorizer.get_feature_names()
#         vectorizer = CountVectorizer(vocabulary = feature)
#         model = vectorizer.fit(data_)  
    else:
        vectorizer = CountVectorizer(vocabulary = feature)
        model = vectorizer.fit(data_)
#         fitur = vectorizer.get_feature_names()
        X = model.transform(data_)
    return {
        "weight":X,
        "features":np.array(feature),
        "model":model
    }


def create_chromosome(min, max, length):
    chromosome = []
    for _ in range(length):
        chromosome.append(randint(min, max))
    return chromosome

# def fitness(chrm, ffunc=sum):
#     return ffunc(chrm)
def get_index(data, cek=1):
    d = list()
    for ix, i in enumerate(data):
        i=int(i)
        if i == cek:
            d.append(ix)
    return d


def fitness_(X,y, X_, y_, features_bin, features, alpha = 1, metode = "tfidf"):
    features = np.array(features)
    index = get_index(features_bin)
#     print(index)
#     print("00000|",features[index])
    data = weighting(X, metode = metode, feature=features[index])
    X_training = data['weight']
    clf = MultinomialNB(alpha = 1)
    clf.fit(X_training, y)
    X_ = data['model'].transform(X_)
    return clf.score(X_,y_)

# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
def fitness_kf(X, y, binary_fitur, fitur, alpha = 1, metode = 'tfidf', K=10):
    # skf = StratifiedKFold(n_splits=K, shuffle=False)
    kf = KFold(n_splits=K)
    X = np.array(X)
    y = np.array(y)
    
    skor_all = list()
    # for train_index, test_index in skf.split(X, y):
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        skor_all.append(fitness_(X_train,y_train, X_test,y_test, binary_fitur, fitur, alpha = alpha, metode = metode))
    return sum(skor_all)/len(skor_all)
        

def Modeling(X,y, X_, y_, features_bin, features, alpha = 1, metode = "tfidf"):
    features = np.array(features)
    index = get_index(features_bin)
#     print(index)
#     print("00000|",features[index])
    data = weighting(X, metode = metode, feature=features[index])
    X_training = data['weight']
    clf = MultinomialNB(alpha = 1)
    clf.fit(X_training, y)
    X_ = data['model'].transform(X_)
    return clf.score(X_,y_)


def create_population(nelem,  length, min=0, max=1):
    return [create_chromosome(min, max, length) for i in range(nelem)]    



def get_roulette_wheel_(population, inc=0):
    roulette_wheel = []
    index = 0
    sum_pop = sum(population)
    for x in population:
        x = int(round((x/sum_pop)*(100+inc)))
        if x< 0.6:
            x = 1
        for _ in range(x):
            roulette_wheel.append(index)
        index += 1
    return roulette_wheel

import random
def select_parents(rw):
    rw_co = list(rw)
    poppp = list()
    while True:
        r1 = random.randrange(0, len(rw_co), 1)
        p1 = rw_co[r1]

    #     print(r, end=" ")
        r2 = random.randrange(0, len(rw_co), 1)
        p2 = rw_co[r2]

        if p1!=p2:
            poppp.append([p1,p2])
            rw_co = list(filter(lambda a: a != p2, rw_co))
            rw_co = list(filter(lambda a: a != p1, rw_co))

        if len(list(set(rw_co))) == 1:
            p1 = rw_co[-1]
            p2 = rw[0]
            poppp.append([p1,p2])
            return poppp
#             rw_co = list(filter(lambda a: a != p2, rw_co))
#             rw_co = list(filter(lambda a: a != p1, rw_co))

        if len(list(set(rw_co)))<=1:
            return poppp
#         return poppp



def changesBinary(d):
    for ix, i in enumerate(d):
        if i == 1:
            d[ix]=0
        else:
            d[ix]=1
    return d

def crossover(pp1_, pp2_, panjang_fitur, jumlah_titik = 7, prob_mutasi = 0.7):
    pp1 = list(pp1_)
    pp2 = list(pp2_)
    jumlah_point = jumlah_titik+1
#     prob_mutasi = 0.7
    titik = int(panjang_fitur/jumlah_point)
    cek = 0
    mulai = list()
    akhir = list()
    for i in range(panjang_fitur):
        if i % titik ==0:
            if cek%2==0:
                mulai.append(i)
            else:
                akhir.append(i)
            cek+=1
    for m, a in zip(mulai,akhir):
        pp1_Q = list(pp1[m:a])
#         print("pp1",pp1_Q)
        pp2_Q = list(pp2[m:a])
#         print("pp2",pp2_Q)

        pp1[m:a] = pp2_Q
        pp2[m:a] = pp1_Q

    if random.random() > prob_mutasi:
        pp1 = changesBinary(pp1)

    if random.random() > prob_mutasi:
        pp2 = changesBinary(pp2)


    anak_ke_3 = list()
    for a,b in zip(pp1_, pp2_):
        if a==1 or b==1:
            anak_ke_3.append(1)
        else:
            anak_ke_3.append(0)

    return [pp1, pp2, anak_ke_3]

def int_to_str(d):
    n =list()
    for i in d:
        n.append(str(i)) 
    return "".join(n)