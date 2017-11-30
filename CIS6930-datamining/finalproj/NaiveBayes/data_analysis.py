from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from naive_bayes import MyGaussianNB, CategoricalNB, train_chain
from utils import count_file_lines
import numpy as np
import math
from functools import reduce
import linecache
import sys
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

def self_made_wheel_mix_mode(train_set, train_label, test_set, test_label):
    '''
    Format the data to
        {
            'd': [(1,2,3..), (1,2,3..)]
            'c': same as d
            'l': [labels]
        }
    '''
    #print(train_set[:5])
    #print(test_set[:5])
    trd = dict()
    trd['d'] = []
    trd['c'] = []
    trd['l'] = train_label


    tsd = dict()
    tsd['d'] = []
    tsd['c'] = []
    tsd['l'] = test_label

    for each in train_set:
        unit_c = []
        unit_d = []
        for v in each:
            vv = str(v)
            if vv.isdigit():
                unit_c.append(int(vv))
            else:
                try:
                    unit_c.append(float(vv))
                except:
                    unit_d.append(vv)
        trd['d'].append(tuple(unit_d))
        trd['c'].append(tuple(unit_c))

    for each in test_set:
        unit_c = []
        unit_d = []
        for v in each:
            vv = str(v)
            if vv.isdigit():
                unit_c.append(int(vv))
            else:
                try:
                    unit_c.append(float(vv))
                except:
                    unit_d.append(vv)
        tsd['d'].append(tuple(unit_d))
        tsd['c'].append(tuple(unit_c))

    #map categorical data from character to level as digit
    # trd['d'] = categorical2level(trd['d'])
    # tsd['d'] = categorical2level(tsd['d'])
    # print(trd)
    # cmodel = CategoricalNB()
    # cmodel.train(trd['d'], trd['l'])
    # l, p1 = cmodel.predict(tsd['d'])

    acc = train_chain(trd, tsd)
    return acc


def on_large_sample(data_file):
    #tot: 9000000 lines of data
    #take 100000 as individual data set with 0.8 : 0.2
    #run test on every individual set record all the results
    #TODO explore paralle training
    print("***************************************************************************************")
    print("start...")
    print()
    ratio = 0.8

    tot_size = count_file_lines(data_file)
    #left_data = tot_size % 300000
    global_test_set = dict()
    global_test_set['data'] = []
    global_test_set['label'] = []

    while tot_size % 9000000 != 0:
        #print(tot_size)
        line = linecache.getline(data_file, tot_size)
        tot_size -= 1
        sdata = line[:-1].split(",")
        if "" in sdata:
            continue
        global_test_set['label'].append(sdata[13])
        s = []
        for i, each in enumerate(sdata):
            if i != 0 and i != 16 and i != 13:
                if each.isdigit():
                    s.append(int(each))
                else:
                    try:
                        s.append(float(each))
                    except:
                        pass
        global_test_set['data'].append(tuple(s))

    predict_acc_res_cv = []
    predict_acc_res_g = []

    clf = GaussianNB()

    rounds = 1

    labelss = []
    setss = []

    with open(data_file, "r") as f:
        for i, line in enumerate(f):
            sdata = line[:-1].split(",")
            if i == 0 or "" in sdata:
                pass
            else:
                labelss.append(sdata[13])
                s = []
                for j, each in enumerate(sdata):
                    if j != 0 and j != 16 and j != 13:
                        if each.isdigit():
                            s.append(int(each))
                        else:
                            try:
                                s.append(float(each))
                            except:
                                pass
                setss.append(s)

            if i != 0 and i % 300000 == 0:
                print("round {}".format(rounds))
                print("***************************************************************************************")
                #create train and test data set from labels and sets

                tot = list(zip(setss, labelss))
                np.random.shuffle(tot)
                sets = []
                labels = []
                for each in tot:
                    sets.append(each[0])
                    labels.append(each[1])

                bisect_num = math.ceil(len(sets) * ratio)

                classes = ["NO VALUES AVAILABLE", "PUBLIC/SEMI-PUBLIC", "INSTITUTIONAL", "RETAIL/OFFICE", "VACANT NONRESIDENTIAL",
                           "INDUSTRIAL", "CENTRALLY ASSESSED", "OTHER", "ACREAGE NOT ZONED FOR AGRICULTURE", "RECREATION",
                           "VACANT RESIDENTIAL", "WATER", "MINING", "AGRICULTURE", "RESIDENTIAL", "ROW"]

                clf = clf.partial_fit(sets[:bisect_num], labels[:bisect_num], classes)

                matched = reduce(lambda x, y: x + y,
                                 map(lambda x: 1 if x[0] == x[1] else 0, zip(clf.predict(sets[bisect_num:]), labels[bisect_num:])))
                tot = len(labels[bisect_num:])
                res_cv = matched / tot

                print("Sciki-Learn Gaussian Naive Bayes mode prediction cross validation accuracy result for round {}: {:.3f}".format(rounds, res_cv))

                matched1 = reduce(lambda x, y: x + y,
                                 map(lambda x: 1 if x[0] == x[1] else 0,
                                     zip(clf.predict(np.asarray(global_test_set['data'])),
                                         np.asarray(global_test_set['label']))))
                tot1 = len(global_test_set['label'])
                res_gl =  matched1 / tot1
                print()
                print("Sciki-Learn Gaussian Naive Bayes mode prediction global accuracy result for round {}: {:.3f}".format(rounds, res_gl))
                print("***************************************************************************************")
                print()
                #reset the data for partial training
                labels = []
                sets = []

                rounds += 1
                predict_acc_res_cv.append((rounds, res_cv))
                predict_acc_res_g.append((rounds, res_gl))
                indicator = False

    matched1 = reduce(lambda x, y: x + y,
                      map(lambda x: 1 if x[0] == x[1] else 0,
                          zip(clf.predict(np.asarray(global_test_set['data'])),
                              np.asarray(global_test_set['label']))))
    tot1 = len(global_test_set['label'])
    res_gl = matched1 / tot1
    print("######################################################################################")
    print("######################################################################################")
    print()
    print("After train on the data set, the prediction accuracy on global test data set is {:.3f}".format(res_gl))
    print()

    return predict_acc_res_cv, predict_acc_res_g


def comparsion_sklearn_myimplementation(train_set, train_label, test_set, test_label):
    sklg = []
    mg = []
    for k in range(5):
        #gaussian from sciki-learn
        clf = GaussianNB()
        clf.fit(train_set, train_label)
        matched = reduce(lambda x, y: x + y,
                         map(lambda x: 1 if x[0] == x[1] else 0, zip(clf.predict(test_set), test_label)))
        tot = len(test_label)
        sklg.append(matched/tot)

        #gaussian from mine
        mclf = MyGaussianNB()
        mclf.train(train_set, train_label)
        l, p = mclf.predict(test_set)
        matched = reduce(lambda x, y: x + y,
                         map(lambda x: 1 if x[0] == x[1] else 0, zip(l, test_label)))
        tot = len(test_label)
        mg.append(matched/tot)
    return sum(sklg)/float(len(sklg)), sum(mg)/float(len(mg))

def data_preprocess(data_file, ratio, mixed=False):
    '''
    set: [[d1],[d2],[d3],[d4]...] to np.asarray
    label:[l1,l2,l3] to np.asarray
    '''
    sets = []
    labels = []
    with open(data_file, "r") as fr:
        for i, line in enumerate(fr):
            if i == 0 or "NA" in line:
                continue
            else:
                sdata = line[:-1].split(",")
                labels.append(sdata[-1])
                s = []
                for i, each in enumerate(sdata):
                    if i != 0 and i != len(sdata) - 1:
                        if each.isdigit():
                            s.append(int(each))
                        else:
                            try:
                                s.append(float(each))
                            except:
                                if mixed:
                                    s.append(each)
                                else:
                                    pass
                sets.append(s)

    #shuffle
    tot = list(zip(sets, labels))
    np.random.shuffle(tot)
    sets = []
    labels = []
    for each in tot:
        sets.append(each[0])
        labels.append(each[1])

    #split data and labels into train and test sets
    tot = len(sets)
    bisect_num = math.ceil(tot * ratio)
    return np.asarray(sets[:bisect_num]), np.asarray(labels[:bisect_num]), np.asarray(sets[bisect_num:]), np.asarray(labels[bisect_num:])

def main():
    data_file = "SampleData.csv"
    large_file = "ProjectData.txt"

    ##********************************************************************************************************
    print("######################## Comparision Gaussian Naive Bayes with My implementation  ###########################")

    train_set, train_label, test_set, test_label = data_preprocess(data_file, 0.8)
    r1, r2 = comparsion_sklearn_myimplementation(train_set, train_label, test_set, test_label)
    print("Sciki-Learn Gaussian Naive Bayes mode prediction accuracy result (average of 5 times run): {:.3f}".format(r1))
    print("Homemade-wheel Gaussian Naive Bayes mode prediction accuracy result (average of 5 times run): {:.3f}".format(r2))

    print("#############################################################################################################")
    print()
    print()
    print()

    ## ********************************************************************************************************
    print("############### my unique implementation allow train on both continuous and categorical data ################")

    train_set_mix, train_label_mix, test_set_mix, test_label_mix = data_preprocess(data_file, 0.8, True)
    rs = []
    for k in range(5):
        r = self_made_wheel_mix_mode(train_set_mix, train_label_mix, test_set_mix, test_label_mix)
        rs.append(r)
    print("Homemade-wheel Mixed Naive Bayes mode prediction accuracy result: {}".format(sum(rs)/float(len(rs))))

    print("#############################################################################################################")
    print()
    print()
    print()

    ##********************************************************************************************************
    print("####################################### train the whole data set ############################################")
    X1, X2 = on_large_sample(large_file)
    x1 = [x for x, y in X1]
    y1 = [y for x, y in X1]

    x2 = [x for x, y in X2]
    y2 = [y for x, y in X2]

    plt.xlabel("rounds")
    plt.ylabel("accuracy")

    plt.figure(0)
    plt.plot(x1, y1)
    plt.savefig("cross_validation.png")

    plt.figure(1)
    plt.plot(x2, y2)
    plt.savefig("global_test.png")
    print(
        "#############################################################################################################")

if __name__ == '__main__':
    main()