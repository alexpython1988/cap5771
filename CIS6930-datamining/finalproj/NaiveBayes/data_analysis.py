from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from naive_bayes import MyGaussianNB, CategoricalNB, train_chain
from utils import count_file_lines
import numpy as np
import math
from functools import reduce

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
    print("Homemade-wheel Mixed Naive Bayes mode prediction accuracy result: {}".format(acc))


def on_large_sample(data_file):
    #tot: 9000000 lines of data
    #take 100000 as individual data set with 0.8 : 0.2
    #run test on every individual set record all the results
    #save the last
    #TODO explore paralle training

    tot_size = count_file_lines(data_file)
    global_test_set = []
    predict_acc_res_cv = []
    predict_acc_res_g = []

    clf = GaussianNB()

    with open(data_file, "r") as f:
        for i, line in enumerate(f):
            labels = []
            sets = []
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
                                pass
                sets.append(s)

                if i % 300000 == 0:
                    #create train and test data set from labels and sets

                    clf.partial_fit()
                    clf.predict()

                    #reset the data for partial training
                    labels = []
                    sets = []

def comparsion_sklearn_myimplementation(train_set, train_label, test_set, test_label):
    #gaussian from sciki-learn
    clf = GaussianNB()
    clf.fit(train_set, train_label)
    matched = reduce(lambda x, y: x + y,
                     map(lambda x: 1 if x[0] == x[1] else 0, zip(clf.predict(test_set), test_label)))
    tot = len(test_label)
    print("Sciki-Learn Gaussian Naive Bayes mode prediction accuracy result: {:.3f}".format(matched/tot))

    #gaussian from mine
    mclf = MyGaussianNB()
    mclf.train(train_set, train_label)
    l, p = mclf.predict(test_set)
    matched = reduce(lambda x, y: x + y,
                     map(lambda x: 1 if x[0] == x[1] else 0, zip(l, test_label)))
    tot = len(test_label)
    print("Homemade-wheel Gaussian Naive Bayes mode prediction accuracy result: {:.3f}".format(matched/tot))


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
    # train_set, train_label, test_set, test_label = data_preprocess(data_file, 0.8)
    # #print(test_set)
    # # print(test_label)
    # comparsion_sklearn_myimplementation(train_set, train_label, test_set, test_label)

    ## ********************************************************************************************************
    # train_set_mix, train_label_mix, test_set_mix, test_label_mix = data_preprocess(data_file, 0.8, True)
    # self_made_wheel_mix_mode(train_set_mix, train_label_mix, test_set_mix, test_label_mix)

    ##********************************************************************************************************
    on_large_sample(large_file)


if __name__ == '__main__':
    main()