'''
This is a simple version of naive bayes classifier

GaussianNB
'''

import numpy as np
import os
import json
from utils import update_mean, update_variance, evaluation

if not os.path.exists("temp"):
    os.makedirs("temp")
if os.path.isfile("temp/medium.json"):
    os.remove("temp/medium.json")

class MyGaussianNB():
    '''
    Naive Bayes classifier using Gaussian function to handle continuous variables
    '''
    def __init__(self):
        self.prior_prob = dict()
        self.variance = dict()
        self.mean = dict()
        self.size = dict()

    def _mean_variance(self, array_data):
        '''
        return mean of input data (must be array)
        :return: float
        '''
        mean = np.sum(array_data, axis=0)/len(array_data)
        var = np.sum((array_data - mean) ** 2, axis=0) / len(array_data)
        return mean, var

    def train(self, train_data, train_label, partial=False):
        '''
             give a flag named partial with default=False
             if the flag set to True
                check if there is a previously trained json
                    if not train a new model and save it
                if having a previously trained json
                    combine previous prob, mean, variance with current value
        '''
        self.train_data = np.asarray(train_data)
        self.train_label = np.asarray(train_label)
        len_train_data = len(self.train_data)
        len_train_label = len(self.train_label)
        if len_train_data != len_train_label:
            raise ValueError("Train data and train label are not the same. data size: {}; label size: {}".format(len_train_data, len_train_label))

        self.uniq_label = np.unique(self.train_label)

        for ul in self.uniq_label:
            labeled_data = self.train_data[self.train_label == ul]
            self.size[ul] = len(labeled_data)
            self.prior_prob[ul] = np.log(len(labeled_data) / float(len_train_data))
            self.mean[ul], self.variance[ul] = self._mean_variance(labeled_data)

        # save sample size, prior_prob, mean and variance for continuous training
        if partial:
            train_file = "temp/medium.json"
            if os.path.isfile(train_file):
                with open(train_file, "r") as fr:
                    previous_train_param = json.load(fr)
                self._refactor_params(previous_train_param)

            with open(train_file, "w") as fw:
                td = dict()
                td['tot'] = self.size
                td['prob'] = self.prior_prob
                td['mean'] = self.mean
                td['var'] = self.variance
                json.dump(td, fw)

        return self.prior_prob

    def _refactor_params(self, pre_params):
        pre_prob = pre_params['prob']
        pre_size = pre_params['tot']
        pre_mean = pre_params['mean']
        pre_var = pre_params['var']

        #TODO merge following code as an function
        for k, v in pre_prob.items():
            if k in self.prior_prob:
                self.prior_prob[k] = self.prior_prob[k] + v
            else:
                self.prior_prob[k] = v

        for k, v in pre_mean.items():
            if k in self.mean:
                self.mean[k] = update_mean(self.mean[k], v, self.size[k], pre_size[k])
            else:
                self.mean[k] = v

        for k, v in pre_var.items():
            if k in self.variance:
                self.variance[k] = update_variance(self.variance[k], v, self.mean[k], pre_mean[k], self.size[k], pre_size[k])
            else:
                self.variance[k] = v

        for k, v in pre_size.items():
            if k in self.size:
                self.size[k] = self.size[k] + v
            else:
                self.size[k] = v

    def _gaussian(self, val, tag, i):
        #print(self.variance)
        '''Returns conditional-probability (likelihood) using Gaussian distribution'''
        return (1 / np.sqrt(2 * np.pi * self.variance[tag][i])) * np.exp(-0.5 * (val - self.mean[tag][i]) ** 2 / self.variance[tag][i])

    def predict(self, test_data):
        test_data = np.asarray(test_data)
        pred_label = []
        gaussian_lps = []

        if len(test_data.shape) == 1 or test_data.shape[1] == 1:
            test_data = test_data.reshape(1, len(test_data))

        for i, vec in enumerate(test_data):
            gaussian_lp = {tag: 0.0 for tag in self.uniq_label}
            #print(vec)
            for j, val in enumerate(vec):
                for tag in self.uniq_label:
                    gs_prob = self._gaussian(val, tag, j)
                    if gs_prob:
                        gaussian_lp[tag] += np.log(gs_prob)
            for tag in self.uniq_label:
                gaussian_lp[tag] += self.prior_prob[tag]
            gaussian_lps.append(gaussian_lp)
            pred_label.append(max(gaussian_lp.items(), key=lambda x:x[1])[0])

        return pred_label, gaussian_lps


class CategoricalNB():
    '''
    Naive Bayes classifier that can handle categorical variables
    '''

    def train(self, X, y, partial=False):
        self.training_data = np.asarray(X)
        self.training_labels = np.asarray(y)

        unique_labels = np.unique(self.training_labels)
        unique_feats = np.unique(self.training_data)
        label_count = dict()

        self.feats_count = len(unique_feats)
        self.feat_tag_cmat = np.zeros((len(unique_labels), self.feats_count))
        self.tag_id = {tag: i for i, tag in enumerate(unique_labels)}
        self.feat_id = {feat: i for i, feat in enumerate(unique_feats)}

        for vec, lbl in zip(self.training_data, self.training_labels):
            label_count.setdefault(lbl, 0)
            label_count[lbl] += 1
            for x in vec:
                self.feat_tag_cmat[self.tag_id[lbl]][self.feat_id[x]] += 1

        self.prior_count = label_count
        self.prior_prob = {tag: np.log(label_count[tag] / float(len(self.training_labels))) for tag in unique_labels}

        return self.prior_prob

    def laplacian(self, val, tag):
        if val in self.feat_id:
            return (self.feat_tag_cmat[self.tag_id[tag]][self.feat_id[val]] + 1.0) / (self.prior_count[tag] + self.feats_count)
        else:
            return 1.0 / (self.prior_count[tag] + self.feats_count)

    def predict(self, testing_data):
        labels = []
        testing_data = np.asarray(testing_data)
        smoothed_lps = []

        if len(testing_data.shape) == 1 or testing_data.shape[1] == 1:
            testing_data = testing_data.reshape(1, len(testing_data))

        for i, vec in enumerate(testing_data):
            smoothed_lp = {tag: 0.0 for tag in self.tag_id}
            for val in vec:
                for tag in self.tag_id:
                    sl_prob = self.laplacian(val, tag)
                    smoothed_lp[tag] += np.log(sl_prob)
            for tag in self.tag_id:
                smoothed_lp[tag] += self.prior_prob[tag]
            labels.append(max(smoothed_lp.items(), key=lambda x: x[1])[0])
            smoothed_lps.append(smoothed_lp)

        return labels, smoothed_lps


#TODO need to fix how to train on single column data
def train_chain(train_set, test_set):
    '''
    the data set should be pre_shuffled and pre-processed to a format as
    {
        'd': [(1,2,3..), (1,2,3..)]
        'c': same as d
        'l': [labels]
    }

    the data in label column is in 'l'
    the columns contain continuous data are in 'c'
    the columns contain categorical data are in 'd'
    if one type of data is missing, using empty list to occupy the space => 'd':[]
    [col1] is a list of all the data

    test_set should be format in the same way
    '''
    ps = []

    gd = train_set['c']
    cd = train_set['d']
    ld = train_set['l']

    gdt = test_set['c']
    cdt = test_set['d']
    ldt = test_set['l']


    model = MyGaussianNB()
    model.train(gd, ld)
    l, p = model.predict(gdt)
    #print(p)

    cmodel = CategoricalNB()
    cmodel.train(cd, ld)
    l, p1 = cmodel.predict(cdt)
    #print(p1)

    final = []
    labels = []
    for k in range(len(p)):
        d = dict()
        m1 = p[k]
        m2 = p1[k]
        for k, v in m1.items():
            d[k] = m2[k] + v
        final.append(d)

    #print(final)
    for each in final:
        labels.append(max(each.items(), key=lambda x: x[1])[0])
    # print(labels)
    return evaluation(labels, ldt)


def test():
    from sklearn import datasets
    import numpy as np
    iris = datasets.load_iris()
    dataset = list(zip(iris.data, iris.target))
    np.random.shuffle(dataset)
    #print(dataset[:10])
    trainset = dataset[:100]
    testset = dataset[100:120]

    #col 1,3 are c and 2,4 are d
    # dim = len(trainset[0][0])

    trd = dict()
    trd['d'] = dict()
    trd['c'] = dict()
    trd['l'] = [each[1] for each in trainset]

    tsd = dict()
    tsd['d'] = dict()
    tsd['c'] = dict()
    tsd['l'] = [each[1] for each in testset]


    trd['d'] = []
    for each in trainset:
        trd['d'].append((each[0][0], each[0][2]))

    trd['c'] = []
    for each in trainset:
        trd['c'].append((each[0][1], each[0][3]))

    tsd['d'] = []
    for each in testset:
        tsd['d'].append((each[0][0], each[0][2]))

    tsd['c'] = []
    for each in testset:
        tsd['c'].append((each[0][1], each[0][3]))

    #print(trd)
    # print(tsd)
    train_chain(trd, tsd)


if __name__ == '__main__':
    test()