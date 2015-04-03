import numpy as np
from StringIO import StringIO
from pprint import pprint
import argparse
from matplotlib import pyplot as plt
from collections import Counter


from sklearn.decomposition.pca import PCA as PCA
from sklearn.decomposition import FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as RandomProjection
from sklearn.cluster import KMeans as KM
from sklearn.mixture import GMM as EM
from sklearn.feature_selection import SelectKBest as best
from sklearn.feature_selection import chi2


from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# first map things to things
def create_mapper(l):
    return {l[n] : n for n in xrange(len(l))}

workclass = create_mapper(["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
education = create_mapper(["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
marriage = create_mapper(["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
occupation = create_mapper(["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
relationship = create_mapper(["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = create_mapper(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = create_mapper(["Female", "Male"])
country = create_mapper(["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])
income = create_mapper(["<=50K", ">50K"])

adultDataSetConverters = {
    1: lambda x: workclass[x],
    3: lambda x: education[x],
    5: lambda x: marriage[x],
    6: lambda x: occupation[x],
    7: lambda x: relationship[x],
    8: lambda x: race[x],
    9: lambda x: sex[x],
    13: lambda x: country[x],
    14: lambda x: income[x]
}

hillsDataSetConverters = {}

converters = {"adult": adultDataSetConverters, "hill": hillsDataSetConverters}


def load(filename, converter):
    with open(filename) as data:
        instances = [line for line in data if "?" not in line]  # remove lines with unknown data

    return np.loadtxt(instances,
                      delimiter=',',
                      converters=converter,
                      dtype='u4',
                      skiprows=1
                      )

def create_dataset(name, test, train):
    training_set = load(train, converters[name])
    testing_set = load(test, converters[name])
    train_x, train_y = np.hsplit(training_set, [training_set[0].size-1])
    test_x, test_y = np.hsplit(testing_set, [testing_set[0].size-1])
    # this splits the dataset on the last instance, so your label must
    # be the last instance in the dataset
    return train_x, train_y, test_x, test_y


def plot(axes, values, x_label, y_label, title, name):
    plt.clf()
    plt.plot(*values)
    plt.axis(axes)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(name+".png", dpi=500)
    #plt.show()
    plt.clf()


def pca(tx, ty, rx, ry):
    compressor = PCA(n_components = tx[1].size/2)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wPCAtr", times=10)
    km(newtx, ty, newrx, ry, add="wPCAtr", times=10)
    nn(newtx, ty, newrx, ry, add="wPCAr")


def ica(tx, ty, rx, ry):
    compressor = ICA(whiten=True)  # for some people, whiten needs to be off
    newtx = compressor.fit_transform(tx)
    newrx = compressor.fit_transform(rx)
    em(newtx, ty, newrx, ry, add="wICAtr", times=10)
    km(newtx, ty, newrx, ry, add="wICAtr", times=10)
    nn(newtx, ty, newrx, ry, add="wICAtr")


def randproj(tx, ty, rx, ry):
    compressor = RandomProjection(tx[1].size)
    newtx = compressor.fit_transform(tx)
    compressor = RandomProjection(tx[1].size)
    newrx = compressor.fit_transform(rx)
    em(newtx, ty, newrx, ry, add="wRPtr", times=10)
    km(newtx, ty, newrx, ry, add="wRPtr", times=10)
    nn(newtx, ty, newrx, ry, add="wRPtr")


def kbest(tx, ty, rx, ry):
    compressor = best(chi2)
    newtx = compressor.fit_transform(tx, ty)
    newrx = compressor.fit_transform(rx, ry)
    em(newtx, ty, newrx, ry, add="wKBtr", times=10)
    km(newtx, ty, newrx, ry, add="wKBtr", times=10)
    nn(newtx, ty, newrx, ry, add="wKBtr")


def em(tx, ty, rx, ry, add="", times=5):
    errs = []

    # this is what we will compare to
    checker = EM(n_components=2)
    checker.fit(ry)
    truth = checker.predict(ry)

    # so we do this a bunch of times
    for i in range(2,times):
        clusters = {x:[] for x in range(i)}

        # create a clusterer
        clf = EM(n_components=i)
        clf.fit(tx)  #fit it to our data
        test = clf.predict(tx)
        result = clf.predict(rx)  # and test it on the testing set

        # here we make the arguably awful assumption that for a given cluster,
        # all values in tha cluster "should" in a perfect world, belong in one
        # class or the other, meaning that say, cluster "3" should really be
        # all 0s in our truth, or all 1s there
        # 
        # So clusters is a dict of lists, where each list contains all items
        # in a single cluster
        for index, val in enumerate(result):
            clusters[val].append(index)

        # then we take each cluster, find the sum of that clusters counterparts
        # in our "truth" and round that to find out if that cluster should be
        # a 1 or a 0
        mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}

        # the processed list holds the results of this, so if cluster 3 was
        # found to be of value 1, 
        # for each value in clusters[3], processed[value] == 1 would hold
        processed = [mapper[val] for val in result]
        errs.append(sum((processed-truth)**2) / float(len(ry)))
    plot([0, times, min(errs)-.1, max(errs)+.1],[range(2, times), errs, "ro"], "Number of Clusters", "Error Rate", "Expectation Maximization Error", "EM"+add)

    # dank magic, wrap an array cuz reasons
    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    newtx = np.append(tx, td, 1)
    newrx = np.append(rx, rd, 1)
    nn(newtx, ty, newrx, ry, add="onEM"+add)    



def km(tx, ty, rx, ry, add="", times=5):
    #this does the exact same thing as the above
    errs = []

    checker = KM(n_clusters=2)
    checker.fit(ry)
    truth = checker.predict(ry)

    # so we do this a bunch of times
    for i in range(2,times):
        clusters = {x:[] for x in range(i)}
        clf = KM(n_clusters=i)
        test = clf.predict(tx)
        clf.fit(tx)  #fit it to our data
        result = clf.predict(rx)  # and test it on the testing set
        for index, val in enumerate(result):
            clusters[val].append(index)
        mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}
        processed = [mapper[val] for val in result]
        errs.append(sum((processed-truth)**2) / float(len(ry)))
    plot([0, times, min(errs)-.1, max(errs)+.1],[range(2, times), errs, "ro"], "Number of Clusters", "Error Rate", "KMeans clustering error", "KM"+add)

    newtx = np.append(tx, [test], 1)
    newrx = np.append(rx, [result], 1)
    nn(newtx, ty, newrx, ry, add="onEM"+add)  


def nn(tx, ty, rx, ry, add="", iterations=250):
    """
    trains and plots a neural network on the data we have
    """
    resultst = []
    resultsr = []
    positions = range(iterations)
    network = buildNetwork(tx[1].size, 5, 1, bias=True)
    ds = ClassificationDataSet(tx[1].size, 1)
    for i in xrange(len(tx)):
        ds.addSample(tx[i], [ty[i]])
    trainer = BackpropTrainer(network, ds, learningrate=0.01)
    for i in positions:
        trainer.train()
        resultst.append(sum((np.array([round(network.activate(test)) for test in tx]) - ty)**2)/float(len(ty)))
        resultsr.append(sum((np.array([round(network.activate(test)) for test in rx]) - ry)**2)/float(len(ry)))
        print i
    plot([0, iterations, 0, 1], (positions, resultst, "ro", positions, resultsr, "bo"), "Network Epoch", "Percent Error", "Neural Network Error", "NN"+add)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run clustering algorithms on stuff')
    parser.add_argument("name")
    args = parser.parse_args()
    name = args.name
    test = name+".data"
    train = name+".test"
    train_x, train_y, test_x, test_y = create_dataset(name, test, train)
    em(train_x, train_y, test_x, test_y, times = 10)
    #km(train_x, train_y, test_x, test_y, times = 10)
    #pca(train_x, train_y, test_x, test_y)
    #ica(train_x, train_y, test_x, test_y)
    #randproj(train_x, train_y, test_x, test_y)
    #kbest(train_x, train_y, test_x, test_y)