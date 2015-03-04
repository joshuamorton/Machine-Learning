"""
This makes use of scikit-learn
dependencies:
numpy
scipy
scikit-learn
matplotlib
pybrain
etc.
"""

import numpy as np
import pydot
from sklearn import tree, neighbors, ensemble, cross_validation
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from StringIO import StringIO
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.validation import CrossValidator
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pprint import pprint

import matplotlib.pyplot as plt



"""
load the data into numpy
this section is also written for use in a
"""
train = "Hill_Valley_without_noise_Training.data"
test = "Hill_Valley_without_noise_Testing.data"


def load(filename):
    with open(filename) as data:
        hills = [line for line in data if "?" not in line]  # remove lines with unknown data

    return np.loadtxt(hills,
        delimiter=',',
        dtype='u4',
        skiprows=1
        )

def start_hills():
    """
    tx - training x axes
    ty - training y axis
    rx - result (testing) x axes
    ry - result (testing) y axis
    """
    tr = load(train)
    te = load(test)
    tx, ty = np.hsplit(tr, [100])
    rx, ry = np.hsplit(te, [100])
    ty = ty.flatten()
    ry = ry.flatten()
    return tx, ty, rx, ry 


def decisionTree(tx, ty, rx, ry, height):
    """
    """
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=height)
    clf.fit(tx, ty)
    dotdata = StringIO()
    tree.export_graphviz(clf, out_file=dotdata) 
    graph = pydot.graph_from_dot_data(dotdata.getvalue())
    graph.write_pdf("out.pdf")
    return sum((clf.predict(rx) - ry)**2)/float(len(ry))  # + cross_validation.cross_val_score(clf, tx, ty).mean()


def knn(tx, ty, rx, ry, n):
    """
    """
    neigh = neighbors.KNeighborsClassifier(n_neighbors=n)
    neigh = neigh.fit(tx, ty)
    return sum((neigh.predict(rx) - ry)**2)/float(len(ry))


def knntester(tx, ty, rx, ry, iterations):
    """
    """
    er = []
    et = []
    positions = range(1,iterations)
    for n in xrange(1,iterations):
        neigh = neighbors.KNeighborsClassifier(n_neighbors=n)
        neigh = neigh.fit(tx, ty)
        er.append(sum((neigh.predict(rx) - ry)**2)/float(len(ry)))
        et.append(sum((neigh.predict(tx) - ty)**2)/float(len(ty)))
        print n
    plt.plot(positions, et, 'ro', positions, er, 'bo')
    plt.axis([0, iterations, 0, 1])
    plt.title("Unweighted KNN error")
    plt.ylabel("Error Rate")
    plt.xlabel("Number of Neighbors")
    plt.savefig('knngraph.png', dpi=300)
    print er
    print et


def nn(tx, ty, rx, ry, iterations):
    network = buildNetwork(14, 5, 5, 1)
    ds = ClassificationDataSet(14,1, class_labels=["<50K", ">=50K"])
    for i in xrange(len(tx)):
        ds.addSample(tx[i], [ty[i]])
    trainer = BackpropTrainer(network, ds)
    trainer.trainOnDataset(ds, iterations)
    NetworkWriter.writeToFile(network, "network.xml")
    results = sum((np.array([round(network.activate(test)) for test in rx]) - ry)**2)/float(len(ry))
    return results

def loadnn(name):
    network = NetworkReader(name)


def boosting(tx, ty, rx, ry, n):
    clf = ensemble.AdaBoostClassifier(n_estimators=n)
    clf.fit(tx, ty)
    return sum((clf.predict(rx) - ry)**2)/float(len(ry))


def svm(tx, ty, rx, ry):
    clf = SVC(kernel="linear")
    clf.fit(tx, ty)
    return sum((clf.predict(rx) - ry)**2)/float(len(ry))


def boostTest(tx, ty, rx, ry, iterations):
    resultst = []
    resultsr = []
    num = range(iterations)
    for i in xrange(iterations):
        print i
        clf = ensemble.AdaBoostClassifier(n_estimators=i+1)
        clf.fit(tx, ty)
        resultst.append(sum((clf.predict(tx) - ty)**2)/float(len(ty)))
        resultsr.append(sum((clf.predict(rx) - ry)**2)/float(len(ry)))
    plt.plot(num, resultst, 'ro', num, resultsr, 'bo', num+[iterations], [.5 for i in xrange(iterations+1)], "k--")
    plt.axis([0, iterations, 0, 1])
    plt.title("Boosted Decision Tree error")
    plt.ylabel("Percent Error")
    plt.xlabel("Number of Estimators")
    plt.savefig('boostgraph.png', dpi=500)
    return resultsr


def nntester(tx, ty, rx, ry, iterations):
    """
    builds, tests, and graphs a neural network over a series of trials as it is
    constructed
    """
    resultst = []
    resultsr = []
    positions = range(iterations)
    network = buildNetwork(100, 50, 1, bias=True)
    ds = ClassificationDataSet(100,1, class_labels=["valley", "hill"])
    for i in xrange(len(tx)):
        ds.addSample(tx[i], [ty[i]])
    trainer = BackpropTrainer(network, ds, learningrate=0.01)
    for i in positions:
        print trainer.train()
        resultst.append(sum((np.array([round(network.activate(test)) for test in tx]) - ty)**2)/float(len(ty)))
        resultsr.append(sum((np.array([round(network.activate(test)) for test in rx]) - ry)**2)/float(len(ry)))
        print i, resultst[i], resultsr[i]
    NetworkWriter.writeToFile(network, "network.xml")
    plt.plot(positions, resultst, 'ro', positions, resultsr, 'bo')
    plt.axis([0, iterations, 0, 1])
    plt.ylabel("Percent Error")
    plt.xlabel("Network Epoch")
    plt.title("Neural Network Error")
    plt.savefig('3Lnn.png', dpi=300)


def cvnntester(tx, ty, rx, ry, iterations, folds):
    network = buildNetwork(100, 50, 1, bias=True)
    ds = ClassificationDataSet(100,1, class_labels=["valley", "hill"])
    for i in xrange(len(tx)):
        ds.addSample(tx[i], [ty[i]])
    trainer = BackpropTrainer(network, ds, learningrate=0.005)
    cv = CrossValidator(trainer, ds, n_folds=folds, max_epochs=iterations, verbosity=True)
    print cv.validate()
    print sum((np.array([round(network.activate(test)) for test in rx]) - ry)**2)/float(len(ry))


def treeTest(tx, ty, rx, ry, iterations):
    resultst = []
    resultsr = []
    num = range(iterations)
    for i in xrange(iterations):
        print i
        clf = tree.DecisionTreeClassifier(max_depth=i+1, criterion="entropy")
        clf.fit(tx, ty)
        resultst.append(sum((clf.predict(tx) - ty)**2)/float(len(ty)))
        resultsr.append(sum((clf.predict(rx) - ry)**2)/float(len(ry)))
    plt.plot(num, resultst, 'ro', num, resultsr, 'bo', num+[50], [.5 for i in range(iterations+1)], 'k')
    plt.axis([0, iterations, 0, 1])
    plt.title("Decision Tree error")
    plt.ylabel("Error Rate")
    plt.xlabel("Maximum Tree Depth")
    plt.savefig('entropytree.png', dpi=500)
    return resultsr


def treeConfusion(tx, ty, rx, ry):
    clf = tree.DecisionTreeClassifier(max_depth=40, criterion="gini")
    results = clf.fit(tx, ty).predict(rx)

    cm = confusion_matrix(ry, results)
    return cm

if __name__ == "__main__":
    tx, ty, rx, ry = start_hills()
    print "Decision Tree: " + str(decisionTree(rx, ry, tx, ty, 1))

    # print "Nearest Neighbor: " + str(knn(rx, ry, tx, ty, 1))
    # print "3-Nearest Neighbors: " + str(knn(rx, ry, tx, ty, 3))
    # print "5-Nearest Neighbors: " + str(knn(rx, ry, tx, ty, 25))
    # print "Neural Network: " + str(nn(tx, ty, rx, ry, 100))
    # print "Boosting (100): " + str(boosting(tx, ty, rx, ry, 100))
    # print "SVM: " + str(svm(tx, ty, rx, ry))
    # print "Boosting (500): " + str(boosting(tx, ty, rx, ry, 500))
    # pprint(boostTest(tx, ty, rx, ry, 500))
    # pprint(treeTest(tx, ty, rx, ry, 50))
    # nntester(tx, ty, rx, ry, 500)
    # knntester(tx, ty, rx, ry, 100)
    # cvnntester(tx, ty, rx, ry, 500, 10)
    # print "SVM: " + str(svm(tx, ty, rx, ry))
    # print "Boosting (500): " + str(boosting(tx, ty, rx, ry, 500))
    # print treeConfusion(tx, ty, rx, ry)
