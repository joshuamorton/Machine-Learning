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
from StringIO import StringIO
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

import matplotlib.pyplot as plt


#training data is contained in "adult.data"

"""
the csv data is stored as such:
age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income
these values are explained in the file adult.names
our first step is to parse the data
"""

#first we define a set of conversion functions from strings to integer values because working with strings is dumb
#especially since the computer doens't care when doing machine learning
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

converters = {
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

"""
load the data into numpy
this section is also written for use in a
"""
train = "adult.data"
test = "adult.test"


def load(filename):
    with open(filename) as data:
        adults = [line for line in data if "?" not in line]  # remove lines with unknown data

    return np.loadtxt(adults,
        delimiter=', ',
        converters=converters,
        dtype='u4',
        skiprows=1
        )

def start_adult():
    """
    tx - training x axes
    ty - training y axis
    rx - result (testing) x axes
    ry - result (testing) y axis
    """
    tr = load(train)
    te = load(test)
    tx, ty = np.hsplit(tr, [14])
    rx, ry = np.hsplit(te, [14])
    ty = ty.flatten()
    ry = ry.flatten()
    return tx, ty, rx, ry 


def decisionTree(tx, ty, rx, ry):
    """
    """
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
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
    clf = SVC()
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
    plt.plot(num, resultst, 'ro', num, resultsr, 'bo')
    plt.axis([0, iterations, 0, 1])
    plt.show()


if __name__ == "__main__":
    tx, ty, rx, ry = start_adult()
    # print "Decision Tree: " + str(decisionTree(rx, ry, tx, ty))
    # print "Nearest Neighbor: " + str(knn(rx, ry, tx, ty, 1))
    # print "3-Nearest Neighbors: " + str(knn(rx, ry, tx, ty, 3))
    # print "5-Nearest Neighbors: " + str(knn(rx, ry, tx, ty, 25))
    # print "Neural Network: " + str(nn(tx, ty, rx, ry, 100))
    # print "Boosting (100): " + str(boosting(tx, ty, rx, ry, 100))
    # print "Boosting (500): " + str(boosting(tx, ty, rx, ry, 500))
    # print "SVM: " + str(svm(tx, ty, rx, ry))
    boostTest(tx, ty, rx, ry, 25)

