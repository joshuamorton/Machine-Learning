import numpy as np
from StringIO import StringIO
from pprint import pprint
import argparse
from matplotlib import pyplot as pl


from sklearn.decomposition.pca import PCA as PCA
from sklearn.decomposition import FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as RandomProjection
from sklearn.cluster import KMeans as KM
from sklearn.mixture import GMM as EM
from sklearn.feature_selection import SelectKBest as best
from sklearn.feature_selection import chi2


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
    plt.axis(*axes)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(name+".png", dpi=500)
    plt.show()
    plt.clf()


def pca(tx, ty, rx, ry):
    pass

def ica():
    pass

def em(tx, ty, rx, ry, times=5):
    errs = []

    checker = EM(n_components=2)
    checker.fit(ry)
    compare = checker.predict(ry)
    print compare

    for i in range(2,times):
        clf = EM(n_components=i)
        clf.fit(tx)
        result = clf.predict(rx)
        errs.append(sum((result-compare)**2) / float(len(ry)))
    #print errs

def km():
    pass

def kbest():
    pass



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run clustering algorithms on stuff')
    parser.add_argument("name")
    args = parser.parse_args()
    name = args.name
    test = name+".data"
    train = name+".test"
    train_x, train_y, test_x, test_y = create_dataset(name, test, train)
    em(train_x, train_y, test_x, test_y)