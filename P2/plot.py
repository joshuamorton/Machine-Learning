import matplotlib.pyplot as plt
from sys import argv



def plottheThing(sourceFile):
    plt.clf()
    with open(sourceFile+".txt") as f:
        data = f.read()
    errors = [float(value) for value in data.split("\n") if value]
    plt.plot(range(len(errors)), errors, 'ro')
    plt.axis([0, len(errors), 0, 100])
    plt.title("Error from "+sourceFile)
    plt.ylabel("percent error")
    plt.xlabel("iteration")
    plt.savefig(sourceFile + "plot" + '.png', dpi=300)
    plt.show


if __name__ == "__main__":
    if len(argv) > 1:
        for arg in argv[1:]:
            plottheThing(arg)

