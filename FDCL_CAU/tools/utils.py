import numpy as np

from matplotlib import pyplot as plt

class Plotter:
    def __init__(self, attributes=[('trainloss', 1)]):
        self.attributes = attributes[0]

        attr = []
        for i, dictionary in enumerate(self.attributes):
            if i < len(self.attributes) - 1:
                attr.append(dictionary)
                freq = self.attributes[len(self.attributes)-1]

                setattr(self, attr[i], [])
                setattr(self, attr[i] + '_freq', freq)

        # self.attributes = attributes
        #
        # for dictionary in attributes:
        #     attr = dictionary[0]
        #     freq = dictionary[1]
        #     setattr(self, attr, [])
        #     setattr(self, attr + '_freq', freq)


    def log(self, attr, value):
        getattr(self, attr).append(value)

    def savelog(self, filename):
        pass

    def plot(self, ylabel, attributes=None, ymax=None, filename='plot.png'):

        plt.style.use('ggplot')
        color = ['r', 'b', 'g', 'y']
        if ymax is not None:
            plt.ylim(ymax=ymax)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)

        # if kwargs is not None:
        #     for key, value in kwargs:
        #         getattr(plt, key)(value)

        if attributes is None:
            attributes = [attr[0] for attr in self.attributes]

        for i, attr in enumerate(attributes):
            Xs = getattr(self, attr + '_freq') * np.arange(1, len(getattr(self, attr)) + 1)
            Ys = getattr(self, attr)
            # print(Xs)
            # print(Ys)
            plt.plot(Xs, Ys, label=attr, color = color[i])

        plt.legend()
        plt.savefig(filename)
        plt.close()




