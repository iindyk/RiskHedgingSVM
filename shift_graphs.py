import matplotlib.pyplot as plt
import numpy as np

n = 100


def graph_concept_drift(n):
    # generate the data
    data = np.random.uniform(0, 1, (n, 2))
    labels = []
    colors = []
    for x in data:
        if x[1] > x[0]:
            labels.append(1.)
            colors.append((1, 0, 0))
        else:
            labels.append(-1.)
            colors.append((0, 0, 1))
    # plot original data and concept drift
    plt.subplot(121)
    plt.scatter([float(i[0]) for i in data], [float(i[1]) for i in data], c=colors, cmap=plt.cm.coolwarm)
    x = np.arange(-0.5, 1.5, 0.01)
    plt.plot(x, x, 'g-', linewidth=3, label='true decision boundary')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, labelleft=False)
    plt.subplot(122)
    plt.scatter([float(i[0]) for i in data], [float(i[1])**2 for i in data], c=colors, cmap=plt.cm.coolwarm)
    plt.plot(x, np.square(x), 'g-', linewidth=3, label='true decision boundary')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax = plt.gca()
    fig = plt.gcf()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    plt.show()


def graph_covariate_shift(n):
    # generate the data
    data = np.random.uniform(0, 1, (n, 2))
    labels = []
    colors = []
    for x in data:
        if x[1] > ((x[0]-0.5)**2)*3+0.25:
            labels.append(1.)
            colors.append((1, 0, 0))
        else:
            labels.append(-1.)
            colors.append((0, 0, 1))
    # plot original data and concept drift
    plt.subplot(122)
    plt.scatter([float(i[0]) for i in data], [float(i[1]) for i in data], c=colors, cmap=plt.cm.coolwarm)
    x = np.arange(-0.5, 1.5, 0.01)
    plt.plot(x, np.square(x-0.5)*3+0.25, 'g-', linewidth=3, label='true decision boundary')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, labelleft=False)
    plt.subplot(121)
    plt.scatter([float(i[0]) for i in data if i[1]<3*i[0]-0.85],
                [i[1] for i in data if i[1]<3*i[0]-0.85], c=[colors[i] for i in range(n) if data[i, 1] < 3*data[i, 0]-0.85],
                cmap=plt.cm.coolwarm)
    plt.plot(x, np.square(x-0.5)*3+0.25, 'g-', linewidth=3, label='true decision boundary')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax = plt.gca()
    fig = plt.gcf()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    plt.show()


if __name__ == '__main__':
    graph_covariate_shift(n)

    '''top=0.881,
    bottom=0.039,
    left=0.028,
    right=0.972,
    hspace=0.2,
    wspace=0.072'''