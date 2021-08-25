import matplotlib.pyplot as plt
import csv
import numpy as np


def show_scatter_plot():
    scatter_plot = dict()
    for x in ['L', 'B', 'R']:
        scatter_plot[x] = list()
    for line in file:
        data = line.split(',')
        scatter_plot[data[0]].append((int(data[1]) * int(data[2]), int(data[3]) * int(data[4])))
        for x in ['L', 'B', 'R']:
            plt.scatter([lis[0] for lis in scatter_plot[x]], [lis[1] for lis in scatter_plot[x]],
                        label='class {}'.format(x))
    plt.ylabel('left_distance * left_weight')
    plt.xlabel('right_distance * right_weight')
    plt.title('scatter plot')
    plt.show()


def box_plot(data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    classes = {
        'L': [[], [], [], []],
        'R': [[], [], [], []],
        'B': [[], [], [], []]
    }
    left_dis, left_weight, right_weight, right_dis = list(), list(), list(), list()
    for row in data:
        left_dis.append(int(row[1]))
        left_weight.append(int(row[2]))
        right_dis.append(int(row[3]))
        right_weight.append(int(row[4]))
        classes[row[0]][0].append(int(row[1]))
        classes[row[0]][1].append(int(row[2]))
        classes[row[0]][2].append(int(row[3]))
        classes[row[0]][3].append(int(row[4]))

    ax1.boxplot(classes['L'])
    ax2.boxplot(classes['B'])
    ax3.boxplot(classes['R'])


file = open('balance-scale.csv', 'r+')
choice = int(input('Do you want show scatter plot (1) or box plot (2)'))
if choice == 1:
    show_scatter_plot()
else:
    data = csv.reader(file)
    box_plot(data)
    plt.show()
