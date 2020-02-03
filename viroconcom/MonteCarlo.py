# data: WSPD APD
#       xyz  xyz
#       ...  ...
# WSPD: WindSpeed
# WVHT: significantWaveHight
# APD: wavePeriode
#
# Hier erstmal nach dem Paper von elsevier Ocean Engineering
#
# Variablen:
# Tz = wavePeriod
# Hs = significantWaveHeight


#
# Daten aufbereiten.
#

import csv
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
from scipy import integrate


def _plot_data():
    data_list = []
    with open('SampleDataBuoy44009Year2000', 'r') as data:
        reader = csv.reader(data, delimiter=',')
        for row in reader:
            data_list.append(row)
    data_list.pop(0)
    for i in data_list:
        i.pop(1)
    data_list.pop(0)
    array = []
    array1 = []
    array2 = []
    for i in data_list:
        array1.append(i[0])
        array2.append(i[1])
    array.append(array1)
    array.append(array2)

    # print(array1)
    for i in data_list:
        i[0] = float(i[0])
        i[1] = float(i[1])
    print(data_list)

    # Array1 sortieren
    array1 = sorted(array1)

    # Durchschnitt
    a = 0.0
    for i in array1:
        a = a+float(i)
    mean = a/len(array1)
    print(mean)

    # Varianz
    b = 0.0
    for i in array1:
        b = b + ((float(i)-mean)*(float(i)-mean))
    var = b/(len(array1)-1)
    print(var)

    # Standartabweichung
    s = math.sqrt(var)
    print(s)

    # PlotData
    df = pd.DataFrame(data_list, columns=["x", "y"])
    sns.jointplot(x='x', y='y', data=df)
    plt.show()






import random
def __init__(self):
    return None
def _montecarlo():
    roll = random.randint(0,100)

    if roll == 100:
        #print(roll, 'roll was 100 you lose. What are the odds?!')
        return False
    elif roll <= 50:
        #print(roll, 'roll was 1-50, you lose')
        return False
    elif 100 > roll > 50:
        #print(roll, 'roll was 51 to 99, you win! ')
        return True


def doubler_bettor(funds, initial_wager, wager_counter):
    value = funds
    wager = initial_wager

    wX = []
    vY = []

    currentWager = 1
    previousWager = 'win'
    previousWagerAmount = initial_wager

    while currentWager <= wager_counter:
        if previousWager == 'win':
            print('we won last time wager, great')
            if _montecarlo():
                value+=wager
                print(value)
                wX.append(currentWager)
                vY.append(value)
            else:
                value -= wager
                previousWager = 'loss'
                print(value)
                previousWagerAmount = wager
                wX.append(currentWager)
                vY.append(value)
                if value < 0:
                    print('Broke after'.currentWager, 'bets')
                    break

        elif previousWager == 'loss':
            print('we lost the last one, so we will be smart and double')
            if _montecarlo():
                wager = previousWagerAmount * 2
                print('we won', wager)
                value += wager
                print(value)
                wager = initial_wager
                previousWager = 'win'
                wX.append(currentWager)
                vY.append(value)
            else:
                wager = previousWagerAmount * 2
                print('we lost', wager)
                value -= wager
                if value < 0:
                    print('we went broke after',currentWager,'bets')
                    break

                print(value)
                previousWager = 'loss'
                previousWagerAmount = wager
                wX.append(currentWager)
                vY.append(value)

        currentWager += 1
    print(value)
    plt.plot(wX, vY)



def simple_bettor(funds, initial_wager, wager_counter):
    value = funds
    wager = initial_wager

    wX = []
    vY = []
    curretnWager = 1

    while curretnWager <= wager_counter:
        if _montecarlo():
            value += wager
            wX.append(curretnWager)
            vY.append(value)
        else:
            value -= wager
            wX.append(curretnWager)
            vY.append(value)
        curretnWager += 1

    if value < 0:
        value = 'broke'
        #print('Funds:', value)

    plt.plot(wX,vY)


def _random_weibull(shape, loc, scale):
    x = np.linspace(0.0, 100.0, num=100)
    q = np.random.rand(10000)

    density = sts.weibull_min.pdf(x, c=shape, loc=loc, scale=scale)
    distribution = sts.weibull_min.cdf(x, c=shape, loc=loc, scale=scale)
    percent_point = sts.weibull_min.ppf(q, c=shape, loc=loc, scale=scale)

    plt.hist(percent_point)

    plt.show()
    value = sts.weibull_min.ppf(0.50, c=shape, loc=loc, scale=scale)
    print(percent_point)


def _monte_carlo(self, shape, loc, scale, samples):
    return

_random_weibull(1.5, 0.9, 2.8)