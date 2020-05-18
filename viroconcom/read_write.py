#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads datasets, reads and writes contour coordinates.
"""
import numpy as np
import csv


def read_ecbenchmark_dataset(path='datasets/1year_dataset_A.txt'):
    """
    Reads a 2D dataset that uses a an ASCI format with ';' as a seperator.

    This format has been used in the EC benchmark,
    see https://github.com/ec-benchmark-organizers/ec-benchmark .

    Parameters
    ----------
    path : string
        Path to dataset including the file name, defaults
        to '../datasets/1year_dataset_A.txt'
    Returns
    -------
    x : ndarray of doubles
        Observations of the environmental variable 1.
    y : ndarray of doubles
        Observations of the environmental variable 2.
    x_label : str
        Label of the environmantal variable 1.
    y_label : str
        Label of the environmental variable 2.
    """

    x = list()
    y = list()
    x_label = None
    y_label = None
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        idx = 0
        for row in reader:
            if idx == 0:
                x_label = row[1][1:] # Ignore first char (is a white space).
                y_label = row[2][1:] # Ignore first char (is a white space).
            if idx > 0: # Ignore the header
                x.append(float(row[1]))
                y.append(float(row[2]))
            idx = idx + 1

    x = np.asarray(x)
    y = np.asarray(y)
    return (x, y, x_label, y_label)


def write_contour(x, y, path, label_x='Variable x (unit)',
                  label_y='Variable y (unit)'):
    """
    Writes 2D contour coordinates in an ASCI format with ';' as a seperator.

    Parameters
    ----------
    x : ndarray of doubles
        Values in the first dimensions of the contour's coordinates.
    y : ndarray of doubles
        Values in the second dimensions of the contour's coordinates.
    path : string
        Path including folder and file name where the contour should be saved.
    label_x : str
        Name and unit of the first environmental variable,
        defaults to 'Variable x (unit), could be, for exmaple,
        'Significant wave height (m)'.
    label_y : str
        Name and unit of the second environmental variable,
        defaults to 'Variable y (unit)', could be, for example,
         'Zero-up-crossing period (s)'.
    """
    with open(path, mode='w', newline='') as contour_file:
        contour_writer = csv.writer(contour_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        contour_writer.writerow([label_x, label_y])
        for xi,yi in zip(x,y):
            contour_writer.writerow([str(xi), str(yi)])


def read_contour(path):
    """
    Reads 2D contour coordinates in an ASCI format with ';' as a seperator.

    Parameters
    ----------
    path : string
        Path to contour including the file name.
    Returns
    -------
    x : ndarray of doubles
        Observations of the environmental variable 1.
    y : ndarray of doubles
        Observations of the environmental variable 2.
    """

    x = list()
    y = list()
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        idx = 0
        for row in reader:
            if idx > 0: # Ignore the header
                x.append(float(row[0]))
                y.append(float(row[1]))
            idx = idx + 1

    x = np.asarray(x)
    y = np.asarray(y)
    return (x, y)
