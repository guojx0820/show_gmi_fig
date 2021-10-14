'''
For:Display the Ocean parameters such as
（1）Sea surface temperature
（2）Ten-meter wind speed using low frequency channels
（3）Ten-meter wind speed using medium frequency channels
（4）Columnar atmospheric water vapor
（5）Columnar cloud liquid water content
（6）Rain rate
in the Area1 (35˚N-40˚N；120˚E-125˚E) and Area2 (1˚N-6˚N；130˚E -135˚E).

Time:10/09/2021
Author:Guo Jiaxiang
Email：guojiaxiang0820@gmail.com
GitHubBlog:https://github.com/guojx0820
'''

import gzip
import os, sys, re
import numpy as np
import pandas as pd
# Fantistic function lib
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt


# Define a gz file reading function.
# Read all the file which is ending of *.gz.
def read_gz_file(path):
    contents = []
    if os.path.exists(path):
        for i, _, _ in os.walk(path):
            for j in os.listdir(i):
                if j.endswith('.gz'):
                    if re.match('f35_202(\d+)v8.2.gz', j):
                        contents.append(os.path.join(i, j))
    else:
        print('File not exist!')
    return contents
    sys.exit(0)


# Set a title fun.
def set_cli():
    cliPar = ['Sea Surface Temperature', \
              '10m-Wind Speed Using Low Frequency Channels', \
              '10m-Wind Speed Using Medium Frequency Channels', \
              'Columnar Atmospheric Water Vapor', \
              'Columnar Cloud Liquid Water Content', \
              'Rain Rate']
    return cliPar


# Set a function of 6 units.
def set_units():
    units = ['SST(deg Celsius)', \
             'WindLF(m/s)', \
             'WindMF(m/s)', \
             'Vapor(mm)', \
             'Cloud(mm)', \
             'Rain(mm/hr)']
    return units


# Set a functon of getting longitude.
def get_lon(start, end):
    region_lon = 4 * np.array([start, end])
    return region_lon


# Set a function of getting latitude.
def get_lat(start, end):
    region_lat = 359 - 4 * np.array([start, end])
    # Reverse the order of elements in an array along the given axis.
    # Flip an array vertically (axis=0).
    # Flip an array horizontally (axis=1).
    region_lat = np.flip(region_lat, axis=0)
    return region_lat


# Set a function of getting data of 2 regions.
def get_data(region_lon, region_lat, scale, offset, shape, total_data):
    # Get a month list with func total_data.key().
    month_list = list(total_data.keys())
    # Define a global variable which can be use in everywhere of the program.
    global region_data
    region_data = {}
    # Set scale and offset.
    scale = np.array(scale)
    scale = scale.reshape(shape)
    offset = np.array(offset)
    offset = offset.reshape(shape)
    # Extract the numbers of array in interesting regions.
    for i in month_list:
        region_data[i] = []
        temp_data_0 = total_data[i][:, region_lat[0]:region_lat[1], region_lon[0]:region_lon[1]]
        for j in range(temp_data_0.shape[0]):
            temp_data_1 = temp_data_0[j]
            # Array mutiplies by scale and adds offset.
            temp_data_2 = (temp_data_1[np.where((temp_data_1 >= 0) & (temp_data_1 <= 250))]) * scale[j] + offset[j]
            region_data[i].append(np.mean(temp_data_2))
            # Convert dictionary to array.
        region_data[i] = np.array(region_data[i])
    region_data = np.array(list(region_data.values()))
    return region_data


# Set a function using set the styles of x axis.
def set_x_axis(tdata):
    # Set 3 global variables to transfrom the time parameters.
    global month_name, xtime_sort, xtime_label
    month_name = []
    xtime_sort = []
    xtime_label = []
    # Set a list save month name.
    time_list = list(tdata.keys())
    # Set the x labels with self style.
    '''
     The enumerate () function is used to combine a traversable data object (such as list, tuple or string)
     into an index sequence, and list data and data subscripts at the same time. 
     It is generally used in the for loop.
    '''
    for i, j in enumerate(time_list, 0):
        month_name.append(datetime.strptime(j, '%Y%m'))
        if i % 1 == 0:
            xtime_sort.append(datetime.strptime(j, '%Y%m'))
            if i % 2 == 0:
                xtime_label.append(j)
            else:
                xtime_label.append('')
    month_name = np.array(month_name)
    # print(month_name, xtime_sort, xtime_label)


# Figure drawing function.
def draw_fig(save_path):
    # State global variable for calling.
    global month_name, xdate_sort, xdate_label
    # Drawing a curve figure.
    fig = plt.figure('Polyline Drawing')
    # Set x lables but show nothing because of the narrow margin.
    plt.xlabel('Month', fontsize=14)
    # Display 6 pictures in sequence.
    for i in range(6):
        sc = set_cli()
        su = set_units()
        plt.ylabel(su[i], fontsize=14)
        plt.title(sc[i], fontsize=16)
        plt.xticks(xtime_sort, xtime_label, rotation=30, fontsize=12)
        plt.yticks(fontsize=12)
        # Polyline drawing.
        plt.plot(month_name, gd1[:, i], color='r', label='Region1', linestyle='-.', linewidth=2, marker='o',
                 markersize=6)
        plt.plot(month_name, gd2[:, i], color='b', label='Region2', linestyle='--', linewidth=2, marker='v',
                 markersize=6)
        # Mesh rendering.
        plt.grid(linestyle='-', axis='y', which='major')
        plt.legend()
        sns.despine(left=True)
        # sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": " "})
        sns.set_context("notebook")
        plt.savefig(os.path.join(save_path, (sc[i] + '.jpg')), dpi=1200)
        plt.show()
        plt.clf()


# Set a main function which is calling other function.
if __name__ == '__main__':
    # Read *.gz files.
    dir = r'V:\Softwares\PyCharm\Project2'
    row = int(360 / 0.25)
    col = int(180 / 0.25)
    # Because parameter is file path，so it's very reasonable to pass the path parameter.
    fileData = read_gz_file(dir)
    total_data = {}
    # Reshape the data and save it in a data list
    for i in fileData:
        with gzip.open(i, mode='rb') as fp:
            data = np.array(bytearray(fp.read()), dtype=np.float64).reshape(6, col, row)
        data = np.flip(data, 1)
        date = os.path.basename(i)[4:10]
        total_data[date] = data
    # Set x axis
    setx = set_x_axis(total_data)
    # Calling get_lon() and get_lat() to set region1 and region2
    region1_lon = get_lon(120, 125)
    region1_lat = get_lat(35, 40)
    region2_lon = get_lon(130, 135)
    region2_lat = get_lat(1, 6)
    # Pass the parametes to the scale and offset.
    scale = [0.15, 0.2, 0.2, 0.3, 0.01, 0.1]
    offset = [-3, 0, 0, 0, -0.05, 0]
    shape = (6, 1, 1)
    #Calling the function get_data() to plot the figure
    gd1 = get_data(region1_lon, region1_lat, scale, offset, shape, total_data)
    gd2 = get_data(region2_lon, region2_lat, scale, offset, shape, total_data)
    #Calling the draw_fig() and save the images to the folder
    save_path = dir + '\Img2'
    img = draw_fig(save_path)
