'''
Download monthly average GMI data from January 2020 to August 2021
Display and save the monthly parameter images,
which shall be marked with longitude and latitude and color table value

Time:10/10/2021
Author:Guo Jiaxiang
Email：guojiaxiang0820@gmail.com
GitHubBlog:https://github.com/guojx0820
'''

import gzip, os
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap


def setX(temp, position):
    x = int(temp / 4)
    if x < 180:
        x = 'E' + str(x)
    else:
        x -= 360
        x = 'W' + str(-x) + str()
    return x


def setY(temp, position):
    y = int(-(temp / 4 - 90))
    if y > 0:
        y = 'N' + str(y)
    else:
        y = 'S' + str(-y)
    return y


# Set a title fun.
def set_cli():
    cliPar = ['Sea Surface Temperature', \
              '10m-Wind Speed of Low Frequency Channels', \
              '10m-Wind Speed of Medium Frequency Channels', \
              'Columnar Atmospheric Water Vapor', \
              'Columnar Cloud Liquid Water Content', \
              'Rain Rate']
    return cliPar


# Set a function of 6 units.
def set_units():
    units = ['SST(deg Celsius)', \
             'WindLF(dm/s)', \
             'WindMF(dm/s)', \
             'Vapor(mm)', \
             'Cloud(10^-1 mm)', \
             'Rain(10^-1 mm/hr)']
    return units


# Modify the colorabar "jet" to get the color you want
def get_spectral():
    # Create a new array to store color values
    colormap_float = np.zeros((256, 3), np.float64)
    # Some color values are customized, and the author can set them as needed
    for i in range(0, 256, 1):
        colormap_float[i, 0] = cm.jet(i)[0] * 255.0
        colormap_float[i, 1] = cm.jet(i)[1] * 255.0
        colormap_float[i, 2] = cm.jet(i)[2] * 255.0
        # Assign the original "jet" color to the colormap_ In float
        # Some color values are customized, and the author can set them as needed
        colormap_float[255, :] = [130, 130, 130]
        colormap_float[254, :] = [0, 0, 0]
        colormap_float[253, :] = [182, 251, 255]
        colormap_float[252, :] = [255, 255, 255]
        colormap_float[251, :] = [255, 84, 156]
    return (colormap_float)


# Set a function of read *.gz files
def read_gz_file(path):
    contents = []
    if os.path.exists(path):
        with gzip.open(path, 'rb') as fp:
            contents = fp.read()
    else:
        print('File does not exist!')
    return contents


# Set a image drawding function.
def draw_fig(data, save_path, scale, offset, lon, lat, colormap_float):
    # Draw 6 images of every month using for loop.
    for i in range(6):
        temp = data[i, :, :]
        # Eliminate the abnormal value and take the normal value.
        index_valid = np.where(temp <= 250)
        # Calculate the maximum, minimum and mean values.
        min_value = np.round(np.min(temp[index_valid]) * scale[i] + offset[i])
        max_value = np.round(np.max(temp[index_valid]) * scale[i] + offset[i])
        mean_value = np.round(np.mean(temp[index_valid]) * scale[i] + offset[i])
        # Array flip.
        temp = np.flip(temp, 0)
        # Print the maximum, minimum and mean values.
        print(min_value, mean_value, max_value, temp)
        print(temp[np.where((temp >= 0) & (temp <= 250))])
        # Linearly stretch the image according to the mean
        '''
        Use the numpy.where statement to modify the qualified elements in the array. 
        The condition is to set the threshold (0-250), multiply by the scale factor 10, 
        and then divide the unit by 10. 
        The display effect is better and the influence of abnormal values such as land is removed
        '''
        if mean_value <= 10:
            (temp[np.where((temp >= 0) & (temp <= 250))]) = (temp[np.where((temp >= 0) & (temp <= 250))]) * 10
            min_value = min_value * 10
            max_value = max_value * 10
            if mean_value <= 1:
                (temp[np.where((temp >= 0) & (temp <= 250))]) = (temp[np.where((temp >= 0) & (temp <= 250))]) * 100
                min_value = min_value * 100
                max_value = max_value * 100
                # (temp[np.where((temp >= 0) & (temp <= 250))])
        # gap = 3
        # if gap > 3:
        #     gap = 3
        # else:
        #     gap = 1
        # Set color table and convert the obtained RGB value into the format of hash table, which I named 'sst cmap'
        rgb_table = LinearSegmentedColormap.from_list('sst cmap', colormap_float / 255.0)
        # Show Image and set display box properties
        fig = plt.figure("Show Image")
        sc = set_cli()
        su = set_units()
        plt.tight_layout(rect=(0, 0, 1, 0.9))  # 使子图标题和全局标题与坐标轴不重叠
        plt.xlabel(su[i], fontsize=14)
        plt.title('202001 ' + sc[i], fontsize=16)
        X, Y = lon, lat
        # Array normalization.
        temp_str = (temp * 256) / (max_value - min_value)
        # c = plt.pcolor(X, Y, temp, shading='auto', cmap=rgb_table)
        # Set a color bar.
        cd = plt.contourf(X, Y, temp_str, range(int(min_value), int(max_value), 1), cmap=rgb_table)
        # cs = plt.contourf(X, Y, temp, range(int(min_value), int(max_value), 5), cmap='jet')
        # cbar = plt.colorbar(cs, location='bottom', pad='10%')
        # Set x,y axis.
        plt.gca().xaxis.set_major_formatter(FuncFormatter(setX))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(setY))
        plt.axis('on')
        # Show images and colorbar.
        plt.imshow(temp_str, cmap=rgb_table)
        plt.colorbar(cd, orientation='horizontal', spacing='proportional')
        plt.grid(linestyle='-.', color='darkgray')
        # Save and show.
        plt.savefig(os.path.join(save_path, (sc[i] + '.jpg')), dpi=1200)
        # plt.savefig("202001_wspd_mf.png")
        plt.show()


# Set a main function to call other function
if __name__ == "__main__":
    dir = r"V:\Softwares\PyCharm\Project2"
    file_path = dir + r'\f35_202001v8.2.gz'
    # Set the number of row and col.
    row = int(360 / 0.25)
    col = int(180 / 0.25)
    data = read_gz_file(file_path)
    data = np.array(bytearray(data)).reshape(6, col, row)
    # Set scale and offset.
    scale = np.array([0.15, 0.2, 0.2, 0.3, 0.01, 0.1])
    offset = np.array([-3, 0, 0, 0, -0.05, 0])
    # Set longitude and latitude.
    lon = (np.linspace(1, 1, 720).reshape((720, 1)) @
           np.linspace(0., 360., 1440).reshape((1, 1440)))
    lat = (np.linspace(-90., 90., 720).reshape((720, 1)) @
           np.linspace(1, 1, 1440).reshape((1, 1440)))
    # Call the custom function to get large color, colormap_ Float is the array we want
    colormap_float = get_spectral()
    save_path = dir + '\Img1'
    # Call the draw_fig() function to draw the images.
    draw_fig = draw_fig(data, save_path, scale, offset, lon, lat, colormap_float)
    # print(lon, lat)
