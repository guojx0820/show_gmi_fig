

**练习2-1**

![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image001.png)

**（一）** **Python代码与注释详解（第三次更新后的代码）：**

 

```python
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
```

 

**（二）** **结果与总结：**

1、直接显示结果（无拉伸）：



 

**![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image003.png)**

**![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image005.png)**

**![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image007.png)**

**图****1.1 GMI****数据****2020****年一月份****6****个参数月平均数值空间分布（无拉伸与处理）**

 

2、归一化拉伸结果：

将图像的数值归一化到0-255之间所显示的结果：

![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image008.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image009.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image010.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image011.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image012.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image013.jpg)

**图****1.2 GMI****数据****2020****年一月份****6****个参数月平均数值空间分布（归一化处理：****0-255****）**

3、线性拉伸结果：

将图像的数值限行拉伸，乘以比例因子10，再给单位除以10，其显示效果更好，但降雨率由于值太小，显示效果不佳：

![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image014.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image015.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image016.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image017.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image018.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image019.jpg)

**图****1.3 GMI****数据****2020****年一月份****6****个参数月平均数值空间分布（线性拉伸）**

4、阈值法线性拉伸结果：

将图像的数值阈值法线性拉伸，用numpy.where语句修改数组中符合条件的元素，条件即是设置阈值（0-250），再乘以比例因子10，再给单位除以10，其显示效果更好的同时，去除了陆地等异常值的影响：



 

![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image020.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image021.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image022.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image023.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image024.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image025.jpg)

**图****1.4 GMI****数据****2020****年一月份****6****个参数月平均数值空间分布（阈值法线性拉伸）**

2、总结：

2020年1月份的6个不同参数的空间分布图的三种方法处理展示，运行时只需将路径和时间稍作修改即可。在学习编程过程中，各种方法对比分析极为重要，需要分析各个方法的优劣来提升编程速度和运行效率。第三种线性拉伸方法最佳，可以更加清楚明了地展示各个参数的空间分布大小。

**
**

 

**练习2-2**

![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image026.png)

**（一）** **Python代码与注释详解：**

 

```python
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
```

 

**（二）** **结果与总结：**

1、两个区域：

区域1：  35˚N-40˚N；120˚E-125˚E

区域2：  1˚N-6˚N；130˚E -135˚E

SST参数空间分布图：

**![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image027.png)****![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image028.png)**

**图****2.1** **两个地区的****2020****年****1****月****SST****示例图**

2、两个地区的2020年1月至2021年8月GMI数据6个参数：

（1）Sea surface temperature

（2）Ten-meter wind speed using low frequency channels

（3）Ten-meter wind speed using medium frequency channels

（4）Columnar atmospheric water vapor

（5）Columnar cloud liquid water content

（6）Rain rate

变化曲线图绘制结果展示：优化后：上次作业展示结果有误，以下是正确的计算结果：

![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image029.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image030.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image031.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image032.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image033.jpg)![img](file:////Users/leo/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image034.jpg)

**图****2.2** **两个地区的****2020****年****1****月至****2021****年****8****月****GMI****数据****6****个参数变化曲线图（优化后）**

3、总结：

主要的难点在于环境的配置与搭建，学习并熟练掌握Python的各种函数库的配置与应用是以后学习的核心内容，Python语法中，需要掌握各种函数来对数组进行运算与转换，图形绘制与图形处理也开始慢慢学习。