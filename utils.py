import matplotlib.pyplot as plt
import numpy as np
#import wmi

def split_train_test(images, masks, validation_split=0.8):
    split_index = int(images.shape[0] * 0.8)
    return images[0:split_index], masks[0:split_index], images[split_index:], masks[split_index:]

def normalize( images, masks):
    images = images / 255
    masks = (masks > 0).astype(float)
    return images, masks

def display_pair(image1, image2, title1='', title2=''):
   
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1,2,1)
    ax.set_title(title1)
    ax.imshow(image1, cmap='gray')

    ax1 = fig.add_subplot(1,2,2)
    ax1.set_title(title2)
    ax1.imshow(image2, cmap='gray')


    plt.show()

# def display_sys_info():
#     computer = wmi.WMI()
#     computer_info = computer.Win32_ComputerSystem()[0]
#     os_info = computer.Win32_OperatingSystem()[0]
#     proc_info = computer.Win32_Processor()[0]
#     gpu_info = computer.Win32_VideoController()[0]

#     os_name = os_info.Name.encode('utf-8').split(b'|')[0]
#     os_version = ' '.join([os_info.Version, os_info.BuildNumber])
#     system_ram = float(os_info.TotalVisibleMemorySize) / 1048576  # KB to GB

#     print('OS Name: {0}'.format(os_name))
#     print('OS Version: {0}'.format(os_version))
#     print('CPU: {0}'.format(proc_info.Name))
#     print('RAM: {0} GB'.format(system_ram))
#     print('Graphics Card: {0}'.format(gpu_info.Name))