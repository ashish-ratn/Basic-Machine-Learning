import dicom
import os
import pandas as import pd
import matplotlib as plt
import cv2
import numpy as np

IMG_PX_SIZE = 150


data_dir = '../input/sample_images/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('../input/stage1_labels.csv',index_col=0)

labels_df.head()

for patient in patients[:10]:
    label = labels_df.get_value(patient,'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/'+ s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    print(len(slices), label)

    fig = plt.figure()
    for num,each_slice in enumerate(slices[:12]):
        y= fig.add_subplot(3,4,num+1)
        new_image = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE), cmap='grey')
        y.imshow(new_image.pixel_array)
        plt.show()
    #print(slices[0])
