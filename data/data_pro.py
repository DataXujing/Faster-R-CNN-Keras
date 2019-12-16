
import os
import pandas 
import shutil
import random


import cv2
import numpy as np
import xml.etree.ElementTree as ET


# 这部分休要修改


class Data_preprocess(object):
    '''
    解析xml数据
    '''
    def __init__(self,data_path):
        self.data_path = data_path
        self.classes = ["hat","person"]



    def load_data(self):

        image_names = os.listdir(os.path.join(self.data_path,"JPEGImages"))
        image_index = 0

        for image_name in image_names:
            image_index += 1
            image_path = os.path.join(self.data_path,"JPEGImages",image_name)
            # anno_name = image_name.lower().rstrip(".jpg")+".xml"
            # anno_name = image_name.rstrip(".jpg")+".xml"
            anno_name = image_name.split(".")[0]+".xml"
            filename = os.path.join(self.data_path, 'Annotations', anno_name)
            tree = ET.parse(filename)
            image_size = tree.find('size')
            image_width = int(float(image_size.find('width').text))
            image_height = int(float(image_size.find('height').text))
            # h_ratio = 1.0 * self.image_size / image_height
            # w_ratio = 1.0 * self.image_size / image_width

            objects = tree.findall('object')

            for obj in objects:
                box = obj.find('bndbox')
                x1 = int(float(box.find('xmin').text))
                y1 = int(float(box.find('ymin').text))
                x2 = int(float(box.find('xmax').text))
                y2 = int(float(box.find('ymax').text))
                # x1 = max(min((float(box.find('xmin').text)) * w_ratio, self.image_size), 0)
                # y1 = max(min((float(box.find('ymin').text)) * h_ratio, self.image_size), 0)
                # x2 = max(min((float(box.find('xmax').text)) * w_ratio, self.image_size), 0)
                # y2 = max(min((float(box.find('ymax').text)) * h_ratio, self.image_size), 0)
                class_name = obj.find('name').text.lower()
                if class_name not in self.classes:
                    continue
                # class_ind = self.class_to_ind[obj.find('name').text.lower().strip()]

                # boxes = [0.5 * (x1 + x2) / self.image_size, 0.5 * (y1 + y2) / self.image_size, np.sqrt((x2 - x1) / self.image_size), np.sqrt((y2 - y1) / self.image_size)]
                # cx = 1.0 * boxes[0] * self.cell_size
                # cy = 1.0 * boxes[1] * self.cell_size
                # xind = int(np.floor(cx))
                # yind = int(np.floor(cy))
                
                # label[yind, xind, :, 0] = 1
                # label[yind, xind, :, 1:5] = boxes
                # label[yind, xind, :, 5 + class_ind] = 1

                if x1 >= x2 or y1 >= y2:
                    continue

                with open("./data/annotations.txt",'a',encoding='utf-8') as f:
                    #filename,x1,y1,x2,y2,class_name
                    f.write(image_path+","+str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+class_name+"\n")
                # f.close()

            print('[ INFO ] index:{}, path:{}'.format(image_index,image_path))
          





if __name__ == "__main__":
    
    # 做Faster R-CNN需要的训练集
    base_path = os.getcwd()
    data_path = os.path.join(base_path,"data")  # 绝对路径

    data_p = Data_preprocess(data_path)
    data_p.load_data()

    print("==========data pro finish===========")







