import json
import shutil

import numpy as np
import os
import cv2
import torch

# with open("/Users/gu/Downloads/training/label_2/000000.txt",'r') as f:
#     data = f.readlines()
#
# data = [d.strip() for d in data]
# img = cv2.imread("/Users/gu/Downloads/data_object_image_2/training/image_2/000000.png", cv2.IMREAD_UNCHANGED)
#
# print(data)
# for d in data:
#     box = d.split(" ")[4:8]
#     box = [float(b) for b in box]
#     cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]), int(box[3])),color=(255,0,0),thickness=1)
#
#
# cv2.imshow("test",img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# b = torch.FloatTensor(np.array([0, 0, 10, 10]))
# a = torch.FloatTensor(np.array([0, 0, 10, 10])).unsqueeze(0)
#
# print(b.shape)
# print(a.shape)
# path = "/mnt/Jan_data/Data/Object_detetction_data/KIIT2D/training_split/img"
# txt_dir = "/mnt/Jan_data/Data/Object_detetction_data/KIIT2D/training_split/txt"
# save_path = "/mnt/Jan_data/pycharm_file/myyolo/split_img_txt.json"
# names = os.listdir(path)
# json_dir = dict()
# for name in names:
#     img_path = os.path.join(path,name)
#     txt_path = os.path.join(txt_dir,name.replace("png","txt"))
#
#     if os.path.exists(txt_path) and os.path.exists(img_path):
#         json_dir[img_path] = txt_path
# # print(json_dir)
# with open(save_path,'w') as f:
#     json.dump(json_dir,f)
# dict_classes = {
#             'Car': 0, 'Van': 1, 'DontCare': 2,
#             'Pedestrian': 3, 'Truck': 4, 'Cyclist': 5,
#             'Misc': 6, 'Tram': 7, 'Person_sitting': 8
#         }
# def read_txt( txt_path):
#     out_list = []
#     with open(txt_path, 'r') as f:
#         data = f.readlines()
#
#     data = [d.strip() for d in data]
#     for d in data:
#         box = d.split(" ")[4:8]
#         box = [float(b) for b in box]
#         box.append(dict_classes[d.split(" ")[0]])
#         out_list.append(box)
#     return out_list
#
# def read_json(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     return data
# data = read_json("/mnt/Jan_data/myyolo/img_txt.json")
# print(data)
#
# for key,value in data.items():
#     out_list = read_txt(value)
#     print(len(out_list))
#     print(out_list)
#     break


# b = [1,2,3,4,5,6,7,8,9,0]
# b = torch.tensor(b)
# a = b.new(b.shape)
# print(a)
# a = [x for x in b[::2]]
# print(a)

# a = torch.tensor([9,5,4,10,19,0])
#
# # print(a)
# b = a.repeat(13,1)
# print(b)
# num, value = torch.max(b,dim=1,keepdim=True)
# print(num,value)

# a = list(np.random.choice(range(256), size=3))
# print(a)

# dict_classes = {
#             'Car': 0, 'Van': 1, 'DontCare': 2,
#             'Pedestrian': 3, 'Truck': 4, 'Cyclist': 5,
#             'Misc': 6, 'Tram': 7, 'Person_sitting': 8
#         }
#
# print(dict_classes[1])
# json_path = "/mnt/Jan_data/pycharm_file/myyolo/name_box.json"
# png_dir = "/mnt/Jan_data/Data/Object_detetction_data/small_coco/pngs/export"
# with open(json_path, 'r') as f:
#     data = json.load(f)
#
# for k, v in data.items():
#     img_path = os.path.join(png_dir, k)
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     for info in v:
#         x, y, w, h = info[1], info[2], info[3], info[4]
#         x1 = int(x)
#         x2 = x1 + int(w)
#         y1 = int(y)
#         y2 = y1 + int(h)
#         print(x1,y1,x2,y2)
#         cv2.circle(img,(int(x),int(y)),radius=1,color=(255,0,0),thickness=2)
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
#     cv2.namedWindow("test", cv2.WINDOW_NORMAL)
#     cv2.imshow("test",img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
# def list_dict(data):
#     test_json = dict()
#     for d in data:
#         test_json[d[0]] = d[1:]
#     return test_json
#
# def move_png(data,org_dir,trg_dir):
#     for d in data:
#         png_name = d[0]
#         shutil.copy(os.path.join(org_dir,png_name),os.path.join(trg_dir,png_name))
#
# def save_json(data, path):
#     with open(path,'w') as f:
#         json.dump(data,f)


# json_path = "/mnt/Jan_data/pycharm_file/myyolo/name_box.json"
# with open(json_path, 'r') as f:
#     data = json.load(f)
#
# reshape_data = []
# for k,v in data.items():
#     new_info = [k]
#     for info in v:
#         new_info += info
#
#     reshape_data.append(new_info)
#
#
#
# data_length = len(reshape_data)
# print(data_length)
# choose_2200 = np.array(reshape_data)[np.random.rand(data_length) <= (2200 / data_length)]
# val_test_index = np.random.rand(len(choose_2200)) <= 0.2
# train_index = ~val_test_index
# val_test = choose_2200[val_test_index]
# train_ = choose_2200[train_index]
#
# val_index = np.random.rand(len(val_test)) <= 0.5
# test_index = ~val_index
# val = val_test[val_index]
# test = val_test[test_index]
#
# print(test)
# test_json = list_dict(test)
# val_json = list_dict(val)
# train_json = list_dict(train_)
#
#
# png_dir = "/mnt/Jan_data/Data/Object_detetction_data/small_coco/pngs/export"
# root = "/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000"
# # create dir
# train_dir = os.path.join(root, "train")
# val_dir = os.path.join(root, "val")
# test_dir = os.path.join(root, "test")
#
# test_json_path = os.path.join(root,"test.json")
# val_json_path = os.path.join(root,"val.json")
# train_json_path = os.path.join(root,"train.json")
#
# os.makedirs(train_dir,exist_ok=True)
# os.makedirs(val_dir,exist_ok=True)
# os.makedirs(test_dir,exist_ok=True)
#
# move_png(test,png_dir,test_dir)
# move_png(val,png_dir,val_dir)
# move_png(train_,png_dir,train_dir)
#
# save_json(test_json,test_json_path)
# save_json(val_json,val_json_path)
# save_json(train_json,train_json_path)


#
# a = torch.linspace(0,12,13)
# print(a.shape)
# b = a.repeat(13,1)
# c = b.repeat(3*3,1,1)
# print(c.shape)
# anchor = [[116, 90], [156, 198], [373, 326]]
#
# a = torch.cuda.FloatTensor(anchor).index_select(1, torch.cuda.LongTensor([0]))
# print(a.shape)
# b = a.repeat(3,1)
# print(b.shape)

# a = torch.tensor(np.random.rand(3,20,19))
#
#
# for i, v in enumerate(a):
#     print(a[:,4].shape)
#     b = torch.squeeze(a[:,4]<0.1)
#     print(b.shape)


# svg_path = "/home/jan/Downloads/Image/Recall.svg"
# img = cv2.imread(svg_path)
# print(img)
# cv2.imshow("test",img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imwrite(svg_path.replace("svg","png"),img)

# json_path = "/mnt/Jan_data/Data/Object_detetction_data/small_coco/_annotations.coco.json"
# with open(json_path, 'r') as f:
#     data = json.load(f)
#
# images = data["images"]
# annotations = data["annotations"]

# for image in images:
#     anno_info = []
#     img_id = image["id"]
#     file_name = image["file_name"]
#     for annotation in annotations:
#         if annotation["image_id"] == img_id:
#             anno_info.append(annotation)
#


# new_dict = dict()
# id_file_dict = dict()
# for image in images:
#     image_id = image["id"]
#     file_name = image["file_name"]
#     id_file_dict[image_id] = file_name
#
# for annotation in annotations:
#     image_id = annotation["image_id"]
#     category_id = annotation["category_id"]
#     bbox = annotation["bbox"]
#     info = [category_id] + bbox
#
#     if new_dict.get(id_file_dict[image_id]) is None:
#         new_dict[id_file_dict[image_id]] = info
#     else:
#         new_dict[id_file_dict[image_id]] += info
#
#
# with open("/mnt/Jan_data/pycharm_file/myyolo/Udacity_all.json","w") as f:
#     json.dump(new_dict,f)

#
# test_dict=dict()
# a_zip = random.sample(data.items(),3000)
# for a in a_zip:
#     k,v = a[0], a[1]
#     test_dict[str(k)] = v
#     data.pop(str(k))
#
# with open("/mnt/Jan_data/pycharm_file/myyolo/Udacity_train.json","w") as f:
#     json.dump(data,f)
# with open("/mnt/Jan_data/pycharm_file/myyolo/Udacity_test.json","w") as f:
#     json.dump(test_dict,f)

# json_path = "/mnt/Jan_data/pycharm_file/myyolo/Udacity_train.json"
# with open(json_path, 'r') as f:
#     data = json.load(f)
#
# png_dir = "/mnt/Jan_data/Data/Object_detetction_data/small_coco/pngs/export"
# test_dir = "/mnt/Jan_data/Data/Object_detetction_data/small_coco/pngs/train"
# for key, value in data.items():
#     png_path = os.path.join(png_dir,key)
#     dst_path = os.path.join(test_dir,key)
#     shutil.copy(png_path,dst_path)