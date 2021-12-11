import json

import cv2

from network.main_network import ModelMain
import torch
import torchvision
import torchvision.transforms as transforms
from network.loss import *
import torchvision.utils as vutils
from network.Iou_nms import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

#416,416
#precision 0.561855673789978
#recall 0.3900
#mAP 0.2184465
#18.11it/s
# S tensor(0.4673, device='cuda:0') tensor(0.3800, device='cuda:0')
# M tensor(0.7120, device='cuda:0') tensor(0.4126, device='cuda:0')
# L tensor(0.8936, device='cuda:0') tensor(0.3471, device='cuda:0')

#512,512
#precision 0.6949
#recall 0.4210
#mAP 0.28561777
#14.92it/s
# S tensor(0.5747, device='cuda:0') tensor(0.3257, device='cuda:0')
# M tensor(0.8061, device='cuda:0') tensor(0.5370, device='cuda:0')
# L tensor(0.9155, device='cuda:0') tensor(0.5372, device='cuda:0')

#384,384
#precision 0.5385321378707886
#recall 0.3500
#mAP 0.18376184
#20.75it/s
# S tensor(0.4474, device='cuda:0') tensor(0.3648, device='cuda:0')
# M tensor(0.7252, device='cuda:0') tensor(0.3449, device='cuda:0')
# L tensor(0.8649, device='cuda:0') tensor(0.2645, device='cuda:0')

np.random.seed(0)

net = ModelMain()
ngpu = 1
class_num = 12
image_size = (416, 416)#(512, 512) #(416, 416)
batch_size = 2
workers = 2
test_json_path = "/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/test.json"
test_png_path = "/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/test_/test"
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
net.to(device)
net.load_state_dict(
    torch.load('/mnt/Jan_data/pycharm_file/myyolo/train_yolov3_split_2021-11-15_17_30/checkpoints/000006_g.model'))

# root = "/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/test_"
dict_classes = {
    0: 'Car', 1: 'Van', 2: 'DontCare',
    3: 'Pedestrian', 4: 'Truck', 5: 'Cyclist',
    6: 'Misc', 7: 'Tram', 8: 'Person_sitting'
}

# dataset = torchvision.datasets.ImageFolder(root, T)
# dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=3,
#                                          shuffle=False)
test_dataloader = mydataloder(test_json_path, image_size,
                              batch_size=batch_size, workers=workers, shuffle_=False,
                              dir_=test_png_path)
anchors = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]
yolo_loss = []
for i in range(3):
    yolo_loss.append(YoloLoss(anchors[i], device, image_size))

color_ = [(np.random.rand(3) * 255).astype(np.uint8) for x in range(20)]

true_positive = 0
false_positive = 0
false_negative = 0
precision_ls = []
recall_ls = []

with open(test_json_path,'r') as f:
    data = json.load(f)
    for k, v in data.items():
        false_negative += int(len(v) / 5)


for test_data in tqdm(test_dataloader):
    # label (100,5)
    test_img, test_label = test_data
    test_img = test_img.to(device)
    test_label = test_label.to(device)

    # out len = 3
    # out[0].shape = (bs,3*(5+12),13,13)
    test_out = net(test_img)
    predictions = []
    for i, out_ in enumerate(test_out):
        predictions.append(yolo_loss[i](out_, None))
    predictions = torch.cat(predictions, dim=1)
    detection = non_max_suppression(predictions, class_num)
    # bs
    for j, detect in enumerate(detection):
        # j batch size
        label_len = 0
        j_labels = test_label[j]
        for j_label in j_labels:
            if torch.sum(j_label) == 0:
                continue
            label_len += 1

        final_test_box_info = j_labels[:label_len]
        final_test_box_info = final_test_box_info[:, :4]
        final_test_box_info[:, 2], final_test_box_info[:, 3] = final_test_box_info[:, 0] + final_test_box_info[:,
                                                                                           2], \
                                                               final_test_box_info[:, 1] + final_test_box_info[:, 3]

        if detect is None:
            continue
        for dt in detect:
            pr_bbox = dt[:4].unsqueeze(0)
            iou_ = IOU_(pr_bbox, final_test_box_info)

            true_positive += torch.sum(iou_ >= 0.5)
            false_positive += 1 - torch.sum(iou_ >= 0.5)
            false_negative -= torch.sum(iou_ >= 0.5)

            prc = true_positive / (true_positive + false_positive)
            rc = true_positive / (true_positive + false_negative)

            precision_ls.append(prc)
            recall_ls.append(rc)


print(precision_ls[-1].item())
print(recall_ls[-1].item())
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

recall_ls = [x.cpu().detach() for x in recall_ls]
precision_ls = [x.cpu().detach() for x in precision_ls]
recall_ls = np.array(recall_ls)

precision_ls = np.array(precision_ls)

plt.plot(recall_ls, precision_ls)

print(np.trapz(precision_ls, recall_ls))
plt.show()
# std = torch.ones_like(img) * 0.5
# mean = torch.ones_like(img) * 0.5
# img = ((img * std) + mean) * 255
# img = img.permute(0, 2, 3, 1).contiguous()
# img = img[0].cpu().detach().numpy().astype(np.uint8)
# img = img[:, :, ::-1].copy()
# for detect in detection[0]:
#     if detect is None:
#         continue
#     cup_detect = detect.cpu().detach().numpy()

#     x1, y1, x2, y2 = cup_detect[:4]
#     p = cup_detect[4]
#     cp = cup_detect[5]
#     class_ = cup_detect[6]
#     classcolor = (int(color_[int(class_)][0]), int(color_[int(class_)][1]),
#                   int(color_[int(class_)][2]))
#
#     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=classcolor)
#     cv2.putText(img, str(int(class_)) + " " + str(p * cp), (int(x1), int(y1) - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=classcolor, thickness=1)
#
# cv2.namedWindow("test", cv2.WINDOW_NORMAL)
# cv2.imshow("test", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
