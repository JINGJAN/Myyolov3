from network.main_network import ModelMain
from network.Iou_nms import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from network.loss import *
import argparse


# Parmeter
# class_num = 12
# image_size = (384, 384)#(512, 512) #(416, 416)
# batch_size = 2
# workers = 2
# test_json_path = "/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/test.json"
# test_png_path = "/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/test_/test"
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# net.to(device)
# net.load_state_dict(
#     torch.load('/mnt/Jan_data/pycharm_file/myyolo/train_yolov3_2000_2021-11-22_11_34/checkpoints/000200_g.model'))
def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPU')
    parser.add_argument('--image-size', type=int, default=416, help='Size of input image')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--class-num', type=int, default=12, help='class number')
    parser.add_argument('--worker', type=int, default=2, help='worker number of pytorch image loader')
    parser.add_argument('--json-path', type=str,
                        default="/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/test.json",
                        help='the json absolute path')
    parser.add_argument('--png-path', type=str,
                        default="/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/test_/test",
                        help='img dir absolute path')
    parser.add_argument('--weight-path', type=str,
                        default='/mnt/Jan_data/pycharm_file/myyolo/train_yolov3_split_2021-11-15_17_30/checkpoints/000006_g.model',
                        help='weights absolute path')
    parser.add_argument('--parameter-path', type=str,
                        default='/mnt/Jan_data/pycharm_file/myyolo/parmeter.json',
                        help='parameter absolute path')
    args = parser.parse_args()
    return args


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def calculate_fn(test_json_path):
    false_negative = 0
    with open(test_json_path, 'r') as f:
        data = json.load(f)
        for k, v in data.items():
            false_negative += int(len(v) / 5)
    return false_negative


def calculate_bbox_size(test_json_path):
    out_dict = {"S": 0,
                "M": 0,
                "L": 0}
    length_ = 5
    w_index = 3
    h_index = 4
    S = 32 * 32
    L = 96 * 96
    with open(test_json_path, 'r') as f:
        data = json.load(f)
        for k, v in data.items():
            for i in range(int(len(v) / length_)):
                w = v[length_ * i + w_index]
                h = v[length_ * i + h_index]
                if w * h <= S:
                    out_dict["S"] += 1
                if S < w * h <= L:
                    out_dict["M"] += 1
                if w * h >= L:
                    out_dict["L"] += 1

    return out_dict


def bbox_size(bbox):
    S = 32 * 32
    L = 96 * 96
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    area = (y2 - y1) * (x2 - x1)
    if area <= S:
        return 0
    if S < area <= L:
        return 1
    else:
        return 2


def build_size_dict():
    out_dict = {
        "true_positive": 0,
        "false_positive": 0,
        "S_true_positive": 0,
        "S_false_positive": 0,
        "M_true_positive": 0,
        "M_false_positive": 0,
        "L_true_positive": 0,
        "L_false_positive": 0
    }
    return out_dict


def calculate_tp_fp(parm):
    image_size = (parm.image_size, parm.image_size)
    tp_fp_dict = build_size_dict()

    #
    net = ModelMain()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and parm.ngpu > 0) else "cpu")
    print(device)
    net.to(device)
    net.load_state_dict(torch.load(parm.weight_path))
    #
    test_dataloader = mydataloder(parm.json_path, image_size,
                                  batch_size=parm.batch_size, workers=parm.worker, shuffle_=False,
                                  dir_=parm.png_path)
    #
    anchors = read_json(parm.parameter_path)["anchors"]
    #
    yolo_loss = []
    for i in range(3):
        yolo_loss.append(YoloLoss(anchors[i], device, image_size))

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
        detection = non_max_suppression(predictions, parm.class_num)
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

                tp_fp_dict["true_positive"] += torch.sum(iou_ >= 0.5)
                tp_fp_dict["false_positive"] += 1 - torch.sum(iou_ >= 0.5)

                if bbox_size(dt[:4]) == 0:
                    tp_fp_dict["S_true_positive"] += torch.sum(iou_ >= 0.5)
                    tp_fp_dict["S_false_positive"] += 1 - torch.sum(iou_ >= 0.5)
                if bbox_size(dt[:4]) == 1:
                    tp_fp_dict["M_true_positive"] += torch.sum(iou_ >= 0.5)
                    tp_fp_dict["M_false_positive"] += 1 - torch.sum(iou_ >= 0.5)
                if bbox_size(dt[:4]) == 2:
                    tp_fp_dict["L_true_positive"] += torch.sum(iou_ >= 0.5)
                    tp_fp_dict["L_false_positive"] += 1 - torch.sum(iou_ >= 0.5)

    return tp_fp_dict


if __name__ == "__main__":
    parm = parameter_parser()
    tp_fp_dict = calculate_tp_fp(parm)
    false_negative = calculate_fn(parm.json_path)
    fn_diff_size = calculate_bbox_size(parm.json_path)
    print(fn_diff_size)

    precision = tp_fp_dict['true_positive'] / (tp_fp_dict['true_positive'] + tp_fp_dict['false_positive'])
    recall = tp_fp_dict['true_positive'] / false_negative

    S_precision = tp_fp_dict['S_true_positive']/(tp_fp_dict['S_true_positive'] + tp_fp_dict['S_false_positive'])
    S_recall = tp_fp_dict['S_true_positive'] / fn_diff_size["S"]

    M_precision = tp_fp_dict['M_true_positive']/(tp_fp_dict['M_true_positive'] + tp_fp_dict['M_false_positive'])
    M_recall = tp_fp_dict['M_true_positive'] / fn_diff_size["M"]

    L_precision = tp_fp_dict['L_true_positive']/(tp_fp_dict['L_true_positive'] + tp_fp_dict['L_false_positive'])
    L_recall = tp_fp_dict['L_true_positive'] / fn_diff_size["L"]
    print(precision,recall)
    print(S_precision, S_recall)
    print(M_precision, M_recall)
    print(L_precision, L_recall)



