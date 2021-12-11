from torch import nn
from torch.utils.data import Dataset
from datetime import datetime
from network.img_loader import *


def IOU_(box1, box2):
    '''
    calculate the Intersection over Union
    :param box1: (1,4)
    :param box2: (3,4)
    :return:
    '''
    bx1_xtl, bx1_ytl, bx1_xbr, bx1_ybr = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    bx2_xtl, bx2_ytl, bx2_xbr, bx2_ybr = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # intersection
    x_tl = torch.max(bx1_xtl, bx2_xtl)
    y_tl = torch.max(bx1_ytl, bx2_ytl)
    x_br = torch.min(bx1_xbr, bx2_xbr)
    y_br = torch.min(bx1_ybr, bx2_ybr)

    intersection_area = torch.clamp(x_br - x_tl, min=0) * torch.clamp(y_br - y_tl, min=0)
    bx1_area = torch.clamp(bx1_xbr - bx1_xtl, min=0) * torch.clamp(bx1_ybr - bx1_ytl, min=0)
    bx2_area = torch.clamp(bx2_xbr - bx2_xtl, min=0) * torch.clamp(bx2_ybr - bx2_ytl, min=0)

    iou = intersection_area / (bx2_area + bx1_area - intersection_area + 1e-16)

    return iou


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def mydataloder(json_path, image_size, batch_size, workers, shuffle_, dir_):

    # data_set = Image_Data_loader(json_path=json_path, img_size=image_size)
    # data_set = Split_Image_loader(json_path=json_path, img_size=image_size)
    data_set = coco_Image_loader(json_path=json_path, img_size=image_size,
                                 png_dir=dir_)

    dataloader = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size,
                                             shuffle=shuffle_, num_workers=workers)
    return dataloader




def logfile(folder_name):
    date_time = datetime.now()
    log_folder = 'train_%s_%s_%d_%d' % (folder_name, date_time.date(), date_time.hour, date_time.minute)
    mkdir_(log_folder)
    mkdir_(os.path.join(log_folder, 'checkpoints'))
    mkdir_(os.path.join(log_folder, 'images'))
    return log_folder


def mkdir_(path_dir):
    try:
        os.mkdir(path=path_dir)
    except:
        pass


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # pr.shape(bs, 3*13*13+ 3*26*26 + 3*52*52, 5+12)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # image_pred (..., 5+12)

        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = IOU_(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections)
            # Add max detections to outputs
            # for different categories
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output

# dataloader = mydataloder("/mnt/Jan_data/myyolo/img_txt.json", (416, 416), batch_size=3, workers=2, shuffle_=True)
# for data in tqdm(dataloader):
#     img, label = data["image"], data["label"]
#     print(img.shape,label.shape)
#     for b in range(label.size(0)):
#         for n in range(label.size(1)):
#             if torch.sum(label[b,n]) ==0:
#                 print("yes")
#
#     break
