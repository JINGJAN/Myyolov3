from network.main_network import ModelMain
from network.loss import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __name__ == "__main__":
    writer = SummaryWriter()
    write_log = True
    image_size = (384, 384)
    batch_size = 5
    test_batch_size = 1
    workers = 1
    ngpu = 1
    lr = 1e-5
    epochs = 200
    class_num = 12
    anchors = [[[116, 90], [156, 198], [373, 326]],
               [[30, 61], [62, 45], [59, 119]],
               [[10, 13], [16, 30], [33, 23]]]
    train_json_path = "/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/train.json"
    train_png_path = "/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/train_/train"
    test_json_path = "/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/test.json"
    test_png_path = "/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/test_/test"
    weights_path = '/mnt/Jan_data/pycharm_file/myyolo/train_yolov3_2000_2021-11-19_17_3/checkpoints/000100_g.model'
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Using {}".format(device))

    if write_log:
        log_folder = logfile("yolov3_2000")

    net = ModelMain()
    test_net = ModelMain()
    net.to(device)
    net.apply(weights_init)
    # # load weights
    # net.load_state_dict(
    #     torch.load(weights_path))

    dataloader = mydataloder(train_json_path, image_size,
                             batch_size=batch_size, workers=workers, shuffle_=True,
                             dir_=train_png_path)
    test_dataloader = mydataloder(test_json_path, image_size,
                                  batch_size=test_batch_size, workers=workers, shuffle_=False,
                                  dir_=test_png_path)
    optimizer_net = optim.Adam(net.parameters(), lr=lr, betas=(0, 0.999))

    yolo_loss = []
    iters = 0
    test_iter = 0
    for i in range(3):
        yolo_loss.append(YoloLoss(anchors[i], device, image_size))

    for epoch in range(epochs):
        print('Epoch :{}'.format(epoch))
        for data in tqdm(dataloader):
            optimizer_net.zero_grad()
            img, label = data

            img = img.to(device)
            outs = net(img)
            losses_name = ["total_loss", "x_loss", "y_loss", "w_loss", "h_loss", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i, out in enumerate(outs):
                _loss_item = yolo_loss[i](out, label)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(one_loss) for one_loss in losses]
            loss_ = losses[0]
            loss_.backward()
            optimizer_net.step()
            if iters % 10 == 0:
                writer.add_scalar("total_loss", loss_.item(), iters)
                for i in range(1, len(losses_name)):
                    writer.add_scalar(losses_name[i], losses[i], iters)

            iters += 1
        if (epoch + 1) % 2 == 0 or epoch == 0:
            try:
                torch.save(net.state_dict(), f'{log_folder}/checkpoints/{str(epoch + 1).zfill(6)}_g.model')
            except:
                pass
        # calculate the recall and precision
        if (epoch + 1) % 2 == 0 or epoch == 0:
            test_net.load_state_dict(torch.load(f'{log_folder}/checkpoints/{str(epoch + 1).zfill(6)}_g.model',
                                                map_location=torch.device('cpu')))

            true_positive = 0
            false_positive = 0
            false_negative = 0
            precision_ls = []
            recall_ls = []
            with open(test_json_path, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    false_negative += int(len(v) / 5)

            for test_data in test_dataloader:

                test_img, test_label = test_data
                test_img = test_img
                # test_label = test_label.to(device)

                # out len = 3
                # out[0].shape = (bs,3*(5+12),13,13)
                test_out = test_net(test_img)
                predictions = []
                for i, out_ in enumerate(test_out):
                    predictions.append(yolo_loss[i](out_.to("cpu"), None))
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
                    final_test_box_info[:, 2], final_test_box_info[:, 3] = final_test_box_info[:,
                                                                           0] + final_test_box_info[:,
                                                                                2], \
                                                                           final_test_box_info[:,
                                                                           1] + final_test_box_info[:, 3]

                    if detect is None:
                        false_positive += final_test_box_info.size(0)
                        test_iter += 1
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
                        writer.add_scalar("Precision", prc.item(), test_iter)
                        writer.add_scalar("Recall", rc.item(), test_iter)
                        test_iter += 1

            recall_ls = [x.cpu().detach() for x in recall_ls]
            precision_ls = [x.cpu().detach() for x in precision_ls]
            recall_ls = np.array(recall_ls)
            precision_ls = np.array(precision_ls)

            print("Epoch:{} mAP:{}".format(epoch, np.trapz(precision_ls, recall_ls)))
            writer.flush()
