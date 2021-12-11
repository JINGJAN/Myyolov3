import math
from network.Iou_nms import *


class YoloLoss(nn.Module):
    def __init__(self, anchors, device, size):
        super(YoloLoss, self).__init__()
        self.img_size = size
        self.anchors = anchors  # (3)
        self.device = device
        self.num_classes = 12
        self.bbox_attrs = 5 + self.num_classes
        self.num_anchors = len(self.anchors)
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_noobj = 0.5

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, images, annotations):

        batch_size = images.size(0)
        in_h = images.size(2)
        in_w = images.size(3)

        stride_h = self.img_size[0] / in_h
        stride_w = self.img_size[1] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = images.view(batch_size, self.num_anchors,
                                 self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x (bs, 3,in_h,in_w)
        y = torch.sigmoid(prediction[..., 1])  # Center y (bs, 3,in_h,in_w)
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.(bs,3,in_h,in_w,class_num)

        if annotations is not None:
            #  build target
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.getTarget(annotations, scaled_anchors,
                                                                           in_w, in_h)
            mask, noobj_mask = mask.to(self.device), noobj_mask.to(self.device)
            tx, ty, tw, th = tx.to(self.device), ty.to(self.device), tw.to(self.device), th.to(self.device)
            tconf, tcls = tconf.to(self.device), tcls.to(self.device)
            #  losses.
            loss_x = self.bce_loss(x * mask, tx * mask)
            loss_y = self.bce_loss(y * mask, ty * mask)
            loss_w = self.mse_loss(w * mask, tw * mask)
            loss_h = self.mse_loss(h * mask, th * mask)
            loss_conf = self.bce_loss(conf * mask, mask) + \
                        self.lambda_noobj * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])
            #  total loss = losses * weight
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                   loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                   loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
                   loss_h.item(), loss_conf.item(), loss_cls.item()
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            # (start, start + 1*((in_w-1 -0)/ (in_w-1)))
            # 13,13->bs*3,13,13
            grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
                batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
                batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, in_h * in_w).view(h.shape)

            pred_boxes = FloatTensor(prediction[..., :4].shape)  # (bs, 3,in_h,in_w, 4)
            pred_boxes[..., 0] = (x + grid_x) * stride_w
            pred_boxes[..., 1] = (y + grid_y) * stride_h
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w * stride_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h * stride_h

            output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            # output (bs, 3*in_h*in_w, 5+12)
            return output

    def getTarget(self, loc_infos, anchors, in_h, in_w):
        '''

        :param loc_infos:
        :param anchors:
        :param in_h: feature map height
        :param in_w: feature map width
        :return:
        '''
        # loc_infos(batch_size, n_object, 4((top_left),(right_bottom),class)
        bs = loc_infos.size(0)
        stride_h = self.img_size[0] / in_h
        stride_w = self.img_size[1] / in_w

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        for b in range(loc_infos.size(0)):
            for n in range(loc_infos.size(1)):
                if torch.sum(loc_infos[b, n, :]) == 0:
                    continue
                x_tl = loc_infos[b, n, 0] / stride_w
                y_tl = loc_infos[b, n, 1] / stride_h

                class_ = loc_infos[b, n, 4]

                gw = loc_infos[b, n, 2] / stride_w
                gh = loc_infos[b, n, 3] / stride_h
                gx = x_tl + gw / 2
                gy = y_tl + gh / 2

                gi = int(gx)
                gj = int(gy)

                # gt_box (1,4)/anchor_shape(3,4)
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # (3,1)
                ious = IOU_(gt_box, anchor_shapes)
                best_iou = np.argmax(ious)

                mask[b, best_iou, gj, gi] = 1
                noobj_mask[b, ious > 0.5, gj, gi] = 0

                tx[b, best_iou, gj, gi] = gx - gi
                ty[b, best_iou, gj, gi] = gy - gj

                th[b, best_iou, gj, gi] = math.log(gh / anchors[best_iou][1] + 1e-16)
                tw[b, best_iou, gj, gi] = math.log(gw / anchors[best_iou][0] + 1e-16)

                tconf[b, best_iou, gj, gi] = 1

                tcls[b, best_iou, gj, gi, int(class_)] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls
