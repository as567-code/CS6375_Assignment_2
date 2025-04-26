def compute_loss(output, pred_box, gt_box, gt_mask, num_boxes, num_classes, grid_size, image_size):
    batch_size = output.shape[0]
    num_grids = output.shape[2]
    # compute mask with shape (batch_size, num_boxes, 7, 7) for box assignment
    box_mask = torch.zeros(batch_size, num_boxes, num_grids, num_grids)
    box_confidence = torch.zeros(batch_size, num_boxes, num_grids, num_grids)

    # compute assignment of predicted bounding boxes for ground truth bounding boxes
    for i in range(batch_size):
        for j in range(num_grids):
            for k in range(num_grids):
 
                # if the gt mask is 1
                if gt_mask[i, j, k] > 0:
                    # transform gt box
                    gt = gt_box[i, :, j, k].clone()
                    gt[0] = gt[0] * grid_size + k * grid_size
                    gt[1] = gt[1] * grid_size + j * grid_size
                    gt[2] = gt[2] * image_size
                    gt[3] = gt[3] * image_size
                    # print('gt in loss %.2f, %.2f, %.2f, %.2f' % (gt[0], gt[1], gt[2], gt[3]))

                    select = 0
                    max_iou = -1
                    # select the one with maximum IoU
                    for b in range(num_boxes):
                        # center x, y and width, height
                        pred = pred_box[i, 5*b:5*b+4, j, k].clone()
                        iou = compute_iou(gt, pred)
                        if iou > max_iou:
                            max_iou = iou
                            select = b
                    box_mask[i, select, j, k] = 1
                    box_confidence[i, select, j, k] = max_iou
                    print('select box %d with iou %.2f' % (select, max_iou))

    # compute yolo loss
    weight_coord = 5.0
    weight_noobj = 0.5

    # Loss on x coordinate (cx)
    loss_x = weight_coord * torch.sum(box_mask * torch.pow(gt_box[:, 0].unsqueeze(1) - output[:, 0:5*num_boxes:5], 2.0))
    
    # Loss on y coordinate (cy)
    loss_y = weight_coord * torch.sum(box_mask * torch.pow(gt_box[:, 1].unsqueeze(1) - output[:, 1:5*num_boxes:5], 2.0))
    
    # Loss on width (w)
    loss_w = weight_coord * torch.sum(box_mask * torch.pow(torch.sqrt(gt_box[:, 2].unsqueeze(1)) - torch.sqrt(output[:, 2:5*num_boxes:5]), 2.0))
    
    # Loss on height (h)
    loss_h = weight_coord * torch.sum(box_mask * torch.pow(torch.sqrt(gt_box[:, 3].unsqueeze(1)) - torch.sqrt(output[:, 3:5*num_boxes:5]), 2.0))
    
    # Loss on object confidence (already implemented)
    loss_obj = torch.sum(box_mask * torch.pow(box_confidence - output[:, 4:5*num_boxes:5], 2.0))
    
    # Loss on non-object confidence
    loss_noobj = weight_noobj * torch.sum((1 - box_mask) * torch.pow(0 - output[:, 4:5*num_boxes:5], 2.0))
    
    # Loss on class prediction (for multi-class detection, here we have just one class)
    box_cls_mask = torch.sum(box_mask, dim=1)
    box_cls_mask = box_cls_mask > 0
    loss_cls = torch.sum(box_cls_mask * torch.pow(1 - output[:, 5*num_boxes:], 2.0))
    
    print('lx: %.4f, ly: %.4f, lw: %.4f, lh: %.4f, lobj: %.4f, lnoobj: %.4f, lcls: %.4f' % 
          (loss_x, loss_y, loss_w, loss_h, loss_obj, loss_noobj, loss_cls))

    # the total loss
    loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_cls
    return loss