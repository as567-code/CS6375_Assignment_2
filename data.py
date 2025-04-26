def __getitem__(self, idx):
    
    # gt file
    filename_gt = self.gt_paths[idx]
    
    # Get image file path (replace the -box.txt with .jpg)
    filename_image = filename_gt.replace('-box.txt', '.jpg')
    
    # Load image using OpenCV
    image = cv2.imread(filename_image)
    
    # Resize image to YOLO size (448x448)
    image = cv2.resize(image, (self.yolo_image_size, self.yolo_image_size))
    
    # Normalize pixels: subtract mean and divide by 255
    image = image.astype(np.float32) - self.pixel_mean
    image = image / 255.0
    
    # Convert from (H,W,C) to (C,H,W) format for PyTorch
    image = image.transpose((2, 0, 1))
    image_blob = torch.from_numpy(image)
    
    # Initialize ground truth tensors
    gt_box_blob = torch.zeros(5, self.yolo_grid_num, self.yolo_grid_num)
    gt_mask_blob = torch.zeros(self.yolo_grid_num, self.yolo_grid_num)
    
    # Load ground truth bounding box (x1, y1, x2, y2)
    with open(filename_gt, 'r') as f:
        box = f.readline().strip().split()
        x1 = float(box[0]) * self.scale_width
        y1 = float(box[1]) * self.scale_height
        x2 = float(box[2]) * self.scale_width
        y2 = float(box[3]) * self.scale_height
    
    # Calculate center coordinates and width/height
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    # Determine which grid cell the center falls into
    grid_x = int(cx / self.yolo_grid_size)
    grid_y = int(cy / self.yolo_grid_size)
    
    # Set the mask for this grid cell
    gt_mask_blob[grid_y, grid_x] = 1
    
    # Normalize box coordinates for the grid cell
    # cx, cy: offset from the top-left corner of the grid cell, normalized to [0,1]
    cx_norm = (cx - grid_x * self.yolo_grid_size) / self.yolo_grid_size
    cy_norm = (cy - grid_y * self.yolo_grid_size) / self.yolo_grid_size
    
    # w, h: normalized by the image size to [0,1]
    w_norm = w / self.yolo_image_size
    h_norm = h / self.yolo_image_size
    
    # Store normalized values in the grid cell
    gt_box_blob[0, grid_y, grid_x] = cx_norm
    gt_box_blob[1, grid_y, grid_x] = cy_norm
    gt_box_blob[2, grid_y, grid_x] = w_norm
    gt_box_blob[3, grid_y, grid_x] = h_norm
    gt_box_blob[4, grid_y, grid_x] = 1.0  # confidence is 1 for ground truth
    
    # Return the sample dictionary
    sample = {'image': image_blob,
              'gt_box': gt_box_blob,
              'gt_mask': gt_mask_blob}
    
    return sample