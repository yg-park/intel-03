def crop_images(bgr_image, resized_image, boxes, threshold=0.6) -> np.ndarray:
    """
    Use bounding boxes from detection model to find the absolute car position
    
    :param: bgr_image: raw image
    :param: resized_image: resized image
    :param: boxes: detection model returns rectangle position
    :param: threshold: confidence threshold
    :returns: car_position: car's absolute position
    """
    # Fetch image shapes to calculate ratio
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Find the boxes ratio
    boxes = boxes[:, 1:]   # 1  원래 2인데 만약 1로 수정한다면 나머지 부분들에 대해서 적당히 수정해서 실행하시오. 정상적으로 나와야 한다.
    print(boxes)           # 0이 아니라 1부터 출력된다. [image_id, label, conf, x_min, y_min, x_max, y_max] 순서에 주목.
    # Store the vehicle's position
    car_position = []
    # Iterate through non-zero boxes
    for box in boxes:
        # Pick confidence factor from last place in array
        conf = box[1]                                         # 0 -> 1
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio
            # In case that bounding box is found at the top of the image, 
            # upper box  bar should be positioned a little bit lower to make it visible on image 
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y * resized_y, 10)) if idx % 2 
                else int(corner_position * ratio_x * resized_x)
                for idx, corner_position in enumerate(box[2:])             #1 -> 2
            ]
            
            car_position.append([x_min, y_min, x_max, y_max])
            
    return car_position
