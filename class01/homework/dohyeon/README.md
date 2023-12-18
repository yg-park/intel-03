# 218번 boxes의 정보 출력 변경

    # Find the boxes ratio
    boxes = boxes[:, 1:]

    print(boxes)

    # Store the vehicle's position
    car_position = []
    # Iterate through non-zero boxes
    for box in boxes:
        # Pick confidence factor from last place in array
        conf = box[1]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio
            # In case that bounding box is found at the top of the image,
            # upper box  bar should be positioned a little bit lower to make it visible on image
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y * resized_y, 10)) if idx % 2
                else int(corner_position * ratio_x * resized_x)
                for idx, corner_position in enumerate(box[2:])
            ]

            car_position.append([x_min, y_min, x_max, y_max])

    return car_position
