import cv2
import numpy as np
from scipy.special import expit


def yolo_head(predictions, num_classes, input_dims):
    """
    YOLO Head to process predictions from Darknet

    :param num_classes: Total number of classes
    :param input_dims: Input dimensions of the image
    :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N,38, 38, 255) and (N, 76, 76, 255)
    :return: A tensor with the shape (N, num_boxes, 85)
    """

    anchors = [
        [[116, 90], [156, 198], [373, 326]],
        [[30, 61], [62, 45], [59, 119]],
        [[10, 13], [16, 30], [33, 23]]
    ]

    tiny_anchors = [
        [[81, 82], [135, 169], [344, 319]],
        [[10, 14], [23, 27], [37, 58]]
    ]
    results = []

    if len(predictions) == 3: # assume 3 set of predictions is YOLOv3
        model_anchors = anchors
    elif len(predictions) == 2: # 2 set of predictions is YOLOv3-tiny
        model_anchors = tiny_anchors
    else:
        raise ValueError('Unsupported prediction length: {}'.format(len(predictions)))

    for i, prediction in enumerate(predictions):
        results.append(_yolo_head(prediction, num_classes, model_anchors[i], input_dims))

    return np.concatenate(results, axis=1)


def _yolo_head(prediction, num_classes, anchors, input_dims):
    batch_size = np.shape(prediction)[0]
    stride = input_dims[0] // np.shape(prediction)[1]
    grid_size = input_dims[0] // stride
    num_anchors = len(anchors)

    prediction = np.reshape(prediction,
                            (batch_size, num_anchors * grid_size * grid_size, num_classes + 5))

    box_xy = expit(prediction[:, :, :2])  # t_x (box x and y coordinates)
    objectness = expit(prediction[:, :, 4])  # p_o (objectness score)
    objectness = np.expand_dims(objectness, 2)  # To make the same number of values for axis 0 and 1

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = np.reshape(a, (-1, 1))
    y_offset = np.reshape(b, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)

    box_xy += x_y_offset

    # Log space transform of the height and width
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    anchors = np.tile(anchors, (grid_size * grid_size, 1))
    anchors = np.expand_dims(anchors, 0)

    box_wh = np.exp(prediction[:, :, 2:4]) * anchors

    # Sigmoid class scores
    class_scores = expit(prediction[:, :, 5:])

    # Resize detection map back to the input image size
    box_xy *= stride
    box_wh *= stride

    # Convert centoids to top left coordinates
    box_xy -= box_wh / 2

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def predict(model, image_arr, num_classes, model_image_size, confidence=0.5, iou_threshold=0.4):
    image, image_data = preprocess_image(image_arr, model_image_size)

    predictions = yolo_head(model.predict([image_data]), num_classes, input_dims=model_image_size)

    boxes, classes, scores = handle_predictions(predictions,
                                                confidence=confidence,
                                                iou_threshold=iou_threshold)
    boxes = adjust_boxes(boxes, image_arr, model_image_size)

    return boxes, classes, scores


def handle_predictions(predictions, confidence=0.6, iou_threshold=0.5):
    boxes = predictions[:, :, :4]
    box_confidences = np.expand_dims(predictions[:, :, 4], -1)
    box_class_probs = predictions[:, :, 5:]

    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= confidence)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    # Boxes, Classes and Scores returned from NMS
    n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores, iou_threshold)

    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes)
        scores = np.concatenate(n_scores)

        return boxes, classes, scores

    else:
        return [], [], []


def preprocess_image(img_arr, model_image_size):
    image = img_arr.astype('uint8')
    resized_image = cv2.resize(image, tuple(reversed(model_image_size)), cv2.INTER_AREA)
    image_data = resized_image.astype('float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data


def nms_boxes(boxes, classes, scores, iou_threshold):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        x = b[:, 0]
        y = b[:, 1]
        w = b[:, 2]
        h = b[:, 3]

        areas = w * h
        order = s.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
    return nboxes, nclasses, nscores


def adjust_boxes(boxes, image, model_image_size):
    if boxes is None or len(boxes) == 0:
        return []

    height, width = image.shape[:2]
    adjusted_boxes = []

    ratio_x = width / model_image_size[1]
    ratio_y = height / model_image_size[0]

    for box in boxes:
        x, y, w, h = box

        # Rescale box coordinates
        xmin = int(x * ratio_x)
        ymin = int(y * ratio_y)
        xmax = int((x + w) * ratio_x)
        ymax = int((y + h) * ratio_y)

        ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
        xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
        ymax = min(height, np.floor(ymax + 0.5).astype('int32'))
        xmax = min(width, np.floor(xmax + 0.5).astype('int32'))
        adjusted_boxes.append([xmin,ymin,xmax,ymax])

    return np.array(adjusted_boxes)


def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)

    return image

def draw_boxes(image, boxes, classes, scores, class_names, colors):
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box

        predicted_class = class_names[cls]
        label = '{} {:.2f}'.format(predicted_class, score)
        print(label, (xmin, ymin), (xmax, ymax))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[cls], 1, cv2.LINE_AA)
        image = draw_label(image, label, colors[cls], (xmin, ymin))

    return image