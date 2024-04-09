import cv2
import matplotlib.pyplot as plt
import os
from utils import *
from darknet import Darknet

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# cfg_file = './cfg/yolov7.cfg'
weight_file = './weights/yolov7.weights'
cfg_file = './cfg/yolov7.cfg'
namesfile = 'data/coco.names'

m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)
m.print_network()

plt.rcParams['figure.figsize'] = [24.0, 14.0]

img = cv2.imread('./images/crowd1.jpg')

original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(original_image, (m.width, m.height))

nms_thresh = 0.6
iou_thresh = 0.4

boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
print_objects(boxes, class_names)
boxed_image = plot_boxes(original_image, boxes, class_names, plot_labels=False)

num_objects = len(boxes)

print("Number of objects detected:", num_objects)

plt.figure(figsize=(12, 12))
plt.imshow(boxed_image)
plt.axis('off')
plt.show()