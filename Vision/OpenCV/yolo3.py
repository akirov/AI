import cv2 as cv
import numpy as np
import os
import sys


# Get weights with: `wget https://pjreddie.com/media/files/yolov3.weights`
# Get config and labels from https://github.com/pjreddie/darknet
# Specify the correct paths below
HOME = os.getenv('HOME')
model  = HOME + '/.yolo/yolov3.weights'
config = HOME + '/.yolo/yolov3.cfg'
labels = HOME + '/.yolo/yolov3.txt'

net = cv.dnn.readNetFromDarknet(config, model)

classes = None
with open(labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
if classes is None:
    exit(1)

colors = np.random.uniform(128, 255, size=(len(classes), 3))  # different random color for each class

if len(sys.argv) < 2:
    print("Please enter image URI as an argument")
    exit(1)

# Read image(s)
img_uri = sys.argv[1]
#print(os.path.exists(img_uri))

image = cv.imread(img_uri)
if image is None:
    print("Couldn't read the image ", img_uri)
    exit(1)

width = image.shape[1]
height = image.shape[0]
ch = image.shape[2]

# Input size must be (320,320), (416,416) or (608, 608)
blob = cv.dnn.blobFromImage(image, 1.0/255.0, (416,416), swapRB=True, crop=False)  # or use letterbox?

net.setInput(blob)


# YOLOv3: An Incremental Improvement - Joseph Redmon, 2018:
# "YOLOv3 predicts boxes at 3 different scales.  Our system extracts features from
# those scales using a similar concept to feature pyramid networks. From our base
# feature extractor we add several convolutional layers. The last of these predicts
# a 3-d tensor encoding bounding box, objectness,  and  class  predictions.
# In our experiments with COCO we predict 3 boxes at each scale so the tensor is
# NxNx[3*(4 + 1 + 80)] for the 4 bounding box offsets, 1 objectness prediction,
# and 80 class predictions."
# So, we have 3 output layers N1, N2 and N3 (for each scale). At each scale YOLOv3
# predicts 3 boxes for any of the grid cells.

# Get the 3 output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[l[0] - 1] for l in net.getUnconnectedOutLayers()]

all_predictions = net.forward(output_layers)  # Forward pass through the network and gather predictions from output layers
#for p in all_predictions:  # predictions is a list of 3 arrays (for each scale)
#    print("YOLO prediction shape : ", p.shape)  # (num predicted bounding boxes, 85=(4 + 1 + 80))
box_conf_idx = 4
class_prob_idx = 5

# Minimum confidence threshold. Increasing this will improve false positives but
# will also reduce detection rate.
# By default, YOLO only displays objects detected with a confidence of .25 or higher.
# Confidence is confidence_on_box * probability_on_class.
min_confidence = 0.25
min_prob_on_class = 0.25  # TODO make it a parameter
nms_threshold = 0.40  # or min_prob_on_class - 0.1 ?

# Prediction results
class_ids = []
class_probs = []
boxes = []

for predictions in all_predictions:  # Loop over scales
    # Loop over bounding boxes
    for prediction in predictions:
        confidence_on_box = prediction[box_conf_idx]
        class_prob_list = prediction[class_prob_idx:]

        #probability_on_class = np.amax(class_prob_list)
        best_class_index=class_prob_list.argmax(axis=0)  # or np.argmax(class_prob_list)
        probability_on_class = class_prob_list[best_class_index]
        #confidence = confidence_on_box * probability_on_class

        if probability_on_class > min_prob_on_class:  # or confidence > min_confidence
            x_center=int(prediction[0]*width)
            y_center=int(prediction[1]*height)
            width_box=int(prediction[2]*width)
            height_box=int(prediction[3]*height)

            x1=int(x_center-width_box / 2)
            y1=int(y_center-height_box / 2)
            #x2=int(x_center+width_box / 2)
            #y2=int(y_center+height_box / 2)

            class_ids.append(best_class_index)
            class_probs.append(float(probability_on_class))
            boxes.append([x1, y1, width_box, height_box])


# Apply non-max suppression to remove multiple bounding boxes
indices = cv.dnn.NMSBoxes(boxes, class_probs, min_prob_on_class, nms_threshold)


for il in indices:
    i = il[0]
    cls = classes[class_ids[i]]
    probability_on_class = class_probs[i]
    x1 = boxes[i][0]
    y1 = boxes[i][1]
    x2 = x1+boxes[i][2]
    y2 = y1+boxes[i][3]
    color = colors[class_ids[i]]
    print("class :", cls, " ,  Confidence :", "{0:.2f}".format(probability_on_class),
          " ,  ({:d} {:d}) ({:d} {:d})".format(x1,y1,x2,y2))
    cv.rectangle(image, (x1, y1), (x2, y2), color, 2)  # or color (0, 255, 0)
    cv.putText(image, cls + " " + "{0:.2f}".format(probability_on_class),
               (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)  # or (255, 255, 255) color

# Write the result in a file
img_new_uri = os.path.splitext(img_uri)[0] + "_y3py" + os.path.splitext(img_uri)[1]
cv.imwrite(img_new_uri, image)
print("Result written in ", img_new_uri)

cv.imshow(img_new_uri + '  Any key to close the window.',image)
if cv.waitKey(0) > 0 :
    cv.destroyAllWindows()
