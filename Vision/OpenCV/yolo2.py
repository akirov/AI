import cv2 as cv
import numpy as np
import os
import sys


# Get weights with: `wget https://pjreddie.com/media/files/yolov2.weights`
# Get config and labels from https://github.com/pjreddie/darknet
# Specify the correct paths below
HOME = os.getenv('HOME')
model  = HOME + '/.yolo/yolov2.weights'
config = HOME + '/.yolo/yolov2.cfg'
labels = HOME + '/.yolo/coco.names'

net = cv.dnn.readNetFromDarknet(config, model)

classes = None
with open(labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
print("Num classes = ", len(classes))

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

# Input size is (608, 608), not (416, 416) in the cfg file
blob = cv.dnn.blobFromImage(image, 1.0/255.0, (608,608), swapRB=True, crop=False)

net.setInput(blob)

predictions = net.forward()  # Make a prediction - forward pass through the network
print("YOLO prediction shape : ", predictions.shape)  # (num bounding boxes, 85)

# YOLO2's convolutional layers downsample the image by a factor of 32 so by
# using an input image of 416x416 we get an output feature map of 13×13 cells.
# Each cell is responsible for predicting 5 bounding boxes. So, the total number
# of bounding boxes on an 416x416 image is 13 x 13 x 5 = 845. Each detected
# object is specified by three attributes: a bounding box ([x_center, y_center,
# width, height]), confidence on box, and the probability over 80 classes in the
# COCO data. Totally 4+1+80=85 parameters.
box_conf_idx = 4
class_prob_idx = 5

# Minimum confidence threshold. Increasing this will improve false positives but
# will also reduce detection rate.
# By default, YOLO only displays objects detected with a confidence of .25 or higher.
# Confidence is confidence_on_box * probability_on_class.
min_confidence = 0.25

results = []

# Loop over bounding boxes
for i in range(predictions.shape[0]):
    confidence_on_box = predictions[i][box_conf_idx]
    class_prob_list = predictions[i][class_prob_idx:]
#    print("class_prob_list.shape : ", class_prob_list.shape)

    probability_on_class_old = np.amax(class_prob_list)
    class_index=class_prob_list.argmax(axis=0)
    probability_on_class = class_prob_list[class_index]
#    print("probability_on_class_old = ", probability_on_class_old, ", probability_on_class = ", probability_on_class)

    confidence = confidence_on_box * probability_on_class

    if probability_on_class > min_confidence:  # or confidence > ... ???
        x_center=predictions[i][0]*width
        y_center=predictions[i][1]*height
        width_box=predictions[i][2]*width
        height_box=predictions[i][3]*height
        
        x1=int(x_center-width_box * 0.5)
        y1=int(y_center-height_box * 0.5)
        x2=int(x_center+width_box * 0.5)
        y2=int(y_center+height_box * 0.5)
     
        cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),1)
        cv.putText(image, classes[class_index]+" "+"{0:.2f}".format(probability_on_class),
                   (x1,y1), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv.LINE_AA)
        results.append((classes[class_index], probability_on_class, x1, y1, x2, y2))


results.sort(key=lambda tup: tup[1], reverse=True)
for res in results:
    print("class :", res[0], " ,  Confidence :",
          "{0:.2f}".format(res[1]), " ,  ({:d} {:d}) ({:d} {:d})".format(res[2],res[3],res[4],res[5]))


# Write the result in a file
img_new_uri = os.path.splitext(img_uri)[0] + "_y2py" + os.path.splitext(img_uri)[1]
cv.imwrite(img_new_uri, image)
print("Result written in ", img_new_uri)


cv.imshow('image',image)
if cv.waitKey(0) > 0 :
    cv.destroyAllWindows()
