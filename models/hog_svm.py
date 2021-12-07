import cv2
from skimage.feature import hog
import os
import numpy as np
from sklearn import svm

labels_dir = os.listdir('../data/vsl/my_data')
labels = []
for l in labels_dir:
    if l != "test":
        labels.append(l)
print(labels)
label_id = 1
hog_labels = []
hog_features = []
for label in labels:
    print("Starting file dir: ", label)
    # print(os.listdir(f"../data/vsl/my_data/{label}"))
    videos = os.listdir(f"../data/vsl/my_data/{label}")
    for video in videos:
        print("Starting with video: ", video)
        vid = cv2.VideoCapture(f"../data/vsl/my_data/{label}/{video}")
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                resized_frame = cv2.resize(frame, (224, 244))
                blur_frame = cv2.blur(resized_frame, (5, 5))
                fd, hog_frame = hog(blur_frame, orientations=8, pixels_per_cell=(16, 16),
                                    cells_per_block=(2, 2), block_norm='L2', visualize=True, multichannel=True)
                hog_labels.append(label_id)
                hog_features.append(fd)
            else:
                print("Finish proccessing with video: ", video)
                break
    label_id += 1

print("Training Labels: ", len(hog_labels))
print("Training Features: ", len(hog_features))

np.save("labels", hog_labels)
np.save("features", hog_features)

print("Start collect test data")
test_features = []
test_labels = []
test_dir = os.listdir("../data/vsl/my_data/test")
count = 1
for video in test_dir:
    print("Starting with video: ", video)
    vid = cv2.VideoCapture(f"../data/vsl/my_data/test/{video}")
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            resized_frame = cv2.resize(frame, (224, 244))
            blur_frame = cv2.blur(resized_frame, (5, 5))
            fd, hog_frame = hog(blur_frame, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(2, 2), block_norm='L2', visualize=True, multichannel=True)
            test_features.append(fd)
            if count == 1 or count == 3:
                test_labels.append(2)
            elif count == 2:
                test_labels.append(1)
            else:
                test_labels.append(3)
        else:
            print("Finish proccessing with video: ", video)
            count += 1
            break

np.save("test_labels", test_labels)
np.save("test_features", test_features)
