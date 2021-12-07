import cv2
from skimage.feature import hog
import os
import numpy as np
from sklearn import svm

# Data augmentation


def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (img.shape[0]//2, img.shape[1]//2)
        rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)

        dim = (width, height)
        return cv2.warpAffine(img, rotMat, dim)


def hog_my_frame(frame):
    fd, hog_frame = hog(frame, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), block_norm='L2', visualize=True, multichannel=True)

    return fd


# labels
labels = os.listdir('../data/vsl/my_data/train')

label_id = 1
hog_labels = []
hog_features = []
print("Start collect training data...")
for label in labels:
    print("Starting file dir: ", label)
    # print(os.listdir(f"../data/vsl/my_data/{label}"))
    videos = os.listdir(f"../data/vsl/my_data/train/{label}")
    for video in videos:
        print("Starting with video: ", video)
        vid = cv2.VideoCapture(f"../data/vsl/my_data/train/{label}/{video}")
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                resized_frame = cv2.resize(
                    frame, (224, 244), interpolation=cv2.INTER_AREA)

                # Data augmentation from here!
                blur_frame = cv2.GaussianBlur(
                    resized_frame, (5, 5), cv2.BORDER_DEFAULT)
                rotate_frame_1 = rotate(resized_frame, 45)
                rotate_frame_2 = rotate(resized_frame, 90)
                rotate_frame_3 = rotate(resized_frame, 135)
                rotate_frame_4 = rotate(resized_frame, 60)
                flip_1 = cv2.flip(resized_frame, 0)
                flip_2 = cv2.flip(resized_frame, -1)

                hog_labels.append(label_id)
                hog_features.append(hog_my_frame(blur_frame))

                hog_labels.append(label_id)
                hog_features.append(hog_my_frame(rotate_frame_1))

                hog_labels.append(label_id)
                hog_features.append(hog_my_frame(rotate_frame_2))

                hog_labels.append(label_id)
                hog_features.append(hog_my_frame(rotate_frame_3))

                hog_labels.append(label_id)
                hog_features.append(hog_my_frame(rotate_frame_4))

                hog_labels.append(label_id)
                hog_features.append(hog_my_frame(flip_1))

                hog_labels.append(label_id)
                hog_features.append(hog_my_frame(flip_2))
            else:
                print("Finish proccessing with video: ", video)
                break
    label_id += 1

print("Training Labels: ", len(hog_labels))
print("Training Features: ", len(hog_features))

np.save("train_labels", hog_labels)
np.save("train_features", hog_features)

count = 1
print("Start collect test data")
test_labels = []
test_features = []
for label in labels:
    print("Starting file dir: ", label)
    # print(os.listdir(f"../data/vsl/my_data/{label}"))
    videos = os.listdir(f"../data/vsl/my_data/test/{label}")
    for video in videos:
        print("Starting with video: ", video)
        vid = cv2.VideoCapture(f"../data/vsl/my_data/test/{label}/{video}")
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                resized_frame = cv2.resize(
                    frame, (224, 244), interpolation=cv2.INTER_AREA)
                # Data augmentation from here!
                blur_frame = cv2.GaussianBlur(
                    resized_frame, (5, 5), cv2.BORDER_DEFAULT)
                rotate_frame_1 = rotate(resized_frame, 45)
                rotate_frame_2 = rotate(resized_frame, 90)
                rotate_frame_3 = rotate(resized_frame, 135)
                rotate_frame_4 = rotate(resized_frame, 60)
                flip_1 = cv2.flip(resized_frame, 0)
                flip_2 = cv2.flip(resized_frame, -1)

                test_labels.append(count)
                test_features.append(hog_my_frame(blur_frame))

                test_labels.append(count)
                test_features.append(hog_my_frame(rotate_frame_1))

                test_labels.append(count)
                test_features.append(hog_my_frame(rotate_frame_2))

                test_labels.append(count)
                test_features.append(hog_my_frame(rotate_frame_3))

                test_labels.append(count)
                test_features.append(hog_my_frame(rotate_frame_4))

                test_labels.append(count)
                test_features.append(hog_my_frame(flip_1))

                test_labels.append(count)
                test_features.append(hog_my_frame(flip_2))
            else:
                print("Finish proccessing with video: ", video)
                break
    count += 1

np.save("test_labels", test_labels)
np.save("test_features", test_features)
