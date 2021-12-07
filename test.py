import cv2
from skimage.feature import hog, local_binary_pattern


cap = cv2.VideoCapture(f"data/test_1.mp4")
real_data = cv2.VideoCapture("data/vsl/cropped/D0001T.mp4")

hog_features = []
hog_labels = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        resized_frame = cv2.resize(frame, (224, 244))
        blur_frame = cv2.blur(resized_frame, (5, 5))
        fd, hog_frame = hog(blur_frame, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), block_norm='L2', visualize=True, multichannel=True)
        # cv2.imshow('croped', hog_frame)
        hog_labels.append(1)
        hog_features.append(fd)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print(hog_frame)
        #     print(fd)
        #     break
    else:
        break
# cap.release()
# cv2.destroyAllWindows()

"""
    Duyệt các dir. 
    Duyệt các file. 
    Ném feature dimension vào mảng feature dim, ném label vào đó.
    
"""

print(len(hog_labels))
print(len(hog_features))
