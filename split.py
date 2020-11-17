import cv2
import os

if __name__ == "__main__":

    path = "/home/michelexie/桌面/AU/RAF-AU/basic/Image/aligned/"  # change this dirpath.
    listdir = os.listdir(path)
    newdir='./full_dataset/'

    # newdir = os.path.join(path_, 'split_right')  # make a new dir in dirpath.
    if (os.path.exists(newdir) == False):
        os.mkdir(newdir)


    for i in listdir:
        if i.split('.')[1] == "jpg":  # the format of img.
            filepath = os.path.join(path, i)
            filename = i.split('.')[0]
            leftpath = os.path.join(newdir, filename) + "_left.jpg"
            rightpath = os.path.join(newdir, filename) + "_right.jpg"

            img = cv2.imread(filepath)

            [h, w] = img.shape[:2]
            print(filepath, (h, w))

            limg = img[:, :int(w / 2), :]
            rimg = img[:, int(w / 2 + 1):, :]

            cv2.imwrite(leftpath, limg)
            cv2.imwrite(rightpath, rimg)

