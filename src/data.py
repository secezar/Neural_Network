import os
import cv2
import numpy as np

class Data:
    def __init__(self, data=[]):
        self.data = data
        self.labels = []

    def load_dir_dataset(self, dir_path):
        for path in os.listdir(dir_path):
            self.data.append(self.load(os.path.join(dir_path, path)))
            self.labels.append(int(path[0]))
        return np.array(self.data), np.array(self.labels)

    def load(self, path):
        return cv2.imread(path)

    def load_x_dataset(self, array_path):
        assert array_path[-4:] == ".npy", "{}".format(array_path[-4:])
        self.data = np.load(array_path)

    def load_y_dataset(self, array_path):
        assert array_path[-4:] == ".npy", "{}".format(array_path[-4:])
        self.labels = np.load(array_path)

    def save(self):
        np.save("data", np.array(self.data))
        np.save("labels", np.array(self.labels))

    def show_sample(self, index):
        print(self.data[index].shape)
        cv2.imshow("Sample", self.data[index])
        cv2.waitKey()
