
# import numpy as np
# import cv2
# import os
# from image_processing import func

# if not os.path.exists("data2"):
#     os.makedirs("data2")

# path = "dataSet/train"
# path1 = "data2"
# a = ["label"]

# for i in range(64 * 64):
#     a.append("pixel" + str(i))

# label = 0
# var = 0
# c1 = 0
# c2 = 0

# for dirpath, dirnames, filenames in os.walk(path):
#     for dirname in dirnames:
#         print(dirname)
#         for direcpath, direcnames, files in os.walk(path + "/" + dirname):
#             if not os.path.exists(path1 + "/trainingData/" + dirname):
#                 os.makedirs(path1 + "/train/" + dirname)
#             if not os.path.exists(path1 + "/testingData/" + dirname):
#                 os.makedirs(path1 + "/testingData/" + dirname)
#             num = int(0.75 * len(files))  # Number of files for training
#             i = 0
#             for file in files:
#                 var += 1
#                 actual_path = os.path.join(path, dirname, file)
#                 actual_path1 = os.path.join(path1, "trainingData", dirname, file)
#                 actual_path2 = os.path.join(path1, "testingData", dirname, file)
#                 img = cv2.imread(actual_path, 0)
#                 bw_image = func(actual_path)
#                 if i < num:
#                     c1 += 1
#                     cv2.imwrite(actual_path1, bw_image)
#                 else:
#                     c2 += 1
#                     cv2.imwrite(actual_path2, bw_image)

#                 i += 1

#         label += 1

# print("Total images processed:", var)
# print("Total training images:", c1)
# print("Total testing images:", c2)
