import os
import cv2
import numpy as np

import onnx
import onnxruntime as rt

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

IMAGE_SIZE = (75,75)
CLASS_DICT = {0:"b", 1:"r", 2:"w", 3:"y"}

def normalize(img):
    img = img.astype('float32')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img /= 255.0
    img -= mean
    img /= std

    return img

def preprocessing(img):
    # Preprocessing
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = normalize(img)
    x = np.expand_dims(np.transpose(x, (2, 0, 1)), 0)

    return x

def onnx_infer(sess, img):
    # Preprocessing
    x = preprocessing(img)

    # input_names = sess.get_inputs()[0].name
    res = sess.run(None, {'input' : x})

    res_plate = np.array(res[0][0])
    res_char = np.array(res[1][0]).round()
    top_1 = res_plate.argmax(0)

    return [CLASS_DICT[top_1], res_char]

def testing(onnx_path, test_dir="/content/lpc_data_v2/test/*/*.jpg"):
    labels = []
    for key in CLASS_DICT.keys():
        labels.append(CLASS_DICT[key])

    paths = glob.glob(test_dir)
    print(len(paths))

    plate_gt_list = []
    pred_list = []

    N = len(paths)

    # Initiate ONNX Inference session
    sess = rt.InferenceSession(onnx_path)

    for path in paths:
        label = os.path.split(os.path.split(path)[0])[-1]
        label = label#.upper()
        plate_gt_list.append(str(label))
        #print(label)

        img = cv2.imread(path)
        pred = onnx_infer(sess,img)
        pred_list.append(pred)
        #print(pred)

        if label != pred[0]:
            dst_dir = os.path.join("/content/output/test_run/plate_color/",label,pred[0])
            if os.path.exists(dst_dir) == False:
                os.makedirs(dst_dir)
            dst_path = os.path.join(dst_dir, path.split("/")[-1])
            shutil.copy(path, str(dst_dir))

        color_label = 0 if label in ["w","y"] else 1
        if color_label != pred[1]:
            dst_dir = os.path.join("/content/output/test_run/char_color/",str(color_label),str(pred[1]))
            if os.path.exists(dst_dir) == False:
                os.makedirs(dst_dir)
            dst_path = os.path.join(dst_dir, path.split("/")[-1])
            shutil.copy(path, str(dst_dir))

    del sess # Release memory allocated to ONNX session

    # Plate color result
    plate_gt_list = np.array(plate_gt_list)
    plate_pred_list = np.array(pred_list)[:,0]
    plate_acc = np.sum(plate_gt_list == plate_pred_list) / N

    plate_cfs_matrix = confusion_matrix(plate_gt_list, plate_pred_list, labels=labels)

    # Character color result
    char_gt_list = []
    for plate in plate_gt_list:
        if plate in ["w", "y"]: char_gt_list.append("0.0")
        else: char_gt_list.append("1.0")

    char_pred_list = np.array(pred_list)[:,1]
    print(len(char_gt_list), len(char_pred_list))
    char_acc = np.sum(char_gt_list == char_pred_list) / N
    print(char_acc)
    char_cfs_matrix = confusion_matrix(char_gt_list, char_pred_list, labels=["0.0", "1.0"])

    return plate_acc, plate_cfs_matrix, char_acc, char_cfs_matrix

def main():
    onnx_path = "/content/lpc_multitask_pytorch.onnx"

    plate_acc, plate_cfs_matrix, char_acc, char_cfs_matrix = testing(onnx_path=onnx_path)

    print("Plate Color Accuracy:", plate_acc)
    print("Character Color Accuracy:", char_acc)

    labels = []
    for key in CLASS_DICT.keys():
        labels.append(str(CLASS_DICT[key]))

    #Plot the confusion matrix.

    ## Number of sample
    sns.heatmap(plate_cfs_matrix,
                annot=True,
                fmt='g',
                xticklabels=labels,
                yticklabels=labels)
    plt.ylabel('Actual',fontsize=13)
    plt.xlabel('Prediction',fontsize=13)
    plt.title('PLATE COLOR Confusion Matrix (Number of sample)',fontsize=17)
    plt.savefig("/content/plate_cfs_matrix_count.png")
    plt.show()

    cmn = plate_cfs_matrix.astype('float') / plate_cfs_matrix.sum(axis=1)[:, np.newaxis]

    ## Percentage
    sns.heatmap(cmn,
                annot=True,
                fmt='.3f',
                xticklabels=labels,
                yticklabels=labels)
    plt.ylabel('Actual',fontsize=13)
    plt.xlabel('Prediction',fontsize=13)
    plt.title('PLATE COLOR Confusion Matrix (Percentage)',fontsize=17)
    plt.savefig("/content/plate_cfs_matrix_percentage.png")
    plt.show()

    #================================

    #Plot the CHARACTER confusion matrix.

    ## Number of sample
    sns.heatmap(char_cfs_matrix,
                annot=True,
                fmt='g',
                xticklabels=["black", "white"],
                yticklabels=["black", "white"])
    plt.ylabel('Actual',fontsize=13)
    plt.xlabel('Prediction',fontsize=13)
    plt.title('CHARACTER COLOR Confusion Matrix (Number of sample)',fontsize=17)
    plt.savefig("/content/character_cfs_matrix_count.png")
    plt.show()

    cmn = char_cfs_matrix.astype('float') / char_cfs_matrix.sum(axis=1)[:, np.newaxis]

    # Percentage
    sns.heatmap(cmn,
                annot=True,
                fmt='.3f',
                xticklabels=["black", "white"],
                yticklabels=["black", "white"])
    plt.ylabel('Actual',fontsize=13)
    plt.xlabel('Prediction',fontsize=13)
    plt.title('CHARACTER COLOR Confusion Matrix (Percentage)',fontsize=17)
    plt.savefig("/content/character_cfs_matrix_percentage.png")
    plt.show()

if __name__ == '__main__':
    main()
