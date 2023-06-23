import argparse

import os
import glob

import cv2
import numpy as np

import onnxruntime as rt

INPUT_SHAPE = (75,75)
CLASS_DICT = {0:"b", 1:"r", 2:"w", 3:"y"}

def parse_args():
    parser = argparse.ArgumentParser(description='ONNX Inference configuration')
    parser.add_argument('--model_path', default=None, help='ONNX model file path', type=str)
    parser.add_argument('--input_dir', default=False, help='Image directory', type=str)
    parser.add_argument('--input_shape', default=75, help='Input shape', type=int)

    args = parser.parse_args()

    return args

def normalize(img):
    img = img.astype('float32')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img /= 255.0
    img -= mean
    img /= std

    return img

def preprocessing(img, input_shape=INPUT_SHAPE):
    # Preprocessing
    img = cv2.resize(img, input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = normalize(img)
    x = np.expand_dims(np.transpose(x, (2, 0, 1)), 0)

    return x

def onnx_infer(sess, img):
    # Preprocessing
    x = preprocessing(img)

    # input_names = sess.get_inputs()[0].name
    res = sess.run(None, {'input' : x})
    res = np.array(res[0][0])

    idx = res.argmax(0)
    
    return idx

def main():
    args = parse_args()

    onnx_path = args.model_path
    input_dir = os.path.join(args.input_dir, "*.jpg")
    # print(input_dir)
    paths = glob.glob(input_dir)
    print(len(paths))

    pred_list = []

    N = len(paths)

    # Initiate ONNX Inference session
    sess = rt.InferenceSession(onnx_path)

    for path in paths:
        img = cv2.imread(path)
        pred = onnx_infer(sess,img)

        print(path, CLASS_DICT[pred], "\n")
        #pred_list.append(pred)

    del sess # Release memory allocated to ONNX session

if __name__ == '__main__':
    main()