import sys

# for OpenCV 2.0
import cv2
import numpy as np
import argparse

# for Tensorflow 
import tensorflow as tf
import numpy as np
from PIL import Image


import glob
sys.path.append('svhn')
sys.path.append('svhn/svhn')
sys.path.append('thrift')

from svhn.svhn import SVHN, ttypes

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer


#Create new image(numpy array) filled with certain color in RGB

def create_blank(width, height, rgb_color=(0, 0, 0)):
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

# return tensorflow decoded image from queue
def read_image(filename_queue):
    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    return image

# return tensorflow decoded image from filename 
def inputs(filename):
    filenames = [filename, ]#'10.jpg', ]
    filename_queue = tf.train.string_input_producer(filenames)#,num_epochs=1)
    read_input = read_image(filename_queue)
    return read_input

# initialize weights
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# build CNN model
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 32, 32, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w = init_weights([3, 3, 3, 32])       # 3x3x3 conv, 32 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # 128 filter  4 * 4  Image
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

predict_op = tf.argmax(py_x, 1)

#####################################
sess = tf.Session()

init = tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, "sean_cnn_model3.model")


#####################################

# sorting contours
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

class SVHNHandler:
    def __init__(self):
        self.log = {}

    def recognize(self, query):
        result = ''
        print('filename : %s' % (query.filename))
        with open(query.filename, "wb") as f:
            f.write(query.image)


        ###########################################3
        # 1) Slice each digit using OpenCV
        ###########################################3
        original = cv2.imread(query.filename)
        im= original.copy()
        out = np.zeros(im.shape,np.uint8)
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
        
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contours, boundingBoxes = sort_contours(contours)

        idx = 0
        for cnt in contours:
            ca = cv2.contourArea(cnt)
            if ca >50:
                [x,y,w,h] = cv2.boundingRect(cnt)
                #print x, y, w, h
                #if  h>28:

                if  h>22:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                    roi = thresh[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)

                    # crop each contour 
                    cropped = original[y:y+h, x:x+w]
                    cropped = cv2.resize(cropped, (32, 32))
        
                    filename = str(idx) + '.jpg'
                    cv2.imwrite(filename, cropped)
        
                    #cv2.imwrite(str(idx) + '.jpg', original[y:y+h, x:x+w])
                    #cv2.imwrite(str(y) + "_" + str(x) + '.jpg', original[y:y+h, x:x+w])
                    idx += 1

                    ###########################################3
                    # 2) Digit recognition using tensorflow
                    ###########################################3
            
                    with sess.as_default():
                        image = inputs(filename)
                        coord = tf.train.Coordinator()
                        threads = tf.train.start_queue_runners(coord=coord)
                        
                        img = sess.run(image)
                        img = img.reshape(-1, 32, 32, 3)
                        
                        prd = sess.run(predict_op, feed_dict={X:img, p_keep_conv:1.0, p_keep_hidden:1.0})
                        result += str(prd[0])

                    #coord.request_stop()
                    #coord.join(threads)
    
        return result;

if __name__ == '__main__':
    handler = SVHNHandler()
    processor = SVHN.Processor(handler)
    transport = TSocket.TServerSocket(host="163.152.111.74", port=5425)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    
    print('Starting the server...')
    server.serve()
    print('done.')
