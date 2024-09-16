#!/usr/bin/python3
import sys
import struct
import numpy

### input helper functions ###

## read images and labels from binary files, one by one (generator function) ##
def datareader(imagename,labelname):
    with open(imagename,"rb") as images, open(labelname,"rb") as labels:
        magic,length,w,h=struct.unpack('>IIII',images.read(4*4))
        magic,lengthL=struct.unpack('>II',labels.read(4*2))
        print("dataset of {} images of size {}x{}".format(length,w,h),file=sys.stderr)
        assert length == lengthL
        for i in range(length):
            yield numpy.frombuffer(images.read(w*h),dtype='uint8'),labels.read(1)[0]

### main loop ###
for image,label in datareader("train-images-idx3-ubyte","train-labels-idx1-ubyte"):
    # replace with real code
    print("image:",numpy.reshape(image,(28,28)))
    print("label:",label)
    break

