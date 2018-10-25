#!/usr/bin/python
import os
import sys
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from matplotlib import pyplot
import scipy
from scipy import misc
import Image

if __name__ == "__main__":
  if len(sys.argv) < 6:
    print "Usage: *.py protoxt caffemodel mean.binaryproto input_image output_seg"
    sys.exit(-1)
  else:
    proto_file = sys.argv[1]
    model_file = sys.argv[2]
    mean_file = sys.argv[3]
    image_file = sys.argv[4]
    seg_file = sys.argv[5]

  caffe.set_mode_gpu()
  # load net
  net = caffe.Net(proto_file, model_file, caffe.TEST)
  # load mean
  blob = caffe.io.caffe_pb2.BlobProto()
  data = open(mean_file, 'rb').read()
  blob.ParseFromString(data)
  nparray = caffe.io.blobproto_to_array(blob)
  img_mean = np.squeeze(nparray)
  img_mean = img_mean.mean(1).mean(1)
  print img_mean
  #np.save('fashionista_mean.npy', img_mean)
  #net.set_mean('data', np.load('fashionista_mean.npy')i)
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2,0,1))
  transformer.set_mean('data', img_mean)
  transformer.set_raw_scale('data', 255.0)
  transformer.set_channel_swap('data', (2,1,0))

  # load image
  im = caffe.io.load_image(image_file)
  height, width, dummy = im.shape
  # run network
  out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
  # save results
  prob = np.squeeze(out['fc8_interp'])
  seg = np.argmax(prob, axis=0)
  seg = scipy.misc.toimage(seg, cmin=0, cmax=255)
  seg = scipy.misc.imresize(seg, (height, width), 'nearest')
  seg = scipy.misc.toimage(seg, cmin=0, cmax=255)
  seg.save(seg_file)
  #pyplot.imsave(seg_file, seg)
  #seg = Image.fromarray(seg)
  #seg.save(seg_file)
