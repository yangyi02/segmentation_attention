#!/usr/bin/python
import os
import sys
pycaffe_root = "../code/python" # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe
import numpy as np
import matplotlib
from matplotlib import pyplot
import scipy
import time
import Image

def seg2img(seg):
    # colormaps for part segmentation
    result = np.zeros([seg.shape[0], seg.shape[1], 3])
    colormap = np.zeros([7, 3])
    colormap[0, :] = [0, 0, 0]
    colormap[1, :] = [1, 0, 0]
    colormap[2, :] = [0, 1, 0]
    colormap[3, :] = [1, 1, 0]
    colormap[4, :] = [0, 0, 1]
    colormap[5, :] = [1, 0, 1]
    colormap[6, :] = [0, 1, 1]
    for i in np.arange(seg.shape[0]):
        for j in np.arange(seg.shape[1]):
            result[i, j, 0] = colormap[seg[i, j], 0]
            result[i, j, 1] = colormap[seg[i, j], 1]
            result[i, j, 2] = colormap[seg[i, j], 2]
    return result

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print "Usage: *.py protoxt caffemodel mean.npy input_path output_path layer_name"
        sys.exit(-1)
    else:
        proto_file = sys.argv[1]
        model_file = sys.argv[2]
        mean_file = sys.argv[3]
        input_dir = sys.argv[4]
        output_dir = sys.argv[5]
        layer_name = sys.argv[6]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    caffe.set_mode_gpu()
    caffe.set_device(1)
    # load net
    net = caffe.Classifier(
        proto_file, model_file,
        image_dims=(256, 256), raw_scale=255,
        mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
    )
    in_dims = net.transformer.inputs[net.inputs[0]][2:]

    for root, dirs, files in os.walk(input_dir):
        for imgfile in files:
            if not imgfile.endswith('.jpg') and not imgfile.endswith('.png'):
                continue
            print os.path.join(input_dir, imgfile)
            # load image
            im = caffe.io.load_image(os.path.join(input_dir, imgfile))
            height, width, dummy = im.shape
            h_ratio = np.float32(in_dims[0]) / np.float32(height)
            w_ratio = np.float32(in_dims[1]) / np.float32(width)
            ratio = np.min([h_ratio, w_ratio])
            if ratio == h_ratio:
                h_new = in_dims[0]
                w_new = np.round(ratio * width).astype(np.int)
            elif ratio == w_ratio:
                h_new = np.round(ratio * height).astype(np.int)
                w_new = in_dims[1]
            image2 = caffe.io.resize_image(im, [h_new, w_new])
            # image3 will be the image used for prediction
            image3 = np.zeros([in_dims[0], in_dims[1], 3])
            channel_swap = net.transformer.channel_swap.get(net.inputs[0])
            mean = np.zeros(3)
            for i in np.arange(3):
                mean[channel_swap[i]] = net.transformer.mean.get(net.inputs[0])[i]
            image3[0:h_new, 0:w_new, :] = image2
            if h_new < in_dims[0]:
                image3[h_new:in_dims[0], :, :] = mean / net.transformer.raw_scale.get(net.inputs[0])
            elif w_new < in_dims[1]:
                image3[:, w_new:in_dims[1], :] = mean / net.transformer.raw_scale.get(net.inputs[0])
            # run network
            starttime = time.time()
            caffe_in = net.transformer.preprocess(net.inputs[0], image3)
            out = net.forward_all(**{net.inputs[0]: np.asarray([caffe_in])})
            prob = np.squeeze(out[layer_name])
            print time.time() - starttime
            # save results
            seg = np.argmax(prob, axis=0)
            seg = seg[0:h_new, 0:w_new]

            #seg = scipy.misc.toimage(seg, cmin=0, cmax=255)
            #seg = scipy.misc.imresize(seg, (height, width), 'nearest')
            #seg = scipy.misc.toimage(seg, cmin=0, cmax=255)
            #filename = imgfile.split('.')
            #filename = filename[0] + '.png'
            #seg.save(os.path.join(output_dir, filename))

            seg = seg2img(seg)
            im = caffe.io.resize_image(im, seg.shape[:2])
            seg = seg * 0.2 + im * 0.8
            seg = Image.fromarray((255 * seg).astype('uint8'))

            filename = imgfile.split('.')
            filename = filename[0] + '.png'
            seg.save(os.path.join(output_dir, filename))

