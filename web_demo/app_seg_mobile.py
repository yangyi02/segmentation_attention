import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import sys

pycaffe_root = "../code/python" # substitute your path here
sys.path.insert(0, pycaffe_root)
import caffe

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index_mobile.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index_mobile.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result, image_show, seg, overlay, seg2, overlay2 = app.clf.classify_image(image)
    return flask.render_template(
        'index_mobile.html', has_result=True, result=result,
        imagesrc=imageurl, resultsrc=embed_image_html(seg), overlaysrc=embed_image_html(overlay),
        resultsrc2=embed_image_html(seg2), overlaysrc2=embed_image_html(overlay2))


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index_mobile.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result, image_show, seg, overlay, seg2, overlay2 = app.clf.classify_image(image)
    return flask.render_template(
        'index_mobile.html', has_result=True, result=result,
        imagesrc=embed_image_html(image_show), resultsrc=embed_image_html(seg), overlaysrc=embed_image_html(overlay),
        resultsrc2=embed_image_html(seg2), overlaysrc2=embed_image_html(overlay2))


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    #image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/web_demo/models/part_seg/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/web_demo/models/part_seg/train2_iter_8000.caffemodel'.format(REPO_DIRNAME)),
        'model_def_file2': (
            '{}/web_demo/models/yi_cloth_2/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file2': (
            '{}/web_demo/models/yi_cloth_2/train_iter_12000.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/code/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, model_def_file2, pretrained_model_file2, mean_file,
                 raw_scale, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        caffe.set_device(1)

        self.model_def_file = model_def_file
        self.pretrained_model_file = pretrained_model_file
        self.model_def_file2 = model_def_file2
        self.pretrained_model_file2 = pretrained_model_file2
        self.mean_file = mean_file
        self.raw_scale = raw_scale
        self.image_dim = image_dim

        self.net = caffe.Classifier(
            self.model_def_file, self.pretrained_model_file,
            image_dims=(self.image_dim, self.image_dim), raw_scale=self.raw_scale,
            mean=np.load(self.mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

    def seg2img(self, seg):
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

    def seg2img2(self, seg):
        #colormaps for cloth segmentation
        result = np.zeros([seg.shape[0], seg.shape[1], 3])
        colormap = np.zeros([6, 3])
        colormap[0, :] = [0, 0, 0]
        colormap[1, :] = [1, 1, 0.25]
        colormap[2, :] = [0.25, 0.25, 1]
        colormap[3, :] = [1, 0.25, 0.25]
        colormap[4, :] = [1, 0.25, 1]
        colormap[5, :] = [0.25, 1, 1]
        for i in np.arange(seg.shape[0]):
            for j in np.arange(seg.shape[1]):
                result[i, j, 0] = colormap[seg[i, j], 0]
                result[i, j, 1] = colormap[seg[i, j], 1]
                result[i, j, 2] = colormap[seg[i, j], 2]
        return result

    def classify_image(self, image):
        try:
            # initialize network
            self.net = caffe.Classifier(
                self.model_def_file, self.pretrained_model_file,
                image_dims=(self.image_dim, self.image_dim), raw_scale=self.raw_scale,
                mean=np.load(self.mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
            )

            # resize image and pad zeros to make rectangular image square
            in_dims = self.net.transformer.inputs[self.net.inputs[0]][2:]
            image2 = image
            h_ratio = np.float32(in_dims[0]) / np.float32(image2.shape[0])
            w_ratio = np.float32(in_dims[1]) / np.float32(image2.shape[1])
            ratio = np.min([h_ratio, w_ratio])
            if ratio == h_ratio:
                h_new = in_dims[0]
                w_new = np.round(ratio * image2.shape[1]).astype(np.int)
            elif ratio == w_ratio:
                h_new = np.round(ratio * image2.shape[0]).astype(np.int)
                w_new = in_dims[1]
            image2 = caffe.io.resize_image(image2, [h_new, w_new])
            image3 = np.zeros([in_dims[0], in_dims[1], 3]) # image3 will be the image used for prediction
            channel_swap = self.net.transformer.channel_swap.get(self.net.inputs[0])
            mean = np.zeros(3)
            for i in np.arange(3):
                mean[channel_swap[i]] = self.net.transformer.mean.get(self.net.inputs[0])[i]
            image3[0:h_new, 0:w_new, :] = image2
            if h_new < in_dims[0]:
                image3[h_new:in_dims[0], :, :] = mean / self.net.transformer.raw_scale.get(self.net.inputs[0])
            elif w_new < in_dims[1]:
                image3[:, w_new:in_dims[1], :] = mean / self.net.transformer.raw_scale.get(self.net.inputs[0])

            # predict part segmentation
            starttime = time.time()
            caffe_in = self.net.transformer.preprocess(self.net.inputs[0], image3)
            out = self.net.forward_all(**{self.net.inputs[0]: np.asarray([caffe_in])})
            scores = np.squeeze(out[self.net.outputs[0]])
            endtime = time.time()

            # get part segmentation results
            seg = np.argmax(scores, axis=0)
            seg = self.seg2img(seg)
            seg = seg[0:h_new, 0:w_new, :]

            # place part segmentation results on original image for display purpose
            im = caffe.io.resize_image(image, seg.shape[:2])
            overlay = seg * 0.5 + im * 0.5

            # Above is for part segmentation
            # Now run cloth segmentation, same steps
            self.net = caffe.Classifier(
                self.model_def_file2, self.pretrained_model_file2,
                image_dims=(self.image_dim, self.image_dim), raw_scale=self.raw_scale,
                mean=np.load(self.mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
            )

            # predict cloth segmentation
            starttime2 = time.time()
            caffe_in2 = self.net.transformer.preprocess(self.net.inputs[0], image3)
            out2 = self.net.forward_all(**{self.net.inputs[0]: np.asarray([caffe_in2])})
            scores2 = np.squeeze(out2[self.net.outputs[0]])
            endtime2 = time.time()

            # get cloth segmentation results
            seg2 = np.argmax(scores2, axis=0)
            seg2 = self.seg2img2(seg2)
            seg2 = seg2[0:h_new, 0:w_new, :]

            # place cloth segmentation on original image for display purpose
            overlay2 = seg2 * 0.5 + im * 0.5

            return (True, '%.3f' % (endtime - starttime), '%.3f' % (endtime2 - starttime2)), image2, seg, overlay, seg2, overlay2

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
