#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ResizeForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int resized_height, const int resized_width,
    const Dtype resize_ratio, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int rw = index % resized_width;
    int rh = (index / resized_width) % resized_height;
    int c = (index / resized_width / resized_height) % channels;
    int n = index / resized_width / resized_height / channels;
    int h = int(rh / resize_ratio);
    int w = int(rw / resize_ratio);
    bottom_data += (n * channels + c) * height * width;
    top_data[index] = bottom_data[h * width + w];
  }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ResizeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_,
      height_, width_, resized_height_, resized_width_, resize_ratio_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ResizeBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int resized_height, const int resized_width,
    const Dtype resize_ratio, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int hstart = int(h * resize_ratio);
    int wstart = int(w * resize_ratio);
    int hend = int(hstart + resize_ratio);
    int wend = int(wstart + resize_ratio);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, resized_height);
    wend = min(wend, resized_width);
    top_diff += (n * channels + c) * resized_height * resized_width;
    for (int rh = hstart; rh < hend; ++rh) {
      for (int rw = wstart; rw < wend; ++rw) {
        bottom_diff[index] += top_diff[rh * resized_width + rw];
      }
    }
  }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  ResizeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, top[0]->num(), channels_,
      height_, width_, resized_height_, resized_width_, resize_ratio_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);


}  // namespace caffe
