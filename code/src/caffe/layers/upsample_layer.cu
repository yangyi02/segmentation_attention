#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void UpsampleForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int upsampled_height, const int upsampled_width,
    const int upsample_ratio, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int hstart = h * upsample_ratio;
    int wstart = w * upsample_ratio;
    int hend = hstart + upsample_ratio;
    int wend = wstart + upsample_ratio;
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, upsampled_height);
    wend = min(wend, upsampled_width);
    top_data += (n * channels + c) * upsampled_height * upsampled_width;
    for (int uh = hstart; uh < hend; ++uh) {
      for (int uw = wstart; uw < wend; ++uw) {
        top_data[uh * upsampled_width + uw] = bottom_data[index];
      }
    }
  }
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  UpsampleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_,
      height_, width_, upsampled_height_, upsampled_width_, upsample_ratio_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void UpsampleBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int upsampled_height, const int upsampled_width,
    const int upsample_ratio, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int hstart = h * upsample_ratio;
    int wstart = w * upsample_ratio;
    int hend = hstart + upsample_ratio;
    int wend = wstart + upsample_ratio;
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, upsampled_height);
    wend = min(wend, upsampled_width);
    top_diff += (n * channels + c) * upsampled_height * upsampled_width;
    for (int uh = hstart; uh < hend; ++uh) {
      for (int uw = wstart; uw < wend; ++uw) {
        bottom_diff[index] += top_diff[uh * upsampled_width + uw];
      }
    }
  }
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  UpsampleBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, top[0]->num(), channels_,
      height_, width_, upsampled_height_, upsampled_width_, upsample_ratio_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(UpsampleLayer);


}  // namespace caffe
