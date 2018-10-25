#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ResizeParameter resize_param = this->layer_param_.resize_param();
  CHECK(resize_param.has_resize_ratio()) << "Resize ratio (A positive number) is required.";
  resize_ratio_ = static_cast<Dtype>(resize_param.resize_ratio());
  CHECK_GT(resize_ratio_, 0);
}

template <typename Dtype>
void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  resized_height_ = static_cast<int>(height_ * resize_ratio_);
  resized_width_ = static_cast<int>(width_ * resize_ratio_);
  top[0]->Reshape(bottom[0]->num(), channels_, resized_height_,
      resized_width_);
}

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int rh = 0; rh < resized_height_; ++rh) {
        for (int rw = 0; rw < resized_width_; ++rw) {
          int h = int(rh / resize_ratio_);
          int w = int(rw / resize_ratio_);
          h = min(h, height_);
          w = min(w, width_);
          top_data[rh * resized_width_ + rw] = 
            bottom_data[h * width_ + w];
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          int hstart = int(h * resize_ratio_);
          int wstart = int(w * resize_ratio_);
          int hend = int(hstart + resize_ratio_);
          int wend = int(wstart + resize_ratio_);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, resized_height_);
          wend = min(wend, resized_width_);
          for (int rh = hstart; rh < hend; ++rh) {
            for (int rw = wstart; rw < wend; ++rw) {
              bottom_diff[h * width_ + w] +=
                top_diff[rh * resized_width_ + rw];
            }
          }
        }
      }
      // offset
      bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ResizeLayer);
#endif

INSTANTIATE_CLASS(ResizeLayer);
REGISTER_LAYER_CLASS(Resize);

}  // namespace caffe
