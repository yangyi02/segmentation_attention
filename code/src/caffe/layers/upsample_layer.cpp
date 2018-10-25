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
void UpsampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UpsampleParameter upsample_param = this->layer_param_.upsample_param();
  CHECK(upsample_param.has_upsample_ratio())
      << "Upsample ratio (A positive integer) is required.";
  upsample_ratio_ = static_cast<int>(upsample_param.upsample_ratio());
  CHECK_GT(upsample_ratio_, 0);
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  upsampled_height_ = static_cast<int>(height_ * upsample_ratio_);
  upsampled_width_ = static_cast<int>(width_ * upsample_ratio_);
  top[0]->Reshape(bottom[0]->num(), channels_, upsampled_height_,
      upsampled_width_);
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          int hstart = h * upsample_ratio_;
          int wstart = w * upsample_ratio_;
          int hend = hstart + upsample_ratio_;
          int wend = wstart + upsample_ratio_;
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, upsampled_height_);
          wend = min(wend, upsampled_width_);
          for (int uh = hstart; uh < hend; ++uh) {
            for (int uw = wstart; uw < wend; ++uw) {
              top_data[uh * upsampled_width_ + uw] = 
                bottom_data[h * width_ + w];
            }
          }
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
          int hstart = h * upsample_ratio_;
          int wstart = w * upsample_ratio_;
          int hend = hstart + upsample_ratio_;
          int wend = wstart + upsample_ratio_;
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, upsampled_height_);
          wend = min(wend, upsampled_width_);
          for (int uh = hstart; uh < hend; ++uh) {
            for (int uw = wstart; uw < wend; ++uw) {
              bottom_diff[h * width_ + w] +=
                top_diff[uh * upsampled_width_ + uw];
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
STUB_GPU(UpsampleLayer);
#endif

INSTANTIATE_CLASS(UpsampleLayer);
REGISTER_LAYER_CLASS(Upsample);

}  // namespace caffe
