// Copyright Liang-Chieh Chen
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
void SpatialPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), 1) <<
      "current implementation assumes num = 1.";
  CHECK_EQ(bottom[1]->num(), 1) <<
      "current implementation assumes num = 1.";

  SpatialPoolingParameter spatial_pooling_param =
      this->layer_param_.spatial_pooling_param();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  real_height_ = *bottom[1]->cpu_data();
  real_width_ = *(bottom[1]->cpu_data() + 1);
  CHECK_GT(height_, 0) << "Input dimensions cannot be zero.";
  CHECK_GT(width_, 0) << "Input dimensions cannot be zero.";

  num_bin_ = spatial_pooling_param.num_bin();
  kernel_h_ = floor(real_height_ / static_cast<double>(num_bin_));
  kernel_w_ = floor(real_width_ / static_cast<double>(num_bin_));
  pad_h_ = 0;
  pad_w_ = 0;
  stride_h_ = kernel_h_;
  stride_w_ = kernel_w_;
}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), 1) <<
      "current implementation assumes num = 1.";
  CHECK_EQ(bottom[1]->num(), 1) <<
      "current implementation assumes num = 1.";

  SpatialPoolingParameter spatial_pooling_param =
      this->layer_param_.spatial_pooling_param();
  height_ = bottom[0]->height();
  width_  = bottom[0]->width();
  real_height_ = *bottom[1]->cpu_data();
  real_width_ = *(bottom[1]->cpu_data() + 1);
  CHECK_GT(height_, 0) << "Input dimensions cannot be zero.";
  CHECK_GT(width_, 0) << "Input dimensions cannot be zero.";

  // if not set, use bottome dimensions
  if (real_height_ == 0) {
    real_height_ = height_;
  }
  if (real_width_ == 0) {
    real_width_ = width_;
  }

  kernel_h_ = floor(real_height_ / static_cast<double>(num_bin_));
  kernel_w_ = floor(real_width_ / static_cast<double>(num_bin_));
  stride_h_ = kernel_h_;
  stride_w_ = kernel_w_;
  channels_ = bottom[0]->channels();
  pad_h_ = 0;
  pad_w_ = 0;
  pooled_height_ = num_bin_;
  pooled_width_ = num_bin_;
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
}



template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.spatial_pooling_param().pool()) {
    case SpatialPoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, real_height_);
            int wend = min(wstart + kernel_w_, real_width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case SpatialPoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, real_height_ + pad_h_);
            int wend = min(wstart + kernel_w_, real_width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, real_height_);
            wend = min(wend, real_width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void SpatialPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_CLASS(SpatialPoolingLayer);
REGISTER_LAYER_CLASS(SpatialPooling);

}  // namespace caffe
