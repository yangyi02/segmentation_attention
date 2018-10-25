#include <iostream>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  for (int i = 0;
       i < this->layer_param_.loss_param().ignore_label_size();
       ++i) {
    ignore_label_.insert(
        this->layer_param_.loss_param().ignore_label(i));
  }
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_WEIGHTED_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  CHECK_GE(bottom[0]->num(), 1);
  positive_weights_.Reshape(bottom[0]->num(), 1, 1, 1);
  negative_weights_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Assume
  // bottom[0]: prediction
  // bottom[1]: ground truth
  // bottom[2]: positive loss weight (if provided)
  if (bottom.size() < 3) {
    for (int n = 0; n < bottom[0]->num(); ++n) {
      positive_weights_.mutable_cpu_data()[n] = Dtype(0.5);
      negative_weights_.mutable_cpu_data()[n] = Dtype(0.5);
    }
  } else {
    const Dtype* w = bottom[2]->cpu_data();
    for (int n = 0; n < bottom[2]->num(); ++n) {
      positive_weights_.mutable_cpu_data()[n] = w[n];
      negative_weights_.mutable_cpu_data()[n] = 1 - w[n];
    }
  }
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / bottom[0]->num();
  const int bottom_channel = bottom[0]->channels();

  CHECK(bottom_channel == 1)
      << "Current implemenation assumes bottom channle = 1.";
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* pos_weight = positive_weights_.cpu_data();
  const Dtype* neg_weight = negative_weights_.cpu_data();
  Dtype loss = 0;
  Dtype batch_weight = 0;

  for (int n = 0; n < num; ++n) {
    for (int i = 0; i < dim; ++i) {
      int pos = n * dim + i;
      const int label_value = static_cast<int>(target[pos]);
      if (ignore_label_.count(label_value) != 0) {
        continue;
      }
      CHECK(label_value == 0 || label_value == 1) <<
          "Current implementation only allows target = 0 or 1.";
      Dtype cur_loss;
      if (input_data[pos] >= 0) {
        cur_loss =
         input_data[pos] * neg_weight[n] * (target[pos] - 1)
         + (neg_weight[n] - pos_weight[n]) * target[pos]
            * log(1 + exp(-input_data[pos]))
         - neg_weight[n] * log(1 + exp(-input_data[pos]));
      } else {
        cur_loss =
         input_data[pos] * pos_weight[n] * target[pos]
         + (neg_weight[n] - pos_weight[n]) * target[pos]
          * log(1 + exp(input_data[pos]))
         - neg_weight[n] * log(1 + exp(input_data[pos]));
      }
      loss -= cur_loss;
      if (target[pos]) {
        batch_weight += pos_weight[n];
      } else {
        batch_weight += neg_weight[n];
      }
    }
  }
  if (batch_weight == 0) {
    batch_weight = 1;
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / batch_weight;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / num;
  }
}

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (bottom.size() > 2) {
    if (propagate_down[2]) {
      LOG(FATAL) << this->type()
                 << " Layer cannot backpropagate to loss weight.";
    }
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / bottom[0]->num();
    const Dtype* pred   = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* pos_weight = positive_weights_.cpu_data();
    const Dtype* neg_weight = negative_weights_.cpu_data();
    Dtype batch_weight = 0;
    for (int n = 0; n < num; ++n) {
      for (int i = 0; i < dim; ++i) {
        int pos = n * dim + i;
        const int label_value = static_cast<int>(target[pos]);
        if (ignore_label_.count(label_value) != 0) {
          bottom_diff[pos] = 0;
        } else {
          bottom_diff[pos] =
           neg_weight[n] * pred[pos] * (1 - target[pos])
           - pos_weight[n] * target[pos] * (1 - pred[pos]);

          if (target[pos]) {
            batch_weight += pos_weight[n];
          } else {
            batch_weight += neg_weight[n];
          }
        }
      }
    }
    // Scale gradient
    if (batch_weight == 0) {
      batch_weight = 1;
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
     caffe_scal(count, loss_weight / batch_weight, bottom_diff);
    } else {
      caffe_scal(count, loss_weight / num, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidWeightedCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidWeightedCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidWeightedCrossEntropyLoss);

}  // namespace caffe
