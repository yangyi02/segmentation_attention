#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_backward(const int num, const int dim,
            const Dtype* pred, const Dtype* target,
            Dtype* bottom_diff, Dtype* count_buffer,
            const Dtype* neg_weight,
            const Dtype* pos_weight, 
            const bool has_ignore_label, const int ignore_label) {
  CUDA_KERNEL_LOOP(index, num * dim) {
    int n = index / dim;
    int d = index % dim;
    int pos = n * dim + d;
    const int label_value = static_cast<int>(target[pos]);
    if (has_ignore_label && label_value == ignore_label) {
      bottom_diff[pos] = 0;
      count_buffer[pos] = 0;
    } else {
      bottom_diff[pos] = neg_weight[n] * pred[pos] * (1 - target[pos])
          - pos_weight[n] * target[pos] * (1 - pred[pos]);
      if (target[pos]) {
        count_buffer[pos] = pos_weight[n];
      } else {
        count_buffer[pos] = neg_weight[n];
      }
    }
  }
}

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Backward_gpu(
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
    const int nthreads = bottom[0]->count();
    const int count = bottom[0]->count();
    const int num   = bottom[0]->num();
    const int dim   = bottom[0]->count() / bottom[0]->num();
    const Dtype* pred = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* pos_weight = positive_weights_.gpu_data();
    const Dtype* neg_weight = negative_weights_.gpu_data();
    Dtype batch_weight = 0;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* count_buffer = sigmoid_output_->mutable_gpu_diff();
    
    const bool has_ignore_label = !ignore_label_.empty();
    int ignore_label = 0;
    if (has_ignore_label) {
      CHECK(ignore_label_.size() == 1)
          << "Current gpu implementation only takes one ignore label.";
      ignore_label = *ignore_label_.begin();
    }
    kernel_backward<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(num, dim, pred, target,
                    bottom_diff, count_buffer,
                    neg_weight, pos_weight, 
                    has_ignore_label, ignore_label);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_gpu_asum(nthreads, count_buffer, &batch_weight);
      if (batch_weight == 0) {
        batch_weight = 1;
      }
      caffe_gpu_scal(count, loss_weight / batch_weight, bottom_diff);
    } else {
      caffe_gpu_scal(count, loss_weight / num, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidWeightedCrossEntropyLossLayer);


}  // namespace caffe
