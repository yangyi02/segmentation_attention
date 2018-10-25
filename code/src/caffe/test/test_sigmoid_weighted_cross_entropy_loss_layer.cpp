#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SigmoidWeightedCrossEntropyLossLayerTest :
      public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SigmoidWeightedCrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(5, 1, 4, 3)),
        blob_bottom_targets_(new Blob<Dtype>(5, 1, 4, 3)),
        blob_bottom_weights_(new Blob<Dtype>(5, 1, 4, 3)),
        blob_top_loss_(new Blob<Dtype>()) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);
    // Fill the weights vector
    FillerParameter weights_filler_param;
    weights_filler_param.set_min(0);
    weights_filler_param.set_max(1);
    UniformFiller<Dtype> weights_filler(weights_filler_param);
    weights_filler.Fill(blob_bottom_weights_);
    // Fill the targets vector
    for (int i = 0; i < blob_bottom_targets_->count(); ++i) {
      blob_bottom_targets_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;
    }

    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SigmoidWeightedCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
    delete blob_bottom_weights_;
    delete blob_top_loss_;
  }

  void PrepareTwoBottomDataForTest() {
    blob_bottom_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_targets_);
  }
  void PrepareThreeBottomDataForTest() {
    blob_bottom_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_targets_);
    blob_bottom_vec_.push_back(blob_bottom_weights_);
  }
  Dtype SigmoidWeightedCrossEntropyLossReference(const int num,
           const int dim, const std::vector<Dtype>& weights,
           const Dtype* input, const Dtype* target) {
    Dtype loss = 0;
    Dtype batch_weight = 0;
    for (int n = 0; n < num; ++n) {
      Dtype positive_weight = weights[n];
      Dtype negative_weight = 1 - weights[n];
      for (int i = 0; i < dim; ++i) {
        int pos = n * dim + i;
        const Dtype prediction = 1 / (1 + exp(-input[pos]));
        EXPECT_LE(prediction, 1);
        EXPECT_GE(prediction, 0);
        EXPECT_LE(target[pos], 1);
        EXPECT_GE(target[pos], 0);
        loss -= positive_weight *
            target[pos] * log(prediction + (target[pos] == Dtype(0)));
        loss -= negative_weight *
            (1 - target[pos]) * log(1 - prediction + (target[pos] == Dtype(1)));

        if (target[pos]) {
          batch_weight += positive_weight;
        } else {
          batch_weight += negative_weight;
        }
      }
    }
    if (batch_weight == 0) {
      batch_weight = 1;
    }
    return loss / batch_weight;
  }

  void TestForward() {
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);

    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);

    FillerParameter weights_filler_param;
    weights_filler_param.set_min(0);
    weights_filler_param.set_max(1);
    UniformFiller<Dtype> weights_filler(weights_filler_param);
    
    Dtype eps = 2e-2;

    const int num = this->blob_bottom_data_->num();
    const int dim = this->blob_bottom_data_->count() /
        this->blob_bottom_data_->num();
    const Dtype* blob_bottom_data =
        this->blob_bottom_data_->cpu_data();
    const Dtype* blob_bottom_targets =
          this->blob_bottom_targets_->cpu_data();
    std::vector<Dtype> positive_weights(num, 0);

    for (int i = 0; i < 100; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the weight vector
      weights_filler.Fill(blob_bottom_weights_);
      // Fill the targets vector
      for (int n = 0; n < blob_bottom_targets_->count(); ++n) {
        blob_bottom_targets_->mutable_cpu_data()[n] =
            caffe_rng_rand() % 2;
      }

      SigmoidWeightedCrossEntropyLossLayer<Dtype> layer(layer_param);

      for (int n = 0; n < num; ++n) {
        positive_weights[n] =
           blob_bottom_weights_->cpu_data()[n];
      }
      PrepareThreeBottomDataForTest();

      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      Dtype layer_loss =
          layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      Dtype reference_loss = kLossWeight *
          SigmoidWeightedCrossEntropyLossReference(
              num, dim, positive_weights,
              blob_bottom_data, blob_bottom_targets);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  Blob<Dtype>* const blob_bottom_weights_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SigmoidWeightedCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SigmoidWeightedCrossEntropyLossLayerTest,
           TestSigmoidWeightedCrossEntropyLossForward) {
  this->TestForward();
}

TYPED_TEST(SigmoidWeightedCrossEntropyLossLayerTest,
           TestGradientThreeBottomData) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  SigmoidWeightedCrossEntropyLossLayer<Dtype> layer(layer_param);
  //  this->PrepareTwoBottomDataForTest();
  this->PrepareThreeBottomDataForTest();
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SigmoidWeightedCrossEntropyLossLayerTest,
           TestGradientTwoBottomData) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  SigmoidWeightedCrossEntropyLossLayer<Dtype> layer(layer_param);
  this->PrepareTwoBottomDataForTest();
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SigmoidWeightedCrossEntropyLossLayerTest,
           TestGradientWithIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);


  for (int label = 0; label < 2; ++label) {
    layer_param.mutable_loss_param()->clear_ignore_label();
    layer_param.mutable_loss_param()->add_ignore_label(label);
    SigmoidWeightedCrossEntropyLossLayer<Dtype> layer(layer_param);
    //  this->PrepareTwoBottomDataForTest();
    this->PrepareThreeBottomDataForTest();
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_, 0);
  }
}


}  // namespace caffe
