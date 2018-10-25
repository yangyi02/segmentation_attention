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
class ResizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ResizeLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ResizeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ResizeLayerTest, TestDtypesAndDevices);

TYPED_TEST(ResizeLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ResizeParameter* resize_param = layer_param.mutable_resize_param();
  resize_param->set_resize_ratio(2);
  ResizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 12);
  EXPECT_EQ(this->blob_top_->width(), 10);
  
  resize_param->set_resize_ratio(0.5);
  ResizeLayer<Dtype> layer2(layer_param);
  layer2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

/*
TYPED_TEST(ResizeLayerTest, PrintBackward) {
  LayerParameter layer_param;
  layer_param.set_resize_ratio(2);
  ResizeLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = i;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(ResizeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ResizeParameter* resize_param = layer_param.mutable_resize_param();
  resize_param->set_resize_ratio(2);
  const int num = 2;
  const int channels = 2;
  this->blob_bottom_->Reshape(num, channels, 2, 3);
  // Input: 2 x 2 channels of:
  //     [1 2 5]
  //     [9 4 1]
  for (int i = 0; i < 6 * num * channels; i += 6) {
    this->blob_bottom_->mutable_cpu_data()[i +  0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i +  1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i +  2] = 5;
    this->blob_bottom_->mutable_cpu_data()[i +  3] = 9;
    this->blob_bottom_->mutable_cpu_data()[i +  4] = 4;
    this->blob_bottom_->mutable_cpu_data()[i +  5] = 1;
  }
  ResizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 6);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output: 2 x 2 channels of:
  //     [1 1 2 2 5 5]
  //     [1 1 2 2 5 5]
  //     [9 9 4 4 1 1]
  //     [9 9 4 4 1 1]
  for (int i = 0; i < 12 * num * channels; i += 24) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 2], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 3], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 4], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 5], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 6], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 7], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 8], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 9], 2);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 10], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 11], 5);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 12], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 13], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 14], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 15], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 16], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 17], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 18], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 19], 9);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 20], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 21], 4);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 22], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 23], 1);
  }

  resize_param->set_resize_ratio(0.5);
  this->blob_bottom_->Reshape(num, channels, 3, 4);
  // Input: 3 x 4 channels of:
  //     [1 2 5 7]
  //     [9 4 1 3]
  //     [8 1 6 8]
  for (int i = 0; i < 12 * num * channels; i += 12) {
    this->blob_bottom_->mutable_cpu_data()[i + 0] = 1;
    this->blob_bottom_->mutable_cpu_data()[i + 1] = 2;
    this->blob_bottom_->mutable_cpu_data()[i + 2] = 5;
    this->blob_bottom_->mutable_cpu_data()[i + 3] = 7;
    this->blob_bottom_->mutable_cpu_data()[i + 4] = 9;
    this->blob_bottom_->mutable_cpu_data()[i + 5] = 4;
    this->blob_bottom_->mutable_cpu_data()[i + 6] = 1;
    this->blob_bottom_->mutable_cpu_data()[i + 7] = 3;
    this->blob_bottom_->mutable_cpu_data()[i + 8] = 8;
    this->blob_bottom_->mutable_cpu_data()[i + 9] = 1;
    this->blob_bottom_->mutable_cpu_data()[i + 10] = 6;
    this->blob_bottom_->mutable_cpu_data()[i + 11] = 8;
  }
  ResizeLayer<Dtype> layer2(layer_param);
  layer2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), num);
  EXPECT_EQ(this->blob_top_->channels(), channels);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 2);
  layer2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Expected output: 1 x 2 channels of:
  //     [1 5]
  for (int i = 0; i < 2 * num * channels; i += 2) {
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 0], 1);
    EXPECT_EQ(this->blob_top_->cpu_data()[i + 1], 5);
  }
}

TYPED_TEST(ResizeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  for (int resize_ratio = 2; resize_ratio <= 3; resize_ratio++) {
    LayerParameter layer_param;
    ResizeParameter* resize_param = layer_param.mutable_resize_param();
    resize_param->set_resize_ratio(resize_ratio);
    ResizeLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
}

}  // namespace caffe
