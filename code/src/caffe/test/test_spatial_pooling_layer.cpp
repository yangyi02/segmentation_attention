#include <algorithm>
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
class SpatialPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SpatialPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_2_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(1, 2, 3, 5);
    blob_bottom_2_->Reshape(1, 1, 1, 2);
    // fill the values
    Dtype* bottom_data = blob_bottom_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      bottom_data[i] = i;
    }
    Dtype* bottom_data2 = blob_bottom_2_->mutable_cpu_data();
    bottom_data2[0] = 2;
    bottom_data2[1] = 4;

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SpatialPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SpatialPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(SpatialPoolingLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_spatial_pooling_param()->set_num_bin(2);
  layer_param.mutable_spatial_pooling_param()->set_pool(
      SpatialPoolingParameter_PoolMethod_AVE);
  SpatialPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  EXPECT_EQ(top_data[0], Dtype(0. + 1.) / 2.);
  EXPECT_EQ(top_data[1], Dtype(2. + 3.) / 2.);
  EXPECT_EQ(top_data[2], Dtype(5. + 6.) / 2.);
  EXPECT_EQ(top_data[3], Dtype(7. + 8.) / 2.);
  EXPECT_EQ(top_data[4], Dtype(15. + 16.) / 2.);
  EXPECT_EQ(top_data[5], Dtype(17. + 18.) / 2.);
  EXPECT_EQ(top_data[6], Dtype(20. + 21.) / 2.);
  EXPECT_EQ(top_data[7], Dtype(22. + 23.) / 2.);
}



}  // namespace caffe
