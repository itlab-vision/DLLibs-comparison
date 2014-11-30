#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using namespace std;


bool ReadMatToDatum(cv::Mat& cv_img_origin, const int label,
    const int height, const int width, Datum* datum) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  int num_channels = cv_img_origin.channels();
  datum->set_channels(num_channels);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  if (num_channels!=1) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  } else {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
          static_cast<char>(cv_img.at<uchar>(h, w)));
        }
      }
  }
  return true;
}

int main(int argc, char** argv) {
    //get the net
      Net<float> caffe_test_net(argv[1]);
      //get trained net
      caffe_test_net.CopyTrainedLayersFrom(argv[2]);
     // Run ForwardPrefilled
      float loss;
     const vector<Blob<float>*>& result =  caffe_test_net.ForwardPrefilled(&loss);
    // Now result will contain the argmax results.
     const float* argmaxs = result[1]->cpu_data();
      for (int i = 0; i < result[0]->num(); ++i) {
       LOG(INFO) << " Image: "<< i << " class:" << argmaxs[i];
      }
  return 0;
}
