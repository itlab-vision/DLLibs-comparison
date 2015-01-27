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

  if (argc < 4 || argc > 6) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations "
        << "[CPU/GPU] [Device ID]";
    return 1;
  }
  Caffe::set_phase(Caffe::TEST);

  //Setting CPU or GPU
  if (argc >= 5 && strcmp(argv[4], "GPU") == 0) {
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    if (argc == 6) {
      device_id = atoi(argv[5]);
    }
    Caffe::SetDevice(device_id);
    LOG(ERROR) << "Using GPU #" << device_id;
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  //get the net
  Net<float> caffe_test_net(argv[1]);
  //get trained net
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
  //get datum
  cv::Mat img=cv::imread("mnist_4.png",CV_LOAD_IMAGE_GRAYSCALE);
  Datum datum;
  ReadMatToDatum(img, 1, 28, 28, &datum);
  //get the blob
  Blob<float>* blob = new Blob<float>(1, datum.channels(), datum.height(), datum.width());
//  cout<<datum.channels()<<datum.height()<<datum.width()<<flush<<endl;
  //get the blobproto
  BlobProto blob_proto;
  blob_proto.set_num(1);
  blob_proto.set_channels(datum.channels());
  blob_proto.set_height(datum.height());
  blob_proto.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  cout<<size_in_datum<<flush<<endl;
  for (int i = 0; i < size_in_datum; ++i) {
    blob_proto.add_data(0.);
  }
  const string& data = datum.data();
  if (data.size() != 0) {
    for (int i = 0; i < size_in_datum; ++i) {
      blob_proto.set_data(i, blob_proto.data(i) + (uint8_t)data[i]);
    }
  }
  blob->FromProto(blob_proto);

  //fill the vector
  vector<Blob<float>*> bottom;
  bottom.push_back(blob);
  float type = 0.0;
  const vector<Blob<float>*>& result =  caffe_test_net.Forward(bottom, &type);

  float max = 0;
  float max_i = 0;
  for (int i = 0; i < 10; ++i) {
    float value = result[0]->cpu_data()[i];
    if (max < value){
      max = value;
      max_i = i;
    }
  }
  LOG(ERROR) << "max: " << max << " i " << max_i;

  return 0;
}
