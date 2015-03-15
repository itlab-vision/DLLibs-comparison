#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

int main(int argc, char** argv) {

  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::CPU);

  string img_file = argv[1];

  string net_src = argv[2];
  Net<float> caffe_test_net(net_src);  //get the net

  string trained_net = argv[3];
  caffe_test_net.CopyTrainedLayersFrom(trained_net);

  Datum datum;
  if (!ReadImageToDatum(img_file, 7, 28, 28, false,&datum)) {
    LOG(ERROR) << "Error during file reading";
  }
  std::vector<Datum> images;
  images.push_back(datum);

  float loss = 0.0;

  boost::dynamic_pointer_cast< caffe::MemoryDataLayer<float> >(caffe_test_net.layers()[0])->AddDatumVector(images);
  std::vector<Blob<float>*> result = caffe_test_net.ForwardPrefilled(&loss);
  LOG(INFO)<< "Output result size: "<< result.size();

  int r = 0; // r=1 is for input label data, r=0 for prediction result (actually argmax layer)
  const float* argmaxs = result[r]->cpu_data();
  for (int i = 0; i < result[r]->num(); ++i) {

    LOG(INFO)<< " Image: "<< i << " class:" << argmaxs[i];
  }

  return 0;
}
