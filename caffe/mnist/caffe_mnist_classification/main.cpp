#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

#if 1
    #include <stdio.h>
    #define TIMER_START(name) int64 t_##name = getTickCount()
    #define TIMER_END(name) printf("TIMER_" #name ":\t%6.2fms\n", \
                1000.f * ((getTickCount() - t_##name) / getTickFrequency()))
#else
    #define TIMER_START(name)
    #define TIMER_END(name)
#endif

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}
void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
int main(int argc, char** argv) {

  Caffe::set_phase(Caffe::TEST);
  Caffe::set_mode(Caffe::CPU);

  string img_file = argc > 1 ? argv[1] : "image.bmp";

  string net_src = argc > 2 ? argv[2] : "lenet.prototxt";
  Net<float> caffe_test_net(net_src);  //get the net

  string traied_net = argc > 3 ? argv[3] : "lenet_iter_10000.caffemodel";
  caffe_test_net.CopyTrainedLayersFrom(traied_net);

  cout << "reading file" << endl;
  Datum datum;
  cv::Mat cv_img = ReadImageToCVMat(img_file, 28, 28, false);
  CVMatToDatum(cv_img, &datum);
  datum.set_label(7);
  std::vector<Datum> images;
  images.push_back(datum);

  float loss = 0.0;

  cout << "adding images" << endl;
  boost::dynamic_pointer_cast< caffe::MemoryDataLayer<float> >(caffe_test_net.layers()[0])->AddDatumVector(images);
  cout << "running net" << endl;
  std::vector<Blob<float>*> result
  TIMER_START(predict)
  for(int i=0;i<1000;i++)
  {
      result = caffe_test_net.ForwardPrefilled(&loss);
  }
TIMER_END(predict)
  cout << "got results" << endl;
  LOG(INFO)<< "Output result size: "<< result.size();

  int r = 0; // here in my case r=0 is for input label data, r=1 for prediction result (actually argmax layer)
  const float* argmaxs = result[r]->cpu_data();
  for (int i = 0; i < result[r]->num(); ++i) {

    LOG(INFO)<< " Image: "<< i << " class:" << argmaxs[i];
  }

  return 0;
}
