#include "faceApi.h"
#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include "apipool/apiPool.h"
#include "pbase64/base64.h"
void testFeature(FaceApi &api, cv::Mat &m) {
  std::vector<float> f1;
  int rc = api.getFeature(m, f1);
  std::cout << "rc:" << rc << std::endl;
  std::cout << "f1:" << f1.size() << std::endl;
  return ;
}

int main(int argc, char *argv[]) {
  FaceApi api;
  cv::Mat m, mc;
  if (argc < 2) {
    return -1;
  }
  m = cv::imread(argv[1]);
  if (argc >= 3) {
    mc = cv::imread(argv[2]);
  } else {
    mc = m;
  }
  std::vector<FaceLocation> locations1;
  struct timeval tv[2];
  gettimeofday(&tv[0], NULL);
  for (int i = 0; i < 1; i++) {
    api.getLocations(m, locations1, true);
  }
  gettimeofday(&tv[1], NULL);
  for (int i = 0; i < 2; i++) {
    std::cout << tv[i].tv_sec << "  "<< tv[i].tv_usec << std::endl;
  }
  std::cout <<"sizei1:" << locations1.size() << std::endl;
  if (locations1.size() == 0) {
    return 0;
  }
  cv::Mat m1;
  std::vector<float> f1;
  std::vector<float> f2;
  if(locations1.size() >= 1) {
    auto &l = locations1[0];
    cv::Mat first(m, l.rect());
    int rc = api.getFeature(first, f1);
    std::cout << "The rc is :" << rc;
  }
#if 1
  for(auto &l : locations1) {
    cv::rectangle(m, l.rect(), cv::Scalar(0, 0, 255));
    std::cout << l.rect().x <<" "<< l.rect().y <<" " <<  l.rect().width << " " <<l.rect().height <<" " << l.confidence() << std::endl;
  }
#endif
  imwrite("a.jpg", m);
  std::vector<FaceLocation> locations2;
  api.getLocations(mc, locations2);
  std::cout << "size2 :"<< locations2.size() << std::endl;
  if (locations2.size() == 0) {
    return -1;
  }
  if(locations2.size() >= 1) {
    auto &l = locations2[0];
    cv::Mat first(mc, l.rect());
    api.getFeature(first, f2);
  }
  std::cout << "feature 2:" << f2.size() << std::endl;
  std::cout <<"compare result:" <<  api.compareFeature(f1, f2) << std::endl;
#if 0
  for (auto &v : f1) {
    std::cout << v << ",";
  }
#endif
  std::cout <<std::endl;
  std::string f1base;
  char *start = (char*)&f1[0];
  std::vector<unsigned char> uf(start, start + 4 * 128);
  Base64::getBase64().encode(uf, f1base);
//  std::cout << "f1base:" << f1base << std::endl;
  //std::string featureBase64 = ImageBase64::encode((unsigned char*)&f2[0], f2.size() * sizeof(float));
#if 0
  std::cout << "f2base:" << featureBase64 << std::endl;
  for (auto &v : f2) {
    std::cout << v << ",";
  }
#endif
#if 1
  for(auto &l : locations2) {
    cv::rectangle(mc, l.rect(), cv::Scalar(0, 0, 255));
    imwrite("b.jpg", mc);
    std::cout << l.rect().x <<" "<< l.rect().y <<" " <<  l.rect().width << " " <<l.rect().height <<" " << l.confidence() << std::endl;
  }
#endif
  return 0;
}
