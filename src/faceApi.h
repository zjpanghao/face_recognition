#ifndef INCLUDE_FACE_API_H
#define INCLUDE_FACE_API_H
#include <vector>
#include <string>
#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

struct FaceLocation {
 public:
  FaceLocation(cv::Rect &rect, double confidence) {
    rect_ = rect;
    confidence_ = confidence;
  }
  cv::Rect &rect() {
    return rect_;
  }
  double confidence() {
    return confidence_;
  }
  cv::Rect rect_;
  double confidence_;
};
using namespace dlib;
using dlib::tag1;
using dlib::add_prev2;
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using dlib::avg_pool;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
alevel0<
alevel1<
alevel2<
alevel3<
alevel4<
max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
  input_rgb_image_sized<150>
  >>>>>>>>>>>>;


class FaceApi {
 public:
  FaceApi(){init();}
  int getLocations(const cv::Mat &img, std::vector<FaceLocation> &locations, bool smallFace = false);
  int getFeature(const cv::Mat &img, std::vector<float> &feature);
  float compareFeature(const std::vector<float> &feature, 
                       const std::vector<float> &featureCompare);
 private: 
  int init();
  float confidence_{0.7};
  cv::dnn::Net detectNet_;
  dlib::shape_predictor shapePredict_;
  cv::dnn::Net embeddingNet_;
  anet_type dlibEmbeddingNet_;
};
#endif

