#include "faceApi.h"
#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <dlib/opencv.h>
int FaceApi::init() {
  detectNet_ = cv::dnn::readNetFromCaffe(
      "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel");
  // embeddingNet_ = cv::dnn::readNetFromTorch("nn4.small2.v1.t7");
  deserialize("shape_predictor_5_face_landmarks.dat") >> shapePredict_;
  deserialize("dlib_face_recognition_resnet_model_v1.dat") >> dlibEmbeddingNet_;
  return 0;
}

int FaceApi::getLocations(const cv::Mat &img, std::vector<FaceLocation> &locations, bool smallFace) {
  cv::Mat image = cv::dnn::blobFromImage(img, 1.0,
      smallFace ? cv::Size() : cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false); 
  detectNet_.setInput(image, "data");
  cv::Mat detection = detectNet_.forward("detection_out");
  cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
  for (int i = 0; i < detectionMat.rows; i++) {
    float confidence = detectionMat.at<float>(i, 2);
    if (confidence > confidence_) {
      int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
      int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
      int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
      int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);
      if (x1 < 0 || x1 >= img.cols || x2 < 0 || x2 >= img.cols) {
        continue;
      }
      if (y1 < 0 || y1 >= img.rows || y2 < 0 || y2 >= img.rows) {
        continue;
      }
      if (x2 <= x1 || y2 <= y1) {
        continue;
      }
      cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
      FaceLocation faceLocation(rect, confidence); 
      locations.push_back(faceLocation);
    }
  }
  return 0;
}

int FaceApi::getFeature(const cv::Mat &img, std::vector<float> &feature) {
  //cv::Mat sampleF32(img.size(), CV_32FC3);
  //img.convertTo(sampleF32, sampleF32.type());
  //sampleF32 /= 255;
  //cv::resize(sampleF32, sampleF32, cv::Size(96, 96), 0, 0, 0);
  matrix<rgb_pixel> image;
  assign_image(image, dlib::cv_image<rgb_pixel>(img));
  auto shape = shapePredict_(image, dlib::rectangle(img.cols, img.rows));
  std::vector<matrix<rgb_pixel>> faces;
  matrix<rgb_pixel> faceChip;
  extract_image_chip(image, get_face_chip_details(shape,150,0.25), faceChip);
  //std::vector<matrix<float,0,1>> faceDescriptors = dlibEmbeddingNet_(faceChip);
  auto faceDescriptors = dlibEmbeddingNet_(faceChip);
  std::cout << faceDescriptors.size() << std::endl;
#if 0
  if (faceDescriptors.size() != 1) {
    std::cout <<"error size:" << faceDescriptors.size() << std::endl;
    return -1;
  }
#endif
  auto desc = faceDescriptors;
  feature.reserve(desc.nr());
  for (int i = 0; i < desc.nr(); i++) {
    feature.push_back(desc(i, 0));;
  }
  //std::cout << desc.nc() << "," << desc.nr() << std::endl;
  //std::cout << "The size is:" <<desc.size() << std::endl;
  return 0;
#if 0
  cv::Mat image = cv::dnn::blobFromImage(img, 1.0 / 255,
      cv::Size(96, 96), cv::Scalar(), true); 
  embeddingNet_.setInput(image);
  cv::Mat featureBlob = embeddingNet_.forward();
  float *p = (float*)featureBlob.data;
  if (!p) {
    return -1;
  }
  if (featureBlob.rows * featureBlob.cols != 128) {
    return -2;
  }
  feature.reserve(featureBlob.rows *featureBlob.cols);
  for (int i= 0; i < featureBlob.rows * featureBlob.cols; i++) {
    feature.push_back(*p++);
  }
  return 0;
#endif
}

float FaceApi::compareFeature(const std::vector<float> &feature, 
    const std::vector<float> &featureCompare) {
  int len = feature.size() > featureCompare.size() ? featureCompare.size() : feature.size();
  std::cout << "The len is" << len << std::endl;
  double sum = 0;
  for (int i = 0; i < len; i++) {
    sum += (feature[i] - featureCompare[i]) * (feature[i] - featureCompare[i]);
  }
  float f = 1 - sqrt(sum)*0.6;
  if (f < 0) {
    f = 0;
  } else if (f > 1) {
    f = 1;
  }
  return f*100;
}

