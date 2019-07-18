#include "faceApi.h"
#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <dlib/opencv.h>
#include <cmath>
int FaceApi::init() {
  detectNet_ = cv::dnn::readNetFromCaffe(
      "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel");
  // embeddingNet_ = cv::dnn::readNetFromTorch("nn4.small2.v1.t7");
  deserialize("shape_predictor_68_face_landmarks.dat") >> shapePredict_;
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



static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
{
    return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}



int FaceApi::getFeature(const cv::Mat &img, std::vector<float> &feature) {
#if 0
  cv::Mat gray;
  cv::cvtColor(img, gray, CV_RGB2GRAY);
  cv::Laplacian(gray, gray, CV_16S, 3);
  cv::convertScaleAbs(gray, gray);
  cv::Mat mean, sd;
  cv::meanStdDev(gray, mean, sd);
  double count= sd.at<double>(0, 0);
  std::cout << count << std::endl;
#endif
  //cv::Mat sampleF32(img.size(), CV_32FC3);
  //img.convertTo(sampleF32, sampleF32.type());
  //sampleF32 /= 255;
  //cv::resize(sampleF32, sampleF32, cv::Size(96, 96), 0, 0, 0);
  matrix<rgb_pixel> image;
  assign_image(image, dlib::cv_image<rgb_pixel>(img));
  auto detector = get_frontal_face_detector();
  if (detector(image).empty()) {
    return -1;
  }
  auto shape = shapePredict_(image, dlib::rectangle(img.cols, img.rows));
  if (shape.num_parts() != 68) {
    return -2;
  }
  //cv::rectangle(m2, dlibRectangleToOpenCV(shape.get_rect()), cv::Scalar(0, 0, 255));
  auto p40 = shape.part(40);
  auto p28 = shape.part(28);
  auto p43 = shape.part(43);

  auto p38 = shape.part(38);
  auto p39 = shape.part(39);
  auto p47 = shape.part(47);
  auto p48 = shape.part(48);

  auto p41 = shape.part(41);
  auto p42= shape.part(42);
  auto p44= shape.part(44);
  auto p45= shape.part(45);

  
  int leftTop = p38.y() > p39.y() ? p38.y() : p39.y();
  int rightDown = p48.y() < p47.y() ? p48.y() : p47.y();
  if (leftTop > rightDown) {
    return -3;
  }
   
  int leftDown = p41.y() < p42.y() ? p41.y() : p42.y();
  int rightTop = p44.y() > p45.y() ? p44.y() : p45.y();
  if (leftDown < rightTop) {
    return -4;
  } 
  float yaw = (float)(p43.x() - p28.x()) / (p28.x() - p40.x());
  if (yaw < 0.68 || yaw > 1.5) {
    return -5;
  }
  //cv::Mat m2 = img.clone();
  for (int i = 0; i < shape.num_parts(); i++) {
  //  cv::circle(m2, cv::Point(shape.part(i).x(), shape.part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
  }
  std::cout <<"yaw" <<  yaw << std::endl;
  //cv::imwrite("circle.jpg", m2);
  std::vector<matrix<rgb_pixel>> faces;
  matrix<rgb_pixel> faceChip;
  extract_image_chip(image, get_face_chip_details(shape,150,0.25), faceChip);
  //std::vector<matrix<float,0,1>> faceDescriptors = dlibEmbeddingNet_(faceChip);
  //cv::Mat m3 = dlib::toMat(faceChip);
  //cv::imwrite("circle2.jpg", m3);
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
  float a = 7;
  float b = -4;
  int len = feature.size() > featureCompare.size() ? featureCompare.size() : feature.size();
  double sum = 0;
  for (int i = 0; i < len; i++) {
    sum += (feature[i] - featureCompare[i]) * (feature[i] - featureCompare[i]);
  }
  float x = sqrt(sum);
  std::cout << "The O distence :" <<  x;
  float score = 1 / (1 + std::exp(a * x + b));
  return score;
  float f = 1 - sqrt(sum)*0.6;
  if (f < 0) {
    f = 0;
  } else if (f > 1) {
    f = 1;
  }
  return f*100;
}

