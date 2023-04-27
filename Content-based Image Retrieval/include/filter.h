/*
  Hao Sheng (Jack) Ning

  Utility functions for performing different filters and magnitude/orientation calculations.
  Also contains some image modification functions

  Each function taks in some input and one or more output parameters
  The parameters are passed by reference, in other words variables are pointers
 */
#pragma

int greyscale( cv::Mat &src, cv::Mat &dst);
int blur5x5( cv::Mat &src, cv::Mat &dst );
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );
int orientation( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );
int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );
int medianFilter( cv::Mat &src, cv::Mat &dst);
int addCaption(cv::Mat &src, cv::Mat &dst, std::string caption);
int generateMagnitudeHistogram( cv::Mat &src, std::vector<std::vector<std::vector<float>>> &targetMagnitudeHistogram);
int generateOrientationHistogram( cv::Mat &src, std::vector<float> &targetOrientationHistogram);
int calculateStandardDeviation( std::map<std::string,float> &data, double &sd);