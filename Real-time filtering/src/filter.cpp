#include "filter.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <chrono>
#include <fstream>
#include <string>
#include <ctime>
#include "math.h"
#include <bits/stdc++.h>
#include <iterator>
using namespace std;
using namespace cv;


int greyscale( cv::Mat &src, cv::Mat &dst)
{
    //Copy src image to dst
    src.copyTo(dst);
    //Iterate through every row in the image
    for (int i = 0; i < dst.rows; ++i) 
    { 
        //Assign the row to a row pointer
        cv::Vec3b *rowptr = dst.ptr<cv::Vec3b>(i); 
        //Iterate through every column in the image
        for (int j = 0; j < dst.cols; ++j) 
        { 
            // row_ptr[j] will give you access to the pixel value 
            // any sort of computation/transformation is to be performed here 
            // change all other color channels' value to that of Green
            rowptr[j][0]=rowptr[j][1];
            rowptr[j][2]=rowptr[j][1];
        } 
    } 
    //On success, return 0
    return(0);  
}

int blur5x5( cv::Mat &src, cv::Mat &dst )
{
    //Initialize tmp and create dst and tmp with zeros and src's size and CV_16SC3 data type
    cv::Mat tmp;
    dst.create( src.size(), src.type());
    tmp.create( src.size(), src.type());
    //VERTICAL FILTER [1,2,4,2,1]
    //Iterate through every row in the image
    for(int i=2;i<src.rows-2;i++) 
    {
      // src pointer
      cv::Vec3b *rptrm2 = tmp.ptr<cv::Vec3b>(i-2);
      cv::Vec3b *rptrm1 = tmp.ptr<cv::Vec3b>(i-1);
      cv::Vec3b *rptr = tmp.ptr<cv::Vec3b>(i);
      cv::Vec3b *rptrp1 = tmp.ptr<cv::Vec3b>(i+1);
      cv::Vec3b *rptrp2 = tmp.ptr<cv::Vec3b>(i+2);
      // destination pointer
      cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
      // for each column
      for(int j=2;j<src.cols-2;j++) 
      {
        // for each color channel
        for(int c=0;c<3;c++) 
        {
          //Temporary image equals the first filter * orginal image
          tmp.at<cv::Vec3b>(i, j)[c] =
          ((src.at<cv::Vec3b>(i-2, j)[c]) *1 +
           (src.at<cv::Vec3b>(i-1, j)[c]) *2 +
           (src.at<cv::Vec3b>(i, j)[c]) *4 +
           (src.at<cv::Vec3b>(i+1, j)[c]) *2 +
           (src.at<cv::Vec3b>(i+2, j)[c]) *1 )/10;
        }
      }
  }
    //HORIZONTAL FILTER [1,2,4,2,1]
    //Iterate through every row in the image
    for(int i=2;i<tmp.rows-2;i++) 
    {
      // src pointer
      cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
      // destination pointer
      cv::Vec3b *dptr = tmp.ptr<cv::Vec3b>(i);
      // for each column
      for(int j=2;j<tmp.cols-2;j++) 
      {
        // for each color channel
        for(int c=0;c<3;c++) 
        {
          //Dst image equals the second filter * temporary image
          dst.at<cv::Vec3b>(i, j)[c]=
          ((tmp.at<cv::Vec3b>(i, j-2)[c])*1 + 
           (tmp.at<cv::Vec3b>(i, j-1)[c])*2 + 
           (tmp.at<cv::Vec3b>(i, j)[c])*4 + 
           (tmp.at<cv::Vec3b>(i, j+1)[c])*2 + 
           (tmp.at<cv::Vec3b>(i, j+2)[c])*1)/10;
        }
      }
    }
  //On success, return 0  
  return(0);
}
int sobelX3x3( cv::Mat &src, cv::Mat &dst )
{
  //Initialize tmp and create dst and tmp with zeros and src's size and CV_16SC3 data type
  cv::Mat tmp;
  dst = cv::Mat::zeros( src.size(), CV_16SC3 ); // signed short data type
  tmp = cv::Mat::zeros( src.size(), CV_16SC3 ); // signed short data type
  // loop over src and apply a 3x3 filter
  
  //HORIZONTAL FILTER [-1,0,1]
  //Iterate through every row in the image
  for(int i=1;i<src.rows-1;i++) 
  {
    // src pointer
    //cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i-1);
    cv::Vec3b *rptr = tmp.ptr<cv::Vec3b>(i);
    //cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i+1);
    // destination pointer
    cv::Vec3s *tptr = tmp.ptr<cv::Vec3s>(i);
    cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<src.cols-1;j++) 
    {
      // for each color channel
      for(int c=0;c<3;c++) 
      {
        //Temporary image equals the first filter * orginal image
        tmp.at<cv::Vec3s>(i,j)[c]=(
          -1*(src.at<cv::Vec3b>(i, j-1)[c]) + 1*(src.at<cv::Vec3b>(i, j+1)[c] ));
      }
    }
  }

  //VERTICAL FILTER [1,2,1]
  //Iterate through every row in the image
  for(int i=1;i<src.rows-1;i++) 
  {
    // destination pointer
    cv::Vec3s *dptr = tmp.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<src.cols-1;j++) 
    {
      // for each color channel
      for(int c=0;c<3;c++) 
      {
        //Dst image equals the second filter * temporary image
        dst.at<cv::Vec3s>(i,j)[c]=(
          (1*(tmp.at<cv::Vec3s>(i-1, j)[c])) +
          (2*(tmp.at<cv::Vec3s>(i, j)[c])) +
          (1*(tmp.at<cv::Vec3s>(i+1, j)[c])))/4;
      }
    }
  }
  
  waitKey(1);
  //On success, return 0
  return(0);
}

int sobelY3x3( cv::Mat &src, cv::Mat &dst )
{
  //Create dst and tmp with zeros and src's size and CV_16SC3 data type
  cv::Mat tmp;
  dst = cv::Mat::zeros( src.size(), CV_16SC3 ); // signed short data type
  tmp = cv::Mat::zeros( src.size(), CV_16SC3 ); // signed short data type
  // loop over src and apply a 3x3 filter
  
  //VERTICAL FILTER[1,0,-1]
  //Iterate through every row in the image
  for(int i=1;i<src.rows-1;i++) 
  {
    // src pointer
    //cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i-1);
    cv::Vec3b *rptr = tmp.ptr<cv::Vec3b>(i);
    //cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i+1);
    // destination pointer
    cv::Vec3s *tptr = tmp.ptr<cv::Vec3s>(i);
    cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<src.cols-1;j++) 
    {
      // for each color channel
      for(int c=0;c<3;c++) 
      {
        //Temporary image equals the first filter * orginal image
        tmp.at<cv::Vec3s>(i,j)[c]=(
          1*(src.at<cv::Vec3b>(i-1,j)[c]) + -1*(src.at<cv::Vec3b>(i+1, j)[c] ));
      }
    }
  }

  //HORIZONTAL FILTER [1,2,1]
  //Iterate through every row in the image
  for(int i=1;i<src.rows-1;i++) 
  {
    // destination pointer
    cv::Vec3s *dptr = tmp.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=1;j<src.cols-1;j++) 
    {
      // for each color channel
      for(int c=0;c<3;c++) 
      {
        //Dst image equals the second filter * temporary image
        dst.at<cv::Vec3s>(i,j)[c]=(
          (1*(tmp.at<cv::Vec3s>(i, j-1)[c])) +
          (2*(tmp.at<cv::Vec3s>(i, j)[c])) +
          (1*(tmp.at<cv::Vec3s>(i, j+1)[c])))/4;
      }
    }
  }
  //On success, return 0
  return(0);
}

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst )
{
  //Create dst with zeros and sx's size and CV_8UC3 data type
  dst = cv::Mat::zeros( sx.size(), CV_8UC3 );
  //Iterate through every row in the image
  for(int i=0;i<sx.rows;i++) 
  {
    // src pointer
    cv::Vec3s *rptrx = sx.ptr<cv::Vec3s>(i);
    cv::Vec3s *rptry = sy.ptr<cv::Vec3s>(i);
    // destination pointer
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
    // for each column
    for(int j=0;j<sx.cols;j++) 
    {
      // for each color channel
      for(int c=0;c<3;c++) 
      {
        //Dst image equals the square root of sum of sx and sy squared
        dptr[j][c]=(sqrt(pow(rptrx[j][c],2)+pow(rptry[j][c],2)));
      }
    }
  }
  //On success, return 0
  return(0);
}

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels )
{
  cv::Mat blurred;
  // src.copyTo(blurred);
  //Create blurred with src's size and type
  blurred.create(src.size(),src.type());
  blur5x5(src,blurred);
  
  dst.create( src.size(), src.type());
  int bucket = 255/levels;
  //Iterate through every row in the image
  for(int i=1;i<blurred.rows;i++) 
  {
    // src pointer
    cv::Vec3b *rptr = blurred.ptr<cv::Vec3b>(i);
    // destination pointer
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
    // for each column
    for(int j=0;j<blurred.cols;j++) 
    {
      // for each color channel
      for(int c=0;c<3;c++) 
      {
        //Dst image equals
        int xt = (rptr[j][c])/bucket;
        dptr[j][c]=(xt * bucket);
      }
    }
  }
  //On success, return 0
  return(0);
}

int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold )
{
  //Create dst with src's size and type
  dst.create(src.size(),src.type());
  cv::Mat blurred;
  cv::Mat mag;
  cv::Mat x;
  cv::Mat y;
  sobelX3x3(src,x);
  sobelY3x3(src,y);
  magnitude( x, y, mag);
  blurQuantize( src, blurred, levels );
  waitKey(1);
  //Iterate through every row in the image
  for(int i=1;i<src.rows-1;i++) 
  {
    // blurred pointer
    cv::Vec3b *brptr = blurred.ptr<cv::Vec3b>(i);
    // mag pointer
    cv::Vec3b *mrptr = mag.ptr<cv::Vec3b>(i);
    // destination pointer
    cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
    // for each column
    for(int j=1;j<src.cols-1;j++) 
    {
      // for each color channel
      for(int c=0;c<3;c++) 
      {
        if(mrptr[j][c]>magThreshold)
        {
          dptr[j][c] = 0;
        }
        else
        {
          dptr[j][c] = brptr[j][c];
        }
      }
    }
  }
  //On success, return 0
  return(0);
}

int addCaption(cv::Mat &src, cv::Mat &dst, std::string caption)
{
  //Create dst with src's size and type
  dst.create(src.size(),src.type());
  //Put text method
  putText(src,caption,cv::Point(0,100),cv::FONT_HERSHEY_DUPLEX,2,cv::Scalar(0,255,0),1,8);
  src.copyTo(dst);
  //On success, return 0
  return(0);
}

int medianFilter( cv::Mat &src, cv::Mat &dst)
{
  //Create dst with src's size and type
  dst.create(src.size(),src.type());
  //Built in medianBlur function, 19 is the ksize value
  medianBlur(src,dst,19);
  //On success, return 0
  return(0);
}
