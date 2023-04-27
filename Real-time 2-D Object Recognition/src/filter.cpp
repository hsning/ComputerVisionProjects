/*
Hao Sheng (Jack) Ning

CS 5330 Computer Vision
Spring 2023

CPP functions for performing different filters and magnitude/orientation calculations
- also contains some image modification utility functions

The functions return an int to indicate whether the function completed successfully or not
*/
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
#include <vector>
using namespace std;
using namespace cv;

#define PI 3.14159265
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
        float differenceX = rptrx[j][c];
        float differenceY = rptry[j][c];
        dptr[j][c]=(sqrt(differenceX*differenceX)+(differenceY*differenceY));
      }
    }
  }
  //On success, return 0
  return(0);
}

int orientation( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst )
{
  //Create dst with zeros and sx's size and CV_8UC3 data type
  dst = cv::Mat::zeros( sx.size(), CV_16FC3 );
  //Iterate through every row in the image
  for(int i=0;i<sx.rows;i++) 
  {
    // src pointer
    cv::Vec3s *rptrx = sx.ptr<cv::Vec3s>(i);
    cv::Vec3s *rptry = sy.ptr<cv::Vec3s>(i);
    // destination pointer
    cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
    // for each column
    for(int j=0;j<sx.cols;j++) 
    {
      // for each color channel
      for(int c=0;c<3;c++) 
      {
        //Dst image equals the square root of sum of sx and sy squared
        dptr[j][c]= atan2((rptrx[j][c]),(rptry[j][c]))*180/PI+180;
        //cout<<"result:"<<dptr[j][c]<<endl;
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

int generateMagnitudeHistogram( cv::Mat &src, vector<vector<vector<float>>> &targetMagnitudeHistogram)
{
    cv::Mat targetX;
    cv::Mat targetY;
    cv::Mat targetM;
    sobelX3x3(src,targetX);
    sobelY3x3(src,targetY);
    magnitude(targetX,targetY,targetM);
    int numberOfPixels = targetM.rows*targetM.cols;
    for(int i=0;i<targetM.rows;i++)
    {
        cv::Vec3b *trptr = targetM.ptr<cv::Vec3b>(i);
        for(int j=0;j<targetM.cols;j++)
        {
            int b = trptr[j][0]/32;
            int g = trptr[j][1]/32;
            int r = trptr[j][2]/32;
            targetMagnitudeHistogram[b][g][r]+=(1.0/numberOfPixels);
        }
    }
    targetX.release();
    targetY.release();
    targetM.release();
    return (0);
}

int generateOrientationHistogram( cv::Mat &src, vector<float> &targetOrientationHistogram)
{
    cv::Mat targetX;
    cv::Mat targetY;
    cv::Mat targetO;
    sobelX3x3(src,targetX);
    sobelY3x3(src,targetY);
    orientation(targetX,targetY,targetO);
    int numberOfPixels = targetO.rows*targetO.cols;
    for(int i=0;i<targetO.rows;i++)
    {
        cv::Vec3s *trptr = targetO.ptr<cv::Vec3s>(i);
        for(int j=0;j<targetO.cols;j++)
        {
            for(int c=0; c<3;c++)
            {
                int orientationBin = trptr[j][c]/15;
                targetOrientationHistogram[orientationBin]+=(1.0/numberOfPixels/3.0);
                //cout<<orientationBin<<endl;  
            }
        }
    }
    targetX.release();
    targetY.release();
    targetO.release();
    return (0);
}

int calculateStandardDeviation( std::map<string,float> &data, double &sd)
{
  double mean = 0.0;
  double sum = 0.0;
  for (auto itr = data.begin(); itr != data.end(); ++itr) {
        sum += itr->second;
    }
  mean = sum/data.size();
  for (auto itr = data.begin(); itr != data.end(); ++itr) {
        double difference = itr->second-mean;
        sum += difference * difference;
    }
    sd = sqrt(sum/(data.size()-1));
    return (0);
}

int threshold( cv::Mat &src, cv::Mat &dst)
{
    //Copy src image to dst
    cv::Mat blurred;
    src.copyTo(blurred);
    src.copyTo(dst);
    //Create blurred with src's size and type
    blurred.create(src.size(),src.type());
    // Custom blur function
    blur5x5(src,blurred);
    //GaussianBlur(src, blurred, Size(), 1,1);
    cv::Mat temp;
    Mat sharpened = src*(1+1) + blurred*(-1);
    temp = Mat::zeros( sharpened.size(), sharpened.type() );
    convertScaleAbs(sharpened,temp,1,70);
    //blurred.copyTo(dst);
    
    //Iterate through every row in the image
    for (int i = 0; i < temp.rows; ++i) 
    { 
        //Iterate through every column in the image
        for (int j = 0; j < temp.cols; ++j) 
        { 
          // Background
          if((temp.at<uchar>(i,j) > 150))
          {
            dst.at<uchar>(i,j) =0;
          }
          // Forecolor
          else
          {
            dst.at<uchar>(i,j) =255;
          }
        } 
    } 
    //On success, return 0
    return(0);  
}

int monphological( cv::Mat &src, cv::Mat &dst)
{
    //Copy src image to dst
    cv::Mat temp;
    src.copyTo(temp);
    dst = Mat::zeros(src.size(), CV_16SC3);
    //GaussianBlur(src, blurred, Size(7, 7), 5,0);
    //src.copyTo(dst);
    //erode(src,temp,Mat(),Point(0,0),2,1,1);
    
    erode(src,dst,Mat(),Point(0,0),2,1,1);
    dilate(src,dst,Mat(),Point(0,0),1,1,1);
    //Iterate through every row in the image
    // for (int i = 1; i < src.rows-1; ++i) 
    // { 
    //     //Assign the row to a row pointer
    //     cv::Vec3b *rowptrm1 = src.ptr<cv::Vec3b>(i-1); 
    //     cv::Vec3b *rowptr = src.ptr<cv::Vec3b>(i); 
    //     cv::Vec3b *rowptrp1 = src.ptr<cv::Vec3b>(i+1); 
    //     cv::Vec3b *rowptrd = dst.ptr<cv::Vec3b>(i);
    //     //Iterate through every column in the image
    //     for (int j = 1; j < src.cols-1; ++j) 
    //     { 
    //       // Erosion: erase noise
    //       // If the pixel has a fore-color but the neighbours are background, 
    //       // change it to background color 
    //       if(rowptr[j][0]==255)
    //       {
    //         if((rowptrm1[j-1][0]==0) && (rowptrm1[j][0]==0) && (rowptrm1[j+1][0]==0) && (rowptr[j-1][0]==0) && (rowptr[j+1][0]==0) && (rowptrp1[j-1][0]==0)&&(rowptrp1[j][0]==0)&&(rowptrp1[j+1][0]==0))
    //         {
    //           rowptrd[j][0]=0;
    //           rowptrd[j][1]=0;
    //           rowptrd[j][2]=0;
    //         }
    //       }
    //       else if(rowptr[j][0]==0)
    //       // Dilation: Enlarge holes
    //       // If the pixel has a background color but the neighbours are foreground, 
    //       // change it to background color 
    //       {
    //         if((rowptrm1[j-1][0]==255) && (rowptrm1[j][0]==255) && (rowptrm1[j+1][0]==255) && (rowptr[j-1][0]==255) && (rowptr[j+1][0]==255) && (rowptrp1[j-1][0]==255)&&(rowptrp1[j][0]==255)&&(rowptrp1[j+1][0]==255))
    //         {
    //           rowptrd[j][0]=255;
    //           rowptrd[j][1]=255;
    //           rowptrd[j][2]=255;
    //         }
    //       }
    //         // row_ptr[j] will give you access to the pixel value 
    //         // any sort of computation/transformation is to be performed here 
    //         // change all other color channels' value to that of Green
    //     } 
    // } 
    //On success, return 0
    return(0);  
}

int segement( cv::Mat &src, cv::Mat &dst, vector<vector<float>> &regionsMatrix, map<int, int> regions)
{
    // Stack of Points
    stack<pair<int,int>> stackOfPoints;
    //Copy src image to dst
    cv::Mat temp;
    src.copyTo(temp);
    dst = Mat::zeros(src.size(), CV_16SC3);
    int regionID = 1;
    //Iterate through every row in the image
    for (int i = 0; i < src.rows; ++i) 
    { 
        //Assign the row to a row pointer
        cv::Vec3b *rowptrm1 = src.ptr<cv::Vec3b>(i-1); 
        cv::Vec3b *rowptr = src.ptr<cv::Vec3b>(i); 
        //Iterate through every column in the image
        for (int j = 0; j < src.cols; ++j) 
        { 
          // Look up and back from P
          if(i!=0 || j!=0)
          {
            // If it is forecolor
            if(rowptr[j][0]==255)
            {
                // If it is not labelled
                if(regionsMatrix[i][j]==0)
                {

                }
                // Check for Up and Back pixels, if one of is foreground,
                // Add the regionID to the matrix
                if((rowptrm1[j][0] == 255) || (rowptr[j-1][0] == 255))
                {
                  regionsMatrix[i][j] = regionID;
                }
                // If none around is labelled
                {
                  regionID++;
                }
            }
          }
        } 
    } 
    //On success, return 0
    return(0);  
}

int computeMomentsFeautures( cv::Mat &regionMap, int regionID, Moments &m, RotatedRect &boundingRect, float &orientedCentralMoments)
{
    Mat objImg;
    vector<Vec4i> hierarchy;
    vector< vector< cv::Point> > contours;
    cv::moments(contours);
    objImg = (regionMap == regionID); 
    m = cv::moments(objImg, true); // get moments as world coordinates
    //cv::RotatedRect dboundingRect = minAreaRect(objImg);
    findContours(objImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    boundingRect = minAreaRect(contours[0]);
    float angle;
    angle=atan2(2*m.mu11,m.mu20-m.mu02)*0.5+PI/2;
    float sumDifferences=0.0;
    for(int i = 0; i<objImg.rows; i++)
    {
      for(int j=0; j<objImg.cols;j++)
      {
         float differences =((j-(m.m10/m.m00))*cos(angle)+(i-m.m01/m.m00)*sin(angle));
         sumDifferences += differences*differences;
      }
    }
    orientedCentralMoments = sumDifferences/m.m00;  
    return(0);  
}
