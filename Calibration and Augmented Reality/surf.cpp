/*
Hao Sheng (Jack) Ning

CS 5330 Computer Vision
Spring 2023
Project 4: Calibration and Augmented Reality

CPP surf.cpp file that stores a single function that takes in 2 images of the same scene (but different scale, rotation or translation), 
and extract keypoints from each using SURF and produced a combined image with keypoints matched

The surf program is interacted through command line interfaces with the following format:
surf.exe <source image path> <target image path>
For example:
surf.exe tree.jpg and treeR.jpg
tree.jpg and treeR.jpg are bundled with the submission, and they can be used as a testing base
treeR.jpg is of the exact scene as the tree.jpg but in upside down rotation
The functions return an int to indicate whether the function completed successfully or not
*/

// Include libraries
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "include/filter.h"
#include "include/csv_util.h"
#include <chrono>
#include <fstream>
#include <string>
#include <ctime>
#include <map>
#include <dirent.h>
using namespace cv;
using namespace std;

// Initialize a variable for PI constant
#define PI 3.14159265

//Initialize all the boolean variables determining state of the program
bool saveBoolean = false;

// Variable to hold most recent corner_set 
std::vector<cv::Point2f> corner_set;
std::vector<std::vector<cv::Point2f> > corner_list;
std::vector<cv::Vec3f> point_set;
std::vector<std::vector<cv::Vec3f> > point_list;


// Headers for some utility function defined below
std::string getTimestamp();
void reset();
void getIndex(vector<int> v, int K, int &index);
void tokenize(std::string const &str, const char delim,std::vector<std::string> &out);

// Main function
int main(int argc, char *argv[]) {
    if( argc < 2) {
        printf("usage: %s <c for compare or v for video> <image1> <image2>\n", argv[0]);
        exit(-1);
    }
    
    // Variables for source and target image file
    char actionName[256];
    char imageName[256];
    char imageNameA[256];
    // Get the image names
    strcpy(actionName, argv[1]);
    

    
    // cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
    // open the video device
    if((strcmp(actionName, "compare") == 0) || (strcmp(actionName, "c") == 0))
    {
        if( argc < 3) {
        printf("usage: %s <c for compare or v for video> <image1> <image2>\n", argv[0]);
        exit(-1);
    }
        strcpy(imageName, argv[2]);
        strcpy(imageNameA, argv[3]);
        cv::VideoCapture *capdev;

        // Read the image path and store them into Mat     
        Mat srcImage = imread(imageName);
        Mat targetImage = imread(imageNameA);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;
        for(;;) {
                // *capdev >> frame; // get a new frame from the camera, treat as a stream
                // Error when images are empty
                if( srcImage.empty() ) {
                    printf("source frame is empty\n");
                    break;
                }
                if( targetImage.empty() ) {
                    printf("target frame is empty\n");
                    break;
                }    
                // Initialize matrices 
                cv::Mat dst;
                cv::Mat dstG;
                cv::Mat dstC;
                // see if there is a waiting keystroke
                char key = cv::waitKey(10);
                // q for quit
                if( key == 'q') 
                {
                    break;
                }
                // Greyscale the image
                // cvtColor( srcImage, dstG, cv::COLOR_BGR2GRAY );
                // int blockSize = 2;
                // int apertureSize = 3;
                // double k = 0.04;
                // dst = Mat::zeros( dstG.size(), CV_32FC1 );
                // cornerHarris( dstG, dst, blockSize, apertureSize, k );
                // Initialize matrices and vector variables for SURF
                std::vector<KeyPoint> keypoints1, keypoints2;
                Mat descriptors1,descriptors2;
                // Initialize detector variable
                Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( 20000 );
                // Detect keypoints on both source and target images
                detector->detectAndCompute( srcImage, noArray(), keypoints1, descriptors1 );
                //cv::drawKeypoints(srcImage,keypoints1,dst,Scalar(0,0,255));
                detector->detectAndCompute( targetImage, noArray(), keypoints2, descriptors2 );
                Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
                std::vector< std::vector<DMatch> > knn_matches;
                // Match the descriptors
                matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
                //-- Filter matches using the Lowe's ratio test
                const float ratio_thresh = 0.7f;
                std::vector<DMatch> good_matches;
                for (size_t i = 0; i < knn_matches.size(); i++)
                {
                    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
                    {
                        good_matches.push_back(knn_matches[i][0]);
                    }
                }
                // Draw the matched images
                Mat img_matches;
                drawMatches( srcImage, keypoints1, targetImage, keypoints2, good_matches, img_matches, Scalar::all(-1),
                            Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                //-- Show detected matches
                imshow("Good Matches", img_matches );
                //cv::imshow("Video", dst);
    }
    
            
    }
    if((strcmp(actionName, "video") == 0) || (strcmp(actionName, "v") == 0))
    {
        cv::VideoCapture *capdev;
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame;
    for(;;) {
            *capdev >> frame; // get a new frame from the camera, treat as a stream
            // Error when images are empty  
            // Initialize matrices 
            cv::Mat dst;
            cv::Mat dstG;
            cv::Mat dstC;
            // see if there is a waiting keystroke
            char key = cv::waitKey(10);
            // q for quit
            if( key == 'q') 
            {
                break;
            }
            // Greyscale the image
            // cvtColor( srcImage, dstG, cv::COLOR_BGR2GRAY );
            // int blockSize = 2;
            // int apertureSize = 3;
            // double k = 0.04;
            // dst = Mat::zeros( dstG.size(), CV_32FC1 );
            // cornerHarris( dstG, dst, blockSize, apertureSize, k );
            // Initialize matrices and vector variables for SURF
            std::vector<KeyPoint> keypoints1, keypoints2;
            Mat descriptors1,descriptors2;
            // Initialize detector variable
            Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( 2000 );
            // Detect keypoints on both source and target images
            detector->detectAndCompute( frame, noArray(), keypoints1, descriptors1 );
            //cv::drawKeypoints(srcImage,keypoints1,dst,Scalar(0,0,255));
            for(int i = 0; i<keypoints1.size();i++)
            {
                cv::circle(frame,keypoints1[i].pt,5,Scalar(0,0,255));
            }
            // Draw the matched images
            Mat img_matches;
            imshow("Video", frame );
            //cv::imshow("Video", dst);
    }
    
            
    }
    
    //delete capdev;
    return(0);
}

// Get current timestamp
std::string getTimestamp()
{
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,80,"%Y%m%d%H%M%S",timeinfo);
    return std::string(buffer);
} 

// Reset the states
void reset()
{
    saveBoolean = false;
} 

// Get index in the vector based on the value
void getIndex(vector<int> v, int K, int &index)
{
    auto it = find(v.begin(), v.end(), K);
  
    // If element was found
    if (it != v.end()) 
    {
      
        // calculating the index
        // of K
        index = it - v.begin();
    }
    else {
        // If the element is not
        // present in the vector
        index=-1;
    }
}

// Utility function to parse and split string
// Found the function from this site
// https://favtutor.com/blogs/split-string-cpp
void tokenize(std::string const &str, const char delim,
            std::vector<std::string> &out)
{
    size_t start;
    size_t end = 0;
 
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
    {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}