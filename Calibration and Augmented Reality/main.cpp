/*
Hao Sheng (Jack) Ning

CS 5330 Computer Vision
Spring 2023
Project 4: Calibration and Augmented Reality

CPP main.cpp file that stores a single function comprised of different parts corresponding to different tasks of the assignment
    1. Detect and Extract Chessboard Corners
    2. Select Calibration Images
    3. Calibrate the camera
    4. Calculate Current Position of the Camera
    5. Project Outside Corners or 3D Axes
    6. Create a Virtual Object

The main program is interacted through command line interfaces with the following format:
main.exe <c for calibration or v for video>
For example:
Task 1-3: main.exe c
 - Press 'e' for corner extraction
 - Press 's' for saving the image
 - Press 'c' for calibration
 - Press 'q' for quit
Task 4-6: main.exe v
Once the corners are detected, the virtual object will appear
The functions return an int to indicate whether the function completed successfully or not
*/

// Include libraries
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
// #include <opencv2/calib3d.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include "include/filter.h"
#include "include/csv_util.h"
#include <chrono>
#include <fstream>
#include <string>
#include <ctime>
#include <map>
#include <dirent.h>
//#include "opencv2/aruco/dictionary.hpp"
#include <opencv2/aruco.hpp>
using namespace cv;
using namespace std;

// Initialize a variable for PI constant
#define PI 3.14159265

//Initialize all the boolean variables determining state of the program
bool extractBoolean = false;
bool saveBoolean = false;
bool calibrationBoolean = false;
bool cornerBoolean = false;
bool printBoolean = false;
bool success = false;

// Variable to hold most recent corner_set, corner_list, corresponding point_set and point_list
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
        printf("usage: %s <video (v) or training mode (t) or k1 compare (ck1) or k2 compaure (ck2)>\n", argv[0]);
        exit(-1);
    }
    
    // Variables for action name
    char actionName[256];

    // Get the action name
    strcpy(actionName, argv[1]);

    // If the action is calibration
    if((strcmp(actionName, "calibration") == 0) || (strcmp(actionName, "c") == 0))
    {
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
        // Initialize and open the video device
        cv::VideoCapture *capdev;
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        // Initialize a matrix variable for frame
        cv::Mat frame;
        for(;;) {
                // Get a new frame from the camera, treat as a stream
                *capdev >> frame; 
                if( frame.empty() ) {
                  printf("frame is empty\n");
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
                // s for save a point 
                else if(key == 's')
                {
                    saveBoolean = true;
                }
                // e for extract
                else if(key == 'e')
                {
                    reset();
                    extractBoolean = true;   
                }
                // c for calibration
                else if(key == 'c')
                {
                    reset();
                    calibrationBoolean = true;   
                }
                // p for print
                else if(key == 'p')
                {
                    reset();
                    printBoolean = true;   
                }
                // If in corner extract state
                if(extractBoolean)
                {
                    // Clear the corner_set
                    corner_set.clear();
                    // grayscale, then call the threshold function and show
                    cvtColor( frame, dstG, cv::COLOR_RGB2GRAY );
                    // Extract corners
                    success = extractCorners(dstG,corner_set);
                    // If the corner extraction was successful
                    if(success)
                    {
                        cv::TermCriteria criteria(1, 30, 0.001);
                        // refining pixel coordinates for given 2d points.
                        cv::cornerSubPix(dstG,corner_set,cv::Size(11,11), cv::Size(-1,-1),criteria);
                        // Clear the point set
                        point_set.clear();
                        // Output the corner numbers and first corner coordinates
                        cout<<"Number of corners:"<<corner_set.size()<<endl;
                        cout<<"First Corner x:"<<corner_set[0].x<<" y:"<<corner_set[0].y<<endl;
                        // Iterate through all points in the corner_set, and 
                        // create their corresponding 3D world point coordinates and add it to the point_set
                        for(int i=0;i<corner_set.size();i++)
                        {
                            // (y,-x,0) (i int divide 9, i modulo 9, 0)
                            cv::Vec3f point(i/9, -1*i%9, 0);
                            point_set.insert(point_set.end(),point);
                        }
                        // Output the first corner 3D world coordinates
                        cout<<"First Point x:"<<point_set[0]<<endl;
                        // Draw the corners out on frame
                        cv::drawChessboardCorners(frame,cv::Size(9,6),corner_set,success);
                    }
                    frame.copyTo(dst);
                    // Draw it out
                    cv::imshow("Video", frame);                 
                }
                // If in calibration state
                if(calibrationBoolean)
                {
                    if(corner_list.size()<5)
                    {
                        cout<<"Only "<<corner_list.size()<<" corners saved, minimum 5 needed, please save more."<<endl;
                        reset();
                        extractBoolean = true;  
                    }
                    else{
                        // Initialize camera_matrix and its values
                        cv::Mat camera_matrix = Mat::zeros(3, 3, CV_64FC1);
                        camera_matrix.at<double>(0,0)=1;
                        camera_matrix.at<double>(0,2)=(double)frame.cols/2;
                        camera_matrix.at<double>(1,1)=1;
                        camera_matrix.at<double>(1,2)=(double)frame.rows/2;
                        camera_matrix.at<double>(2,2)=1;
                        // Initialize variables for distortion coefficients, rotation and translation matrices
                        cv::Mat distCoeffs=Mat::zeros(1,5,CV_64FC1);
                        cv::Mat R,T;
                        // Output the camera matrix and distortion coefficients
                        std::cout << "Initial CameraMatrix : " << camera_matrix << std::endl;
                        std::cout << "Initial DistCoeffs : " << distCoeffs << std::endl;
                        // Call the calibrate camera and its return value is the reprojection_error
                        float reprojection_error = cv::calibrateCamera(point_list,corner_list,Size(frame.rows,frame.cols),camera_matrix,distCoeffs,R,T);
                        //Compute mean of reprojection error
                        // float tot_error=0;
                        // float total_points=0;
                        
                        // for(int i = 0; i<point_list.size();i++)
                        // {
                        //     std::vector<cv::Point2f> reprojected_image_points;
                        //     cv::projectPoints(point_list[i],R.row(i),T.row(i),camera_matrix,distCoeffs,reprojected_image_points);
                        //     for(int j = 0; j<reprojected_image_points.size();j++)
                        //     {
                        //         float differencex = reprojected_image_points[j].x-corner_list[i][j].x;
                        //         float differencey = reprojected_image_points[j].y-corner_list[i][j].y;
                        //         total_points++;
                        //         tot_error+=(differencex*differencex+differencey*differencey);
                        //     }
                        // }
                        //cout<<"Reprojection Error computed manually:"<<tot_error/total_points<<endl;
                        cout<<"Reprojection Error computed by opencv:"<<reprojection_error<<endl;
                        // Output before and after calibration matrices and distortion coefficients
                        std::cout << "After Calibration CameraMatrix : " << camera_matrix << std::endl;
                        std::cout << "After Calibration DistCoeffs : " << distCoeffs << std::endl;
                        // Prompt the user to save the calibration matrices and distortion coefficients
                        char x;
                        cout << "Would you like to write this CameraMatrix and Distortion Coefficients to a file? Yes(y) or No(n)"; // Type a number and press enter
                        cin >> x; // Get user input from the keyboard
                        char keyInCalibration = cv::waitKey(10);
                        // If user reply is yes
                        if(x == 'y')
                        {
                            // Initialize ofstream variable 
                            ofstream myfile;
                            // Open the named file
                            myfile.open ("intrinsicParameters.txt");
                            // Iterate all values inside camera matrix and write them to file
                            myfile << "cameraMatrix:";
                            for(int i=0;i<camera_matrix.rows;i++)
                                for(int j=0;j<camera_matrix.cols;j++)
                                {
                                    if(i==camera_matrix.rows-1 && j==camera_matrix.cols-1)
                                        myfile<<camera_matrix.at<double>(i,j)<<"|"<<endl;
                                    else
                                        myfile<<camera_matrix.at<double>(i,j)<<","; 
                                }
                            // Iterate all values inside distortion coefficients and write them to file
                            myfile << "distCoeffs:";
                            for(int i=0;i<distCoeffs.rows;i++)
                                for(int j=0;j<distCoeffs.cols;j++)
                                {
                                    if(j!=(distCoeffs.cols-1))
                                        myfile<<distCoeffs.at<double>(i,j)<<",";
                                    else
                                        myfile<<distCoeffs.at<double>(i,j)<<"|"<<endl;
                                } 
                            // Close the file
                            myfile.close();
                            reset();
                            extractBoolean = true;   
                        }
                        // if user reply is no, keep extracting corners
                        else if(x == 'n')
                        {
                            reset();
                            extractBoolean = true;   
                        }
                        // q for quit
                        if( keyInCalibration == 'q') 
                        {
                            break;
                        }

                    }
                    
                }
                else
                {
                    frame.copyTo(dst);
                    cv::imshow("Video", dst);
                }
                // If user wants to save points
                if(saveBoolean)
                {
                    // Save points only corner extraction was successful
                    if(success)
                    {
                        // Push corners and points to their list
                        corner_list.push_back(corner_set);
                        point_list.push_back(point_set);
                        waitKey(1);
                    }
                    // Flip the state boolean
                    saveBoolean=false;
                }
                if(printBoolean)
                {
                    cv::imwrite("save_"+getTimestamp()+".jpg", frame); // A JPG FILE IS BEING SAVED
                    printBoolean=false;
                    extractBoolean=true;
                }
        }
        // Shut down the video capturing
        delete capdev;

    }
    // If the action is real time video
    else if((strcmp(actionName, "video") == 0) || (strcmp(actionName, "v") == 0))
    {
        // Inititalize variables for camera matrix and distortion coefficients
        cv::Mat camera_matrix = Mat::zeros(3, 3, CV_64FC1);
        cv::Mat distCoeffs;
        // Variable to hold most recent corner_set and point_set
        std::vector<cv::Point2f> corner_set;
        std::vector<cv::Vec3f> point_set;
        // Initialize a read stream variable
        std::ifstream file("intrinsicParameters.txt");
        if(!file)return 0;
        std::string line;
        // Read the file line by line
        while (std::getline(file, line, '\0')){
            for(char ascii : line){
                //std::cout<<(int)ascii << " ";
            }
        }
        // Parse out \n
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.cend());
        // Parse out the strings based on following delimiters
        const char delimL = '|';
        const char delimC = ':';
        std::vector<std::string> out;
        tokenize(line, delimL, out);
        for(int i = 0; i<out.size();i++)
        {
            std::vector<std::string> out1;
            std::size_t pos = 0;
            tokenize(out[i],delimC,out1);
            // convert ',' to ' '
            stringstream ss(out1[1]);
            std::vector< double > vd;
            double d = 0.0;
            while (ss.good()) {
                // Push all doubles to double list
                string substr;
                getline(ss, substr, ',');
                stringstream sss(substr);
                if(sss>>d)
                    vd.push_back(d);
            }
            // Iterate through all vd list and push corresponding
            // camera matrix and distortion coefficient double values to corresponding matrices
            if(out1[0]=="cameraMatrix")
            {
                for(int i=0;i<vd.size();i++)
                {
                    int intDivide = i/camera_matrix.rows;
                    int mod = i-(intDivide*camera_matrix.rows);
                    camera_matrix.at<double>(intDivide,mod)=vd[i];
                }
            }
            else if(out1[0]=="distCoeffs")
            {
                distCoeffs = Mat(1,vd.size(),CV_64FC1);
                for(int i=0;i<vd.size();i++)
                    distCoeffs.at<double>(0,i)=vd[i];
            }
        }
        // Output the matrices
        cout<<"Camera Matrix:"<<camera_matrix<<endl;
        cout<<"Dist Coeffs:"<<distCoeffs<<endl;
        
        bool poseBoolean = false;
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
        // Initialize and open the video device
        cv::VideoCapture *capdev;
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }
        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;
        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }  
                // Initialize matrices 
                cv::Mat dst;
                cv::Mat dstG;
                cv::Mat dstC;
                std::vector<cv::Point2f> projected_image_points;
                // see if there is a waiting keystroke
                char key = cv::waitKey(10);
                corner_set.clear();
                // grayscale, then call the threshold function and show
                cvtColor( frame, dstG, cv::COLOR_RGB2GRAY );
                // Extract corners
                bool success = extractCorners(dstG,corner_set);
                if(success)
                {
                    cv::TermCriteria criteria(1, 30, 0.001);
                    // refining pixel coordinates for given 2d points.
                    cv::cornerSubPix(dstG,corner_set,cv::Size(11,11), cv::Size(-1,-1),criteria);
                    // Clear the point_set
                    point_set.clear();
                    // Output the result
                    cout<<"Number of corners:"<<corner_set.size()<<endl;
                    cout<<"First Corner x:"<<corner_set[0].x<<" y:"<<corner_set[0].y<<endl;
                    // Iterate through all points in the corner_set, and 
                    // create their corresponding 3D world point coordinates and add it to the point_set
                    for(int i=0;i<corner_set.size();i++)
                    {
                        // (y,-x,0) (i int divide 9, i modulo 9, 0)
                        cv::Vec3f point(i/9, -1*i%9, 0);
                        point_set.insert(point_set.end(),point);
                    }
                    // Output first point
                    cout<<"First Point x:"<<point_set[0]<<endl;
                    // Initialize variables for rotation and translation matrices
                    cv::Mat R,T;
                    // Compute the pose
                    solvePnP(point_set,corner_set,camera_matrix,distCoeffs,R,T);
                    // Output the result
                    cout<<"Rotation:"<<R<<endl;
                    cout<<"Translation:"<<T<<endl;
                    // Initialize vector to hold 3D point set
                    std::vector<cv::Vec3f> to_be_projected_point_set;
                    // Upper Right Corner
                    cv::Vec3f point1(point_set[0][0]-1,point_set[0][1]+1,0);
                    // Upper Left Corner
                    cv::Vec3f point2(point_set[point_set.size()-1][0]+1,point_set[0][1]+1,0);
                    // Lower Right Corner
                    cv::Vec3f point3(point_set[0][0]-1,point_set[point_set.size()-1][1]-1,0);
                    // Lower Left Corner
                    cv::Vec3f point4(point_set[point_set.size()-1][0]+1,point_set[point_set.size()-1][1]-1,0);
                    // Push the corners to the list
                    to_be_projected_point_set.push_back(point1);
                    to_be_projected_point_set.push_back(point2);
                    to_be_projected_point_set.push_back(point3);
                    to_be_projected_point_set.push_back(point4);
                    // Project the 3D points onto 2D image plane
                    projectPoints(to_be_projected_point_set,R,T,camera_matrix,distCoeffs,projected_image_points);
                    // Draw the corners out as circles
                    for(int i=0;i<projected_image_points.size();i++)
                    {
                        //Color of the circle
                        Scalar line_Color;
                        if(i==0)
                            line_Color=Scalar(255,0,0);
                        else if(i==1)
                            line_Color=Scalar(0,255,0);
                        else if(i==2)
                            line_Color=Scalar(0,0,255);
                        else if(i==3)
                            line_Color=Scalar(128, 128, 128);
                        cv::circle(frame, Point(projected_image_points[i].x,projected_image_points[i].y),5, line_Color, 1);
                    }
                    if(cornerBoolean==false)
                    {
                        // Project Virtual Object
                        // Initialize vectors for 2D and 3D points
                        std::vector<std::vector<cv::Vec3f>> to_be_projected_vo_point_set(30);
                        std::vector<cv::Vec3f> to_be_projected_vc_point_set;
                        std::vector<std::vector<cv::Point2f>> projected_vo_image_points(30);
                        std::vector<cv::Point2f> projected_vc_image_points;
                        // Initialize all the points
                        cv::Vec3f pointVO1(1,-2,0);
                        cv::Vec3f pointVO2(5,-2,0);
                        cv::Vec3f pointVO3(5,-6,0);
                        cv::Vec3f pointVO4(1,-6,0);
                        cv::Vec3f pointVO5(1,-2,-4);
                        cv::Vec3f pointVO6(5,-2,-4);
                        cv::Vec3f pointVO7(5,-6,-4);
                        cv::Vec3f pointVO8(1,-6,-4);
                        cv::Vec3f pointVO9(3,-4,-6);
                        cv::Vec3f pointVO10(2,-3,-6);
                        cv::Vec3f pointVO11(1.5,-3,-6);
                        cv::Vec3f pointVO12(1.5,-3.5,-6);
                        cv::Vec3f pointVO13(2,-3.5,-6);
                        cv::Vec3f pointVO14(2,-3,-4);
                        cv::Vec3f pointVO15(1.5,-3,-4);
                        cv::Vec3f pointVO16(1.5,-3.5,-4);
                        cv::Vec3f pointVO17(2,-3.5,-4);
                        cv::Vec3f pointVC1(1.75,-3.25,-6.5);
                        cv::Vec3f pointVC2(1.75,-3.25,-7);
                        cv::Vec3f pointVC3(1.75,-3.25,-7.5);
                        cv::Vec3f pointVO18(2.5,-6,0);
                        cv::Vec3f pointVO19(2.5,-6,-1);
                        cv::Vec3f pointVO20(3.5,-6,-1);
                        cv::Vec3f pointVO21(3.5,-6,0);
                        cv::Vec3f pointVO22(3,-6,0);
                        cv::Vec3f pointVO23(3,-6,-1);
                        // Add all points to point_set
                        to_be_projected_vo_point_set[0].push_back(pointVO1);
                        to_be_projected_vo_point_set[0].push_back(pointVO2);
                        to_be_projected_vo_point_set[0].push_back(pointVO3);
                        to_be_projected_vo_point_set[0].push_back(pointVO4);
                        to_be_projected_vo_point_set[1].push_back(pointVO5);
                        to_be_projected_vo_point_set[1].push_back(pointVO6);
                        to_be_projected_vo_point_set[1].push_back(pointVO7);
                        to_be_projected_vo_point_set[1].push_back(pointVO8);
                        to_be_projected_vo_point_set[2].push_back(pointVO1);
                        to_be_projected_vo_point_set[2].push_back(pointVO5);
                        to_be_projected_vo_point_set[3].push_back(pointVO2);
                        to_be_projected_vo_point_set[3].push_back(pointVO6);
                        to_be_projected_vo_point_set[4].push_back(pointVO3);
                        to_be_projected_vo_point_set[4].push_back(pointVO7);
                        to_be_projected_vo_point_set[5].push_back(pointVO4);
                        to_be_projected_vo_point_set[5].push_back(pointVO8);
                        to_be_projected_vo_point_set[6].push_back(pointVO5);
                        to_be_projected_vo_point_set[6].push_back(pointVO9);
                        to_be_projected_vo_point_set[7].push_back(pointVO6);
                        to_be_projected_vo_point_set[7].push_back(pointVO9);
                        to_be_projected_vo_point_set[8].push_back(pointVO7);
                        to_be_projected_vo_point_set[8].push_back(pointVO9);
                        to_be_projected_vo_point_set[9].push_back(pointVO8);
                        to_be_projected_vo_point_set[9].push_back(pointVO9);
                        to_be_projected_vo_point_set[10].push_back(pointVO10);
                        to_be_projected_vo_point_set[10].push_back(pointVO11);
                        to_be_projected_vo_point_set[10].push_back(pointVO12);
                        to_be_projected_vo_point_set[10].push_back(pointVO13);
                        to_be_projected_vo_point_set[11].push_back(pointVO10);
                        to_be_projected_vo_point_set[11].push_back(pointVO14);
                        to_be_projected_vo_point_set[12].push_back(pointVO11);
                        to_be_projected_vo_point_set[12].push_back(pointVO15);
                        to_be_projected_vo_point_set[13].push_back(pointVO12);
                        to_be_projected_vo_point_set[13].push_back(pointVO16);
                        to_be_projected_vo_point_set[14].push_back(pointVO13);
                        to_be_projected_vo_point_set[14].push_back(pointVO17);
                        to_be_projected_vc_point_set.push_back(pointVC1);
                        to_be_projected_vc_point_set.push_back(pointVC2);
                        to_be_projected_vc_point_set.push_back(pointVC3);
                        to_be_projected_vo_point_set[15].push_back(pointVO18);
                        to_be_projected_vo_point_set[15].push_back(pointVO19);
                        to_be_projected_vo_point_set[15].push_back(pointVO20);
                        to_be_projected_vo_point_set[15].push_back(pointVO21);
                        to_be_projected_vo_point_set[16].push_back(pointVO22);
                        to_be_projected_vo_point_set[16].push_back(pointVO23);
                        // Iterate through all points in the point_set vectors
                        // and all them as lines
                        for(int i=0;i<to_be_projected_vo_point_set.size();i++)
                        {
                            if(to_be_projected_vo_point_set[i].size()>0)
                            {
                                projectPoints(to_be_projected_vo_point_set[i],R,T,camera_matrix,distCoeffs,projected_vo_image_points[i]);
                                for(int j = 0; j<projected_vo_image_points[i].size();j++)
                                {
                                    if(projected_vo_image_points[i].size()>0)
                                    {
                                        Scalar line_Color(255, 0, 0);//Color of the line
                                        if(j!=(projected_vo_image_points[i].size()-1))
                                            cv::line(frame, projected_vo_image_points[i][j], projected_vo_image_points[i][j+1], line_Color, 1);
                                        else
                                            cv::line(frame, projected_vo_image_points[i][j], projected_vo_image_points[i][0], line_Color, 1);  
                                    }
                                }
                            }
                        }
                        // Iterate through all points in the point_set vectors
                        // and all them as lines
                        projectPoints(to_be_projected_vc_point_set,R,T,camera_matrix,distCoeffs,projected_vc_image_points);
                        for(int i=0; i<projected_vc_image_points.size();i++)
                        {
                            Scalar line_Color(0, 0, 255);//Color of the circle
                            cv::circle(frame,Point(projected_vc_image_points[i].x,projected_vc_image_points[i].y),i*4,line_Color);
                        }
                    }
                    cv::imshow("Video", frame);
                }
                else
                {
                    cv::imshow("Video", frame);
                }
                
                // q for quit
                if( key == 'q') 
                {
                    break;
                }
                else if(key == 'o')
                {
                    reset();
                    cornerBoolean=true;
                }
                // p for poseBoolean of 
                if( key == 'p') 
                {
                    poseBoolean = true;
                }
                if(poseBoolean)
                {
                                    
                }
                // Save points to corner and point list
                if(saveBoolean)
                {
                    corner_list.push_back(corner_set);
                    point_list.push_back(point_set);
                    saveBoolean=false;
                    waitKey(1);
                }
        }
        delete capdev;

    }
    // Error 
    else
    {
        printf("usage: %s <video (v) or training mode (t) or k1 compare (ck1) or k2 compaure (ck2)>\n", argv[0]);
        exit(-1);
    } 
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
    extractBoolean = false;
    calibrationBoolean = false;
    cornerBoolean=false;
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