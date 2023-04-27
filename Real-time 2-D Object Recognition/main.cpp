/*
Hao Sheng (Jack) Ning

CS 5330 Computer Vision
Spring 2023
Project 3: Real-time 2-D Object Recognition

CPP main.cpp file that stores a single function comprised of different parts corresponding to different tasks of the assignment
    1. Threshold the input video
    2. Clean up the binary image
    3. Segment the image into regions
    4. Compute features for each major region
    5. Collect training data
    6. Classify new images (with 1k and 2k)
    7. Extension

The main program is interacted through command line interfaces with the following format:
main.exe <target path> <image database directory path> <feature type> <matching method> <numberOfMatches>
For example:
Task 1-4: main.exe v
Press 't' for threshold, 'm' for morphological, 'e' for segment, and 'f' for region feature extractions
Training: main.exe t
Classify 1k: main.exe ck1
Classify 2k: main.exe ck2
The functions return an int to indicate whether the function completed successfully or not
*/

// Include libraries
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
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
bool thresholdBoolean = false;
bool saveBoolean = false;
bool monphologicalBoolean = false;
bool segmentBoolean = false;
bool featureBoolean = false;

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
    // Database feature set file name
    char csv[] = "objects.csv";
    
    // Variables for action name and training type
    char actionName[256];
    char trainingType[256];

    // Get the action name
    strcpy(actionName, argv[1]);

    // If action is to compare 
    if((strcmp(actionName, "comparek1") == 0) || (strcmp(actionName, "ck1") == 0) || (strcmp(actionName, "comparek2") == 0) || (strcmp(actionName, "ck2") == 0))
    {
        // Variables for vectors of regions, imageName, imageLabel
        vector<pair<int,int>> regions;
        string fileName, fileLabel;
        // Map variables to store feature vectors:P angles, percentageFilled, and BB Ratio
        map<string, float> mapOfAngles;
        map<string, float> mapOfPercentageFilled;
        map<string, float> mapOfBBRatio;  

        // Vector variable to store image names and feature vector data
        vector<char *> fileNames;
        vector<vector<float>> featureData;
        // Read image names and feature vector data from csv file and store them in variables
        read_image_data_csv(csv,fileNames,featureData,0);

        // Prompt the user to enter the target image path
        cout<<"\tPlease enter the image you want to compare:";
        // Store the input
        cin >> fileName;
        cout<<fileName<<"Processing target image:"<<fileName<<endl;
        // Get the image based on the path
        Mat databaseImage = imread(fileName);
        //If image does not exist, output an error message and terminate the program
        if (!databaseImage.data) 
        {
            cout<<fileName<<"file does not exist."<<endl;
            return 1;
        }

        // Vector to hold target features
        vector<float> targetFeatureVectors(30,0);
        // Matrices
        cv::Mat dst;
        cv::Mat dstT;
        cv::Mat dstM;
        cv::Mat temp=  Mat::zeros(databaseImage.size(), CV_16SC3);
        dstM =  Mat::zeros(databaseImage.size(), CV_16SC3);
        // Grayscale, threshold and monphological
        cvtColor( databaseImage, dstT, cv::COLOR_RGB2GRAY );
        threshold(dstT,temp);
        monphological(temp,dstM);
        // Variables for labels, stats, and centroids
        cv::Mat labels, stats, centroids;
        int connectivity = 8; // or 4
        // Call OpenCV's connectedComponentAnalysis
        int label_count = cv::connectedComponentsWithStats(dstM, labels, stats, centroids, connectivity);
        // Iterate through all labels and push region ID and its corresponding area to regions
        for (int i = 0; i < label_count; i++)
        {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            regions.push_back(make_pair(area,i));
        }
        // Sort the region vectors
        sort(regions.rbegin(), regions.rend());
        // Push regions to regionsByOrder map
        vector<int> regionsByOrder(regions.size(),0);
        for (int index=0; index<(regions.size()); index++)
        {
            regionsByOrder.insert(regionsByOrder.begin()+index,regions[index].second);
        }   
        // Iterate through 10 or number of regions and call moments and rotatedArea function     
        for(int i =1;i<min(10,label_count);i++)
        {
            float angle;
            Moments m;
            RotatedRect boundingRect;
            float orientedCentralMoments;
            computeMomentsFeautures(labels,regionsByOrder[i],m,boundingRect,orientedCentralMoments);
            //m.mu2
            angle=atan2(2*m.mu11,m.mu20-m.mu02)*180/PI;
            int area = stats.at<int>(regionsByOrder[i], cv::CC_STAT_AREA);
            // Push oriented central moments to feature vectors
            targetFeatureVectors[0+i]=orientedCentralMoments;;
            double percentageRatio = ((double)area)/(boundingRect.size.area());
            double bbRatio = (double)(boundingRect.size.height)/boundingRect.size.width;
            // Push percentage filled and bb ratio to feature vectors
            targetFeatureVectors[10+i]=percentageRatio;
            targetFeatureVectors[20+i]=bbRatio;
        }
        // Set up map variables 
        vector< pair <float, string> > vectOfImages;
        set<String>setOfLabels;
        map<String,float> mapOfLabels;
        vector< pair <float, string> > vectOfLabels;
        int featureDataSize = featureData.size();
        // Iterate through all training objects extracted from the DB file
        for(int i = 0; i<featureDataSize; i++)
        {
            float sumDifferenceSquaredAngle=0;
            float differenceAngle = 0;
            // Iterate through all feature vectors
            vector<float> trainingDataSet = featureData[i];
            for (int j =1; j<2;j++)
            {
                // If the oriented central moments feature vector isn't 0, inset it to the map
                if(trainingDataSet[j]!=0 && targetFeatureVectors[j]!=0)
                {
                    mapOfAngles.insert({fileNames[i],trainingDataSet[j]});
                }
            }
            for (int j =1; j<2;j++)
            {
                // If the percentage filled and bb ratop feature vector isn't 0, inset it to the map
                if(trainingDataSet[j+10]!=0 && targetFeatureVectors[j+10]!=0)
                {
                    mapOfPercentageFilled.insert({fileNames[i],trainingDataSet[10+j]});
                }
                if(trainingDataSet[j+20]!=0 && targetFeatureVectors[j+20]!=0)
                {
                    mapOfBBRatio.insert({fileNames[i],trainingDataSet[20+j]});
                }
            }
        }
        // Initialize Standard Deviation variables
        double sdAngle=0.0;
        double sdPercentageFilled=0.0;
        double sdBBRatio=0.0;
        // Call utility function in filter.cpp to use the feature vector map to 
        // calculate standard deviation for each feature vector
        calculateStandardDeviation(mapOfAngles,sdAngle);
        calculateStandardDeviation(mapOfPercentageFilled,sdPercentageFilled);
        calculateStandardDeviation(mapOfBBRatio,sdBBRatio);
        // Iterate through all images extracted from the DB file
        for(int i = 0;i<fileNames.size();i++)
        {
            // Use Euclidean distance to calculate the number of standard deviations for each feature vector
            double numOfSDAngle = abs(mapOfAngles[fileNames[i]]-targetFeatureVectors[1])/sdAngle;
            double numOfSDPercentageFilled = abs(mapOfPercentageFilled[fileNames[i]]-targetFeatureVectors[11])/sdPercentageFilled;
            double numOfSDBBRatio = abs(mapOfBBRatio[fileNames[i]]-targetFeatureVectors[21])/sdBBRatio;
            // Push the summed distance and the image name to the vector
            vectOfImages.push_back(make_pair(1*numOfSDAngle+2*numOfSDPercentageFilled+2*numOfSDBBRatio,fileNames[i]));
        }
        // Sort the vector according to the distance in ascending order
        sort(vectOfImages.begin(), vectOfImages.end());
        // If action name is ck1
        if((strcmp(actionName, "comparek1") == 0) || (strcmp(actionName, "ck1") == 0))
        {
            // Output the closest match's label
            cout<<" TOP MATCHES"<<endl;
            cout<<"----------------------------------"<<endl;
            for (int index=0; index<1; index++)
            {
                // Use the following the tokenize utility function to parse out the label
                std::string s = vectOfImages[index].second;
                const char delim = ':';
                std::vector<std::string> out;
                tokenize(s, delim, out);
                // Out the image label
                cout <<"Closest match:"<< out[1]<< endl;
            }
            cout<<"----------------------------------"<<endl;
        }
        // If action name is ck2
        else if((strcmp(actionName, "comparek2") == 0) || (strcmp(actionName, "ck2") == 0))
        {
            // Iterate through all images, and insert the parsed label to the setOfLabels
            for (int index=0; index<fileNames.size(); index++)
            {
                std::string s = vectOfImages[index].second;
                const char delim = ':';
                std::vector<std::string> out;
                tokenize(s, delim, out);
                setOfLabels.insert(out[1]);
            }
            // Iterate through the set and initialize a mapOfLabels with value of 0
            set<String>:: iterator it;
            for( it = setOfLabels.begin(); it!=setOfLabels.end(); ++it){
                String val = *it;
                mapOfLabels.insert({val,0});
            }
            // Iterate through all, and add distances to corresponding label value
            for (int index=0; index<vectOfImages.size(); index++)
            {
                std::string s = vectOfImages[index].second;
                const char delim = ':';
                std::vector<std::string> out;
                tokenize(s, delim, out);
                std::map<String,float>::iterator itr;
                itr = mapOfLabels.find(out[1]);
                if (itr != mapOfLabels.end())
                        itr->second += vectOfImages[index].first;
            }
            // Iterate through all map element and push item to vectors
            map<String, float>::iterator mapit;
            for (mapit = mapOfLabels.begin(); mapit != mapOfLabels.end(); mapit++)
            {
                vectOfLabels.push_back(make_pair(mapit->second ,mapit->first));
            }
            // Sort the vector
            sort(vectOfLabels.begin(), vectOfLabels.end());
            // Output the 2 closest labels
            cout<<" TOP K-2 MATCHES"<<endl;
            cout<<"----------------------------------"<<endl;
            for (int index=0; index<2; index++)
            {
                cout <<"Closest match #"<<index+1<<":"<< vectOfLabels[index].second <<":"<< vectOfLabels[index].first<<endl;
            }
            cout<<"----------------------------------"<<endl;
        }
    }
    // If the action is to train objects
    else if((strcmp(actionName, "training") == 0) || (strcmp(actionName, "t") == 0))
    {
        // Initialize variables for regions, fileName, and fileLabel
        vector<pair<int,int>> regions;
        string fileName, fileLabel;
        // Ask the user if they want to enter the objects individually or by directory
        cout<<"\tDo you want to train the images individually (i) or by directory (d)";
        // Store the input
        cin >> trainingType;
        // If the answer is individually
        if(strcmp(trainingType, "i") == 0)
        {
            // Ask the user for training object file
            cout<<"\tPlease enter the training file:";
            // Store the name
            cin >> fileName;
            cout<<fileName<<"Processing database image:"<<fileName<<endl;
            // Get the image based on the path
            Mat databaseImage = imread(fileName);

            //If image does not exist, output an error message and terminate the program
            if (!databaseImage.data) 
            {
                cout<<fileName<<"file does not exist."<<endl;
                return 1;
            }
            // Prompt the user for object label
            cout<<"Please enter the training file's label:";
            cin >> fileLabel;
            // Initialize feature vectors and matrices
            vector<float> featureVectors(30,0);
            cv::Mat dst;
            cv::Mat dstT;
            cv::Mat dstM;
            cv::Mat temp=  Mat::zeros(databaseImage.size(), CV_16SC3);
            dstM =  Mat::zeros(databaseImage.size(), CV_16SC3);
            // Grayscale, threshold and monphological
            cvtColor( databaseImage, dstT, cv::COLOR_RGB2GRAY );
            threshold(dstT,temp);
            monphological(temp,dstM);
            cv::Mat labels, stats, centroids;
            int connectivity = 8; // or 4
            // Call OpenCV's connectedComponentAnalysis
            int label_count = cv::connectedComponentsWithStats(dstM, labels, stats, centroids, connectivity);
            // Iterate through all labels and push region ID and its corresponding area to regions
            for (int i = 0; i < label_count; i++)
            {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                regions.push_back(make_pair(area,i));
            }
            // Sort the vectors
            sort(regions.rbegin(), regions.rend());
            vector<int> regionsByOrder(regions.size(),0);
            // Push regions to regionsByOrder map
            for (int index=0; index<(regions.size()); index++)
            {
                regionsByOrder.insert(regionsByOrder.begin()+index,regions[index].second);
            }        
            int orderRegionSize=regionsByOrder.size();
            // Iterate through 10 or number of regions and call moments and rotatedArea function
            for(int i =0;i<min(10,orderRegionSize);i++)
            {
                float angle;
                Moments m;
                RotatedRect boundingRect;
                float orientedCentralMoments;
                // Compute and push oriented central moments to feature vectors
                computeMomentsFeautures(labels,regionsByOrder[i],m,boundingRect,orientedCentralMoments);
                angle=atan2(2*m.mu11,m.mu20-m.mu02)*180/PI;
                double oreintedCentralMoments = 0.0;
                int area = stats.at<int>(regionsByOrder[i], cv::CC_STAT_AREA);
                featureVectors[0+i]=orientedCentralMoments;
                double percentageRatio = ((double)area)/(boundingRect.size.area());
                double bbRatio = (double)(boundingRect.size.height)/boundingRect.size.width;
                // Push percentage filled and bb ratio to feature vectors
                featureVectors[10+i]=percentageRatio;
                featureVectors[20+i]=bbRatio;
            }
            const int length = fileName.length();
            char* char_array = new char[length + 1];
            // Append the filename, label and vector data to csv file
            strcpy(char_array, fileName.c_str());
            std::strcat(char_array,(":"+fileLabel).c_str());
            append_image_data_csv(csv,char_array,featureVectors,0);
            cout<<fileName<<"File processed."<<endl;
        }
        // If the trainingType is by directory
        else if(strcmp(trainingType, "d") == 0)
        {
            // Ask the user for directory path & label and store the input
            cout<<"\tPlease enter the training directory path:";
            cin >> fileName;
            cout<<"Please enter the training file's label:";
            cin >> fileLabel;
            char dirname[256];
            struct dirent *dp;
            char buffer[256];
            DIR *dirp;
            // open the directory
            strcpy(dirname, fileName.c_str());
            dirp = opendir( dirname );
            // If the directory is invalid
            if( dirp == NULL) {
                printf("Cannot open directory %s\n", dirname);
                exit(-1);
            }
            // loop over all the files in the image file listing
            while( (dp = readdir(dirp)) != NULL ) 
            {
                // Check for image file type
                 if( strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".jpeg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".jfif") ||
            strstr(dp->d_name, ".webp") ||
            strstr(dp->d_name, ".tif") ) 
            {
                printf("processing image file: %s\n", dp->d_name);

                // build the overall filename
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);

                printf("full path name: %s\n", buffer);
                Mat databaseImage = imread(buffer);

                //If image does not exist, output an error message and terminate the program
                if (!databaseImage.data) 
                {
                    cout<<fileName<<"file does not exist."<<endl;
                    return 1;
                } 
                // Initialize vectors and matrices
                vector<float> featureVectors(30,0);
                cv::Mat dst;
                cv::Mat dstT;
                cv::Mat dstM;
                cv::Mat temp=  Mat::zeros(databaseImage.size(), CV_16SC3);
                dstM =  Mat::zeros(databaseImage.size(), CV_16SC3);
                // Grayscale, threshold and monphological
                cvtColor( databaseImage, dstT, cv::COLOR_RGB2GRAY );
                threshold(dstT,temp);
                monphological(temp,dstM);
                regions.clear();
                // Variables for labels, stats, and centroids
                cv::Mat labels, stats, centroids;
                int connectivity = 8; // or 4
                // Call OpenCV's connectedComponentAnalysis
                int label_count = cv::connectedComponentsWithStats(dstM, labels, stats, centroids, connectivity);
                // Iterate through all labels and push region ID and its corresponding area to regions
                for (int labelIndex = 0; labelIndex < label_count; labelIndex++)
                {
                    int area = stats.at<int>(labelIndex, cv::CC_STAT_AREA);
                    regions.push_back(make_pair(area,labelIndex));
                }
                // Sort the region vector
                sort(regions.rbegin(), regions.rend());
                vector<int> regionsByOrder(regions.size(),0);
                // Insert all regions to regionByOrder map
                for (int index=0; index<(regions.size()); index++)
                {
                    regionsByOrder.insert(regionsByOrder.begin()+index,regions[index].second);
                }      
                int orderedRegionSize = regionsByOrder.size();  
                // Iterate through 10 or number of regions and call moments and rotatedArea function
                for(int i =0;i<min(10,orderedRegionSize);i++)
                {
                    float angle;
                    Moments m;
                    RotatedRect boundingRect;
                    float orientedCentralMoments;
                    // Compute Push oriented central moments to feature vectors
                    computeMomentsFeautures(labels,regionsByOrder[i],m,boundingRect,orientedCentralMoments);
                    angle=atan2(2*m.mu11,m.mu20-m.mu02)*180/PI;
                    double oreintedCentralMoments = 0.0;
                    int area = stats.at<int>(regionsByOrder[i], cv::CC_STAT_AREA);
                    featureVectors[0+i]=orientedCentralMoments;
                    double percentageRatio = ((double)area)/(boundingRect.size.area());
                    double bbRatio = (double)(boundingRect.size.height)/boundingRect.size.width;
                    // Push percentage filled and bb ratio to feature vectors
                    featureVectors[10+i]=percentageRatio;
                    featureVectors[20+i]=bbRatio;
                }
                const int length = fileName.length();
                char* char_array = new char[length + 1];
                // copying the contents of the
                // string to char array
                strcpy(char_array, fileName.c_str());
                std::strcat(char_array,(":"+fileLabel).c_str());
                std::strcat(buffer,(":"+fileLabel).c_str());
                // Append the file name, label and feature data to csv
                append_image_data_csv(csv,buffer,featureVectors,0);
                featureVectors.clear();
                cout<<fileName<<"File processed."<<endl;
                }   
            }
        }
    }
    // If the action is real time video
    else if((strcmp(actionName, "video") == 0) || (strcmp(actionName, "v") == 0))
    {
        cv::VideoCapture *capdev;
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
        // open the video device
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
                cv::Mat dstT;
                cv::Mat dstM;
                // see if there is a waiting keystroke
                char key = cv::waitKey(10);
                // q for quit
                if( key == 'q') 
                {
                    break;
                }
                // s for save image
                else if(key == 's')
                {
                    saveBoolean = true;
                }
                // t for threshold
                else if(key == 't')
                {
                    reset();
                    thresholdBoolean = true;   
                }
                // m for monphological
                else if(key == 'm')
                {
                    reset();
                    monphologicalBoolean = true;   
                }
                // e for segment
                else if(key == 'e')
                {
                    reset();
                    segmentBoolean = true;   
                }
                // f for feature 
                else if(key == 'f')
                {
                    reset();
                    featureBoolean = true;   
                }
                if(thresholdBoolean)
                {
                    // grayscale, then call the threshold function and show
                    cvtColor( frame, dstT, cv::COLOR_RGB2GRAY );
                    threshold(dstT,dst);
                    cv::imshow("Video", dst);                 
                }
                else if(monphologicalBoolean)
                {
                    // grayscale, then call the threshold function and 
                    // call monphological function and show
                    cv::Mat temp= Mat::zeros(frame.size(), CV_16SC3);
                    dstM =  Mat::zeros(frame.size(), CV_16SC3);
                    cvtColor( frame, dstT, cv::COLOR_RGB2GRAY );
                    threshold(dstT,temp);
                    monphological(temp,dstM);
                    cv::imshow("Video", dstM);   
                    dstM.copyTo(dst);
                }
                else if(segmentBoolean)
                {
                    // grayscale, then call the threshold function and 
                    // call monphological and segment function and show
                    vector<pair<int,int>> regions;
                    cv::Mat temp=  Mat::zeros(frame.size(), CV_16SC3);
                    dstM =  Mat::zeros(frame.size(), CV_16SC3);
                    cvtColor( frame, dstT, cv::COLOR_RGB2GRAY );
                    threshold(dstT,temp);
                    monphological(temp,dstM);
                    cv::Mat labels, stats, centroids;
                    int connectivity = 8; // or 4
                    // Segment the regions
                    int label_count = cv::connectedComponentsWithStats(dstM, labels, stats, centroids, connectivity);
                    //cout<<label_count<<endl;
                    // Iterate through all labels and push area and regionID to vector
                    for (int i = 0; i < label_count; i++)
                    {
                        int area = stats.at<int>(i, cv::CC_STAT_AREA);
                        regions.push_back(make_pair(area,i));
                    }
                    // Sort the vector
                    sort(regions.rbegin(), regions.rend());
                    int regionSize=regions.size();
                    vector<int> regionsByOrder(regionSize,0);
                    // Insert top # regions to map
                    for (int index=0; index<(min(4,regionSize)); index++)
                    {
                        regionsByOrder.insert(regionsByOrder.begin()+index,regions[index].second);
                    }
                    // Set up the colors
                    vector<cv::Vec3b> colors(10);
                    colors[0] = cv::Vec3b(0,0,0);
                    colors[1] = cv::Vec3b(255,0,0);
                    colors[2] = cv::Vec3b(0,255,0);
                    colors[3] = cv::Vec3b(0,0,255);
                    colors[4] = cv::Vec3b(255,165,0);
                    colors[5] = cv::Vec3b(255,255,0);
                    colors[6] = cv::Vec3b(221,160,221);
                    cv::Mat out = cv::Mat::zeros(dstM.size(),CV_8UC3);
                    // Iterate through all pixels in the image
                    for(int i=0; i<dstM.rows;i++)
                    {
                       for(int j=0; j<dstM.cols;j++)
                       {
                        // If the region isn't 0 - background
                        if(labels.at<int>(cv::Point(j,i)) != 0)
                        {
                            // Extract the regionID
                            int colorIndex = (int)labels.at<int>(cv::Point(j,i));
                            int index=-1;
                            // Use the regionID to find its order
                            getIndex(regionsByOrder,colorIndex,index);
                            // Set the colors based on the order
                            if(index==1)
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[1];
                            else if(index==2)
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[2];
                            else if(index==3)
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[3];
                            else if(index==4)
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[4];
                            else if(index==5)
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[5];
                            else
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[6];
                        }
                       } 
                    }
                    // Show the image
                    cv::imshow("Video", out);                  
                    out.copyTo(dst);
                }
                else if(featureBoolean)
                {
                    // Initialize vectors and matrices
                    vector<pair<int,int>> regions;
                    cv::Mat temp=  Mat::zeros(frame.size(), CV_16SC3);
                    dstM =  Mat::zeros(frame.size(), CV_16SC3);
                    // grayscale, then call the threshold function and 
                    // call monphological and segment function and show
                    cvtColor( frame, dstT, cv::COLOR_RGB2GRAY );
                    threshold(dstT,temp);
                    monphological(temp,dstM);
                    vector<vector<int>> regionsMatrix(dstM.rows, vector<int> (dstM.cols, 0));
                    cv::Mat labels, stats, centroids;
                    int connectivity = 8; // or 4
                    // Segment the regions
                    int label_count = cv::connectedComponentsWithStats(dstM, labels, stats, centroids, connectivity);
                    // Iterate through all labels and push region ID and its corresponding area to regions
                    for (int i = 0; i < label_count; i++)
                    {
                        int area = stats.at<int>(i, cv::CC_STAT_AREA);
                        regions.push_back(make_pair(area,i));
                    }
                    // Sort the regions
                    sort(regions.rbegin(), regions.rend());
                    int regionSize=regions.size();
                    vector<int> regionsByOrder(5,0);
                    // Insert top # regions to map
                    for (int index=0; index<(min(4,regionSize)); index++)
                    {
                        regionsByOrder.insert(regionsByOrder.begin()+index,regions[index].second);
                    }
                    // Set up the colors
                    vector<cv::Vec3b> colors(10);
                    colors[0] = cv::Vec3b(0,0,0);
                    colors[1] = cv::Vec3b(255,0,0);
                    colors[2] = cv::Vec3b(0,255,0);
                    colors[3] = cv::Vec3b(0,0,255);
                    colors[4] = cv::Vec3b(255,165,0);
                    colors[5] = cv::Vec3b(255,255,0);
                    colors[6] = cv::Vec3b(221,160,221);
                    cv::Mat out = cv::Mat::zeros(dstM.size(),CV_8UC3);
                    // Iterate through all pixels in the image
                    for(int i=0; i<dstM.rows;i++)
                    {
                       for(int j=0; j<dstM.cols;j++)
                       {
                        // If the pixel value isn't 0
                        if(labels.at<int>(cv::Point(j,i)) != 0)
                        {
                            // Extract regionID 
                            int colorIndex = (int)labels.at<int>(cv::Point(j,i));
                            // Use the regionID to find its order
                            int index=-1;
                            getIndex(regionsByOrder,colorIndex,index);
                            // Set the color based on the order
                            if(index==1)
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[1];
                            else if(index==2)
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[2];
                            else if(index==3)
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[3];
                            else if(index==4)
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[4];
                            else if(index==5)
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[5];
                            else
                                out.at<cv::Vec3b>(cv::Point(j,i))=colors[6];
                        }
                       } 
                    }
                    int orderedRegionSize = regionsByOrder.size();
                    // Iterate through regions
                    for(int i =1;i<min(2,orderedRegionSize);i++)
                    {
                        double angle;
                        Moments m;
                        RotatedRect boundingRect;
                        float orientedCentralMoments;
                        // Call computeMomentsFeautures to retrieve region's moments and minimu area
                        computeMomentsFeautures(labels,regionsByOrder[i],m,boundingRect,orientedCentralMoments);
                        // Compute axis of least central moment
                        angle=atan2(2*m.mu11,m.mu20-m.mu02)*180/PI;
                        Point centerPoint;
                        // Centroid coordinates
                        double cx = centroids.at<double>(regionsByOrder[i], 0);
                        double cy = centroids.at<double>(regionsByOrder[i], 1);
                        // Compute the axis point coordinates
                        Point P1(cx+cos(angle)*100,cy+sin(angle)*100);
                        Point P2(cx-cos(angle)*100,cy-sin(angle)*100);
                        // Compute the rectangle point coordinates
                        Point2f vertices[4];
                        boundingRect.points(vertices);
                        // Draw the rectangle
                        for (int r = 0; r < 4; r++)
                            line(out, vertices[r], vertices[(r+1)%4], Scalar(255,255,255), 2);
                        // Draw the line, axis of least central moment
                        line(out,P1,P2,cv::Scalar(255,255,255));
                    }
                    // Show the image
                    cv::imshow("Video", out);
                    out.copyTo(dst);
                }
                else
                {
                    frame.copyTo(dst);
                    cv::imshow("Video", dst);
                }
                if(saveBoolean)
                {
                    // Save the frame into a file
                    imwrite("save_"+getTimestamp()+".jpg", dst); // A JPG FILE IS BEING SAVED
                    saveBoolean = false;
                }
        }
        delete capdev;

    }
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
    thresholdBoolean = false;
    monphologicalBoolean = false;
    featureBoolean = false;
    segmentBoolean=false;
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