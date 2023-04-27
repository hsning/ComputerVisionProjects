/*
Hao Sheng (Jack) Ning

CS 5330 Computer Vision
Spring 2023
Project 2: Content-based Image Retrieval 

CPP main.cpp file that stores a single function comprised of different parts corresponding to different tasks of the assignment
    1. Baseline Matching
    2. Histogram Matching
    3. Multi-histogram Matching
    4. Texture and Color
    5. Custom
    6. Extension

The main program is interacted through command line interfaces with the following format:
main.exe <target path> <image database directory path> <feature type> <matching method> <numberOfMatches>
For example:
Task 1: main.exe olympus/pic.1016.jpg olympus 9x9 Euclidean 4
Task 2: main.exe olympus/pic.0164.jpg olympus Histogram Histogram-Intersection 4
Task 3: main.exe olympus/pic.0274.jpg olympus Multi-Histogram Histogram-Intersection 4
Task 4: main.exe olympus/pic.0535.jpg olympus TextureAndColor Histogram-Intersection 4
Task 5: main.exe olympus/pic.0013.jpg olympus Custom Custom 10
Extension: main.exe olympus/pic.1082.jpg olympus Texture-Color-Entropy ScaledStandardDeviation 4

The functions return an int to indicate whether the function completed successfully or not
*/

// All libraries needed
#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <charconv>
#include "math.h"
#include "include/csv_util.h"
#include<bits/stdc++.h>
#include "include/filter.h"
#include <map>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    // Initialize variables
    // Target image path
    char targetname[256];
    // Feature type to evaluate
    char featureType[256];
    // Method to caluclate metric distances
    char matchingMethod[256];
    // Number of cloeset matches
    int numberOfMatches;
    // Char array to hold directory path
    char dirname[256];
    // Char array to hold characters for string 
    char buffer[256];
    // Variable to hold file
    FILE *fp;
    // Variable to hold directory 
    DIR *dirp;
    // Struct to hold directory path
    struct dirent *dp;
    

    // Check for sufficient arguments
    // If less than 6, throw error message, and terminate the program
    if( argc < 6) {
        printf("usage: %s <target path> <image database directory path> <feature type> <matching method> <numberOfMatches>\n", argv[0]);
        exit(-1);
    }

    // Get the feature type
    strcpy(featureType, argv[3]);
    // Get the matching method
    strcpy(matchingMethod, argv[4]);
    // Get number of matches
    numberOfMatches = atoi(argv[5]);
    // Output feature type
    printf("Feature type is: %s\n", featureType );
    // Sub-function for featureType = 9x9 and matchingMethod = Euclidean, task 1
    if((strcmp(featureType, "9x9") == 0) && (strcmp(matchingMethod, "Euclidean") == 0))
    {
        // Initialize vector variables for both target and database images
        vector<float> targetVectors;
        vector<float> databaseVectors;
        // Initialize vector variables to hold database image names and its distance value
        vector< pair <float, string> > vectOfImages;
        // Get the target path
        strcpy(targetname, argv[1]);
        printf("Processing target image %s\n", dirname );
        // Get the image based on the path
        Mat targetImage = imread(targetname);

        //If image does not exist, output an error message and terminate the program
        if (!targetImage.data) 
        {
            printf("Does not exist %s\n", targetname );
            return 1;
        }

        // Iterate through pixels in mid 9x9 area and add each color value to target vectors
        for(int i=floor(targetImage.rows/2)-4;i<ceil(targetImage.rows/2)+5;i++) 
        {
            // src pointer
            cv::Vec3b *rptr = targetImage.ptr<cv::Vec3b>(i);
            // for each column
            for(int j=floor(targetImage.cols/2)-4;j<ceil(targetImage.cols/2)+5;j++) 
            {
                // for each color channel
                for(int c=0;c<3;c++) 
                {
                targetVectors.push_back(rptr[j][c]);
                }
            }
        }

        // get the image database path
        strcpy(dirname, argv[2]);
        printf("Processing directory %s\n", dirname );

        // open the directory
        dirp = opendir( dirname );
        if( dirp == NULL) {
            printf("Cannot open directory %s\n", dirname);
            exit(-1);
        }

        // loop over all the files in the image file listing
        while( (dp = readdir(dirp)) != NULL ) 
        {
            // Initialie vector counter
            int vectorIndex=0;

            // Initialize distance value
            float distance=0;
            // check if the file is an image
            if( strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif") ) 
            {
                printf("processing image file: %s\n", dp->d_name);

                // build the overall filename
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);

                printf("full path name: %s\n", buffer);
                //append_image_data_csv('vectors.csv', dp->d_name, dp->d_namlen, 1 );
                Mat databaseImage = imread(buffer);

                //If image does not exist, output an error message and terminate the program
                if (!databaseImage.data) 
                {
                    printf("Does not exist %s\n", buffer);
                    return 1;
                }
                // Iterate through pixels in mid 9x9 area and add each color value to database vectors
                for(int i=floor(databaseImage.rows/2)-4;i<ceil(databaseImage.rows/2)+5;i++) 
                {
                    // src pointer
                    cv::Vec3b *rptr = databaseImage.ptr<cv::Vec3b>(i);
                    // for each column
                    for(int j=floor(databaseImage.cols/2)-4;j<ceil(databaseImage.cols/2)+5;j++) 
                    {
                        // for each color channel
                        for(int c=0;c<3;c++) 
                        {
                            // Use Euclidean (squared sum differences) to calculate the distance
                            distance += pow(rptr[j][c]-targetVectors[vectorIndex],2);
                            vectorIndex++;
                        }
                    }
                }
                printf("processing: %s\n", buffer);
                // Push the image name and distance to vector
                vectOfImages.push_back(make_pair(distance, buffer));  
            }
        }
        
        // Sort the vector in ascending order, ordered by the float object 
        sort(vectOfImages.begin(), vectOfImages.end());
        // Output first numberOfMatches in the vector
        cout<<endl;
        cout<<" TOP MATCHES"<<endl;
        cout<<"----------------------------------"<<endl;
        for (int index=0; index<numberOfMatches; index++)
        {
            // Out the image name and distance
            cout <<" "<<index<<": "<< vectOfImages[index].second << " "
                << vectOfImages[index].first << endl;
        }
        cout<<"----------------------------------"<<endl;
    } 
    else if((strcmp(featureType, "Histogram") == 0) && (strcmp(matchingMethod, "Histogram-Intersection") == 0))
    {
        // Initialize vector variables to hold database image names and its distance value
        vector< pair <float, string> > vectOfImages;
        // Initialize 8x8x8 3D histogram for target color
        vector<vector<vector<float>>> targetColorHistogram(8, vector<vector<float>>(8, vector<float>(8, 0)));
        // Get the target path
        strcpy(targetname, argv[1]);
        printf("Processing target image %s\n", targetname );
        // Get the image based on the path
        Mat targetImage = imread(targetname);

        //If image does not exist, output an error message and terminate the program
        if (!targetImage.data) 
        {
            printf("Does not exist %s\n", targetname );
            return 1;
        }

        // Calculate numberOfPixels the image contains, needed to normalize the histogram probability
        double numberOfPixels = targetImage.rows*targetImage.cols;
        // Iterate through all pixels and add each color value to its histogram bin 
        for(int i=0;i<targetImage.rows;i++) 
        {
            // src pointer
            cv::Vec3b *rptr = targetImage.ptr<cv::Vec3b>(i);
            // for each column
            for(int j=0;j<targetImage.cols;j++) 
            {
                int b = rptr[j][0]/32;
                int g = rptr[j][1]/32;
                int r = rptr[j][2]/32;
                //Increment corresponding histogram bin value by normalized 1
                targetColorHistogram[b][g][r]+=1.0/numberOfPixels;
            }
        }
        // Get the image database path
        strcpy(dirname, argv[2]);
        printf("Processing directory %s\n", dirname );

        // Open the directory
        dirp = opendir( dirname );
        if( dirp == NULL) {
            printf("Cannot open directory %s\n", dirname);
            exit(-1);
        }
         // Loop over all the files in the image file listing
        while( (dp = readdir(dirp)) != NULL ) 
        {
            // Initialize distance variable
            float sum=0.0;
            // Initialize 8x8x8 3D histogram for database image color
            vector<vector<vector<float>>> databaseColorHistogram(8, vector<vector<float>>(8, vector<float>(8, 0)));
            // Check if the file is an image
            if( strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif") ) 
            {
                printf("processing image file: %s\n", dp->d_name);

                // Build the overall filename
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);
                printf("full path name: %s\n", buffer);
                // Get the image based on path
                Mat databaseImage = imread(buffer);
                // Calculate number of pixels again, needed to normalize histogram value
                int numberOfPixels = databaseImage.rows*databaseImage.cols;

                //If image does not exist, output an error message and terminate the program
                if (!databaseImage.data) 
                {
                    printf("Does not exist %s\n", buffer);
                    return 1;
                }
                // Iterate through all pixels and add each color value to its histogram bin 
                for(int i=0;i<databaseImage.rows;i++) 
                {
                    // src pointer
                    cv::Vec3b *dptr = databaseImage.ptr<cv::Vec3b>(i);
                    // for each column
                    for(int j=0;j<databaseImage.cols;j++) 
                    {
                        int b = dptr[j][0]/32.0;
                        int g = dptr[j][1]/32.0;
                        int r = dptr[j][2]/32.0;
                        //Increment corresponding histogram bin value by normalized 1
                        databaseColorHistogram[b][g][r]+=1.0/numberOfPixels;
                    }
                }
                // Iterate through all bins, and add the min of target and database images
                for(int i=0;i<databaseColorHistogram.size();i++)
                {
                    for(int j=0;j<databaseColorHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseColorHistogram[i][j].size();c++)
                        {
                            sum+=min(databaseColorHistogram[i][j][c],targetColorHistogram[i][j][c]);
                        }
                    }
                }
                // Push the image name and distance to vector       
                vectOfImages.push_back(make_pair(1-sum, buffer));
                printf("processing: %s\n", buffer);
            }
        }

        // Sort the vector in ascending order, ordered by the float object 
        sort(vectOfImages.begin(), vectOfImages.end());
        // Output first numberOfMatches in the vector
        cout<<endl;
        cout<<" TOP MATCHES"<<endl;
        cout<<"----------------------------------"<<endl;
        for (int index=0; index<numberOfMatches; index++)
        {
            // Out the image name and distance
            cout <<" "<<index<<": "<< vectOfImages[index].second << " "
                << vectOfImages[index].first << endl;
        }
        cout<<"----------------------------------"<<endl;
    }
    else if((strcmp(featureType, "Multi-Histogram") == 0) && (strcmp(matchingMethod, "Histogram-Intersection") == 0))
    {
        // Initialize vector variables to hold database image names and its distance value
        vector< pair <float, string> > vectOfImages;
        // Initialize 8x8x8 3D histogram for target color top and bottom
        vector<vector<vector<float>>> targetColorHistogramTop(8, vector<vector<float>>(8, vector<float>(8, 0)));
        vector<vector<vector<float>>> targetColorHistogramBot(8, vector<vector<float>>(8, vector<float>(8, 0)));
        // get the target path
        strcpy(targetname, argv[1]);
        printf("Processing target image %s\n", targetname );
        // Get the image based on the path
        Mat targetImage = imread(targetname);

        //If image does not exist, output an error message and terminate the program
        if (!targetImage.data) 
        {
            printf("Does not exist %s\n", targetname );
            return 1;
        }
        // Calculate numberOfPixels the image contains, needed to normalize the histogram probability
        int numberOfPixels = targetImage.rows*targetImage.cols/2;
        // Target Top
        // Iterate through all pixels in the top half of the image
        // and add each color value to its histogram bin
        for(int i=0;i<targetImage.rows/2;i++) 
        {
            // src pointer
            cv::Vec3b *rptr = targetImage.ptr<cv::Vec3b>(i);
            // for each column
            for(int j=0;j<(targetImage.cols);j++) 
            {
                int b = rptr[j][0]/32;
                int g = rptr[j][1]/32;
                int r = rptr[j][2]/32;
                //Increment corresponding histogram bin value by normalized 1
                targetColorHistogramTop[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // Target Bot
        // Iterate through all pixels in the bottom half of the image
        // and add each color value to its histogram bin
        for(int i=targetImage.rows/2;i<targetImage.rows;i++) 
        {
            // src pointer
            cv::Vec3b *rptr = targetImage.ptr<cv::Vec3b>(i);
            // for each column
            for(int j=0;j<(targetImage.cols);j++) 
            {
                int b = rptr[j][0]/32;
                int g = rptr[j][1]/32;
                int r = rptr[j][2]/32;
                // Increment corresponding histogram bin value by normalized 1 
                targetColorHistogramBot[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // get the image database path
        strcpy(dirname, argv[2]);
        printf("Processing directory %s\n", dirname );

        // open the directory
        dirp = opendir( dirname );
        if( dirp == NULL) {
            printf("Cannot open directory %s\n", dirname);
            exit(-1);
        }
         // loop over all the files in the image file listing
        while( (dp = readdir(dirp)) != NULL ) 
        {
            // Initailize distance variable
            float sum=0.0;
            // Initialize 8x8x8 3D histogram for database color, top and bottom
            vector<vector<vector<float>>> databaseColorHistogramTop(8, vector<vector<float>>(8, vector<float>(8, 0)));
            vector<vector<vector<float>>> databaseColorHistogramBot(8, vector<vector<float>>(8, vector<float>(8, 0)));
            // check if the file is an image
            if( strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif") ) 
            {
                printf("processing image file: %s\n", dp->d_name);

                // build the overall filename
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);
                printf("full path name: %s\n", buffer);
                // Get the image based on the path
                Mat databaseImage = imread(buffer);
                // Calculate numberOfPixels the image contains, needed to normalize the histogram probability
                int numberOfPixels = databaseImage.rows*databaseImage.cols/2;

                //If image does not exist, output an error message and terminate the program
                if (!databaseImage.data) 
                {
                    printf("Does not exist %s\n", buffer);
                    return 1;
                }
                // Database Top
                // Iterate through all pixels in the top half of the image
                // and add each color value to its histogram bin
                for(int i=0;i<databaseImage.rows/2;i++) 
                {
                    // src pointer
                    cv::Vec3b *dptr = databaseImage.ptr<cv::Vec3b>(i);
                    // for each column
                    for(int j=0;j<databaseImage.cols;j++) 
                    {
                        int b = dptr[j][0]/32.0;
                        int g = dptr[j][1]/32.0;
                        int r = dptr[j][2]/32.0;
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseColorHistogramTop[b][g][r]+=1.0/numberOfPixels;
                    }
                }
                // Database Bot
                // Iterate through all pixels in the bottom half of the image
                // and add each color value to its histogram bin
                for(int i=databaseImage.rows/2;i<databaseImage.rows;i++) 
                {
                    // src pointer
                    cv::Vec3b *dptr = databaseImage.ptr<cv::Vec3b>(i);
                    // for each column
                    for(int j=0;j<databaseImage.cols;j++) 
                    {
                        int b = dptr[j][0]/32.0;
                        int g = dptr[j][1]/32.0;
                        int r = dptr[j][2]/32.0;
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseColorHistogramBot[b][g][r]+=1.0/numberOfPixels;
                    }
                }
                // Iterate through all bins, and add the min of target and database images
                for(int i=0;i<databaseColorHistogramTop.size();i++)
                {
                    for(int j=0;j<databaseColorHistogramTop[i].size();j++)
                    {
                        for(int c=0;c<databaseColorHistogramTop[i][j].size();c++)
                        {
                            sum+=min(databaseColorHistogramTop[i][j][c],targetColorHistogramTop[i][j][c]);
                            sum+=min(databaseColorHistogramBot[i][j][c],targetColorHistogramBot[i][j][c]);
                            //cout<<"sum:"<<sum<<endl;
                        }
                    }
                }
                // Push the image name and distance to vector
                vectOfImages.push_back(make_pair(2-sum, buffer));
                printf("processing: %s\n", buffer); 
            }
        }
        
        // Sort the vector in ascending order, ordered by the float object 
        sort(vectOfImages.begin(), vectOfImages.end());
        // Output first numberOfMatches in the vector
        cout<<endl;
        cout<<" TOP MATCHES"<<endl;
        cout<<"----------------------------------"<<endl;
        for (int index=0; index<numberOfMatches; index++)
        {
            // Out the image name and distance
            cout <<" "<<index<<": "<< vectOfImages[index].second << " "
                << vectOfImages[index].first << endl;
        }
        cout<<"----------------------------------"<<endl;
    }
    else if((strcmp(featureType, "TextureAndColor") == 0) && (strcmp(matchingMethod, "Histogram-Intersection") == 0))
    {
        // Initialize vector variables to hold database image names and its distance value
        vector< pair <float, string> > vectOfImages;
        // Initialize 8x8x8 3D histogram for target color and texture
        vector<vector<vector<float>>> targetColorHistogram(8, vector<vector<float>>(8, vector<float>(8, 0)));
        vector<vector<vector<float>>> targetTextureHistogram(8, vector<vector<float>>(8, vector<float>(8, 0)));
        // Initialize 25 1D histogram for target orientation
        vector<float> targetOrientationHistogram(25, 0);
        
        // get the target path
        strcpy(targetname, argv[1]);
        printf("Processing target image %s\n", targetname );
        // Get the image based on the path
        Mat targetImage = imread(targetname);

        //If image does not exist, output an error message and terminate the program
        if (!targetImage.data) 
        {
            printf("Does not exist %s\n", targetname );
            return 1;
        }
        // Calculate numberOfPixels the image contains, needed to normalize the histogram probability
        int numberOfPixels = targetImage.rows*targetImage.cols;
        // Iterate through all pixels and add each color value to its histogram bin
        for(int i=0;i<targetImage.rows;i++) 
        {
            // src pointer
            cv::Vec3b *rptr = targetImage.ptr<cv::Vec3b>(i);
            // for each column
            for(int j=0;j<targetImage.cols;j++) 
            {
                int b = rptr[j][0]/32;
                int g = rptr[j][1]/32;
                int r = rptr[j][2]/32;
                // Increment corresponding histogram bin value by normalized 1 
                targetColorHistogram[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // Call function to generate texture and orientation histogram
        generateMagnitudeHistogram(targetImage, targetTextureHistogram);
        generateOrientationHistogram( targetImage, targetOrientationHistogram);
        waitKey(1);
        // get the image database path
        strcpy(dirname, argv[2]);
        printf("Processing directory %s\n", dirname );
        // open the directory
        dirp = opendir( dirname );
        if( dirp == NULL) {
            printf("Cannot open directory %s\n", dirname);
            exit(-1);
        }
         // loop over all the files in the image file listing
        while( (dp = readdir(dirp)) != NULL ) 
        { 
            float sum=0.0;
            
            // Initialize 8x8x8 3D histogram for database color and texture
            vector<vector<vector<float>>> databaseColorHistogram(8, vector<vector<float>>(8, vector<float>(8, 0)));
            vector<vector<vector<float>>> databaseTextureHistogram(8, vector<vector<float>>(8, vector<float>(8, 0)));      
            // Initialize 1x25 1D histogram for database orientation
            vector<float> databaseOrientationHistogram(25, 0);
            // check if the file is an image
            
            if( strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif") ) 
            {
                printf("processing image file: %s\n", dp->d_name);

                // build the overall filename
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);
                printf("full path name: %s\n", buffer);
                // Get the image based on the path
                Mat databaseImage = imread(buffer);
                // Calculate numberOfPixels the image contains, needed to normalize the histogram probability
                int numberOfPixels = databaseImage.rows*databaseImage.cols;
                //If image does not exist, output an error message and terminate the program
                if (!databaseImage.data) 
                {
                    printf("Does not exist %s\n", buffer);
                    return 1;
                }
                // Iterate through all pixels and add each color value to its histogram bin
                for(int i=0;i<databaseImage.rows;i++) 
                {
                    // src pointer
                    cv::Vec3b *dptr = databaseImage.ptr<cv::Vec3b>(i);
                    // for each column
                    for(int j=0;j<databaseImage.cols;j++) 
                    {
                        int b = dptr[j][0]/32.0;
                        int g = dptr[j][1]/32.0;
                        int r = dptr[j][2]/32.0;
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseColorHistogram[b][g][r]+=(1.0/numberOfPixels);
                    }
                }
                // Iterate through all color bins, and add the min of target and database images
                for(int i=0;i<databaseColorHistogram.size();i++)
                {
                    for(int j=0;j<databaseColorHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseColorHistogram[i][j].size();c++)
                        {
                            sum+=min(targetColorHistogram[i][j][c],databaseColorHistogram[i][j][c]);              
                        }
                    }
                }
                // Call utility functions to generate gradient magnitude mat
                cv::Mat databaseX;
                cv::Mat databaseY;
                cv::Mat databaseM;
                sobelX3x3(databaseImage,databaseX);
                sobelY3x3(databaseImage,databaseY);
                magnitude(databaseX,databaseY,databaseM);
                float mSum=0;
                // Iterate through all pixels and add each texture value to its histogram bin
                for(int i=0;i<databaseM.rows;i++)
                {
                    cv::Vec3b *drptr = databaseM.ptr<cv::Vec3b>(i);
                    for(int j=0;j<databaseM.cols;j++)
                    {
                        int b = drptr[j][0]/32.0;
                        int g = drptr[j][1]/32.0;
                        int r = drptr[j][2]/32.0;
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseTextureHistogram[b][g][r]+=(1.0/numberOfPixels);
                    }
                }
                // Iterate through all texture bins, and add the min of target and database images
                for(int i=0;i<databaseTextureHistogram.size();i++)
                {
                    for(int j=0;j<databaseTextureHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseTextureHistogram[i][j].size();c++)
                        {
                            mSum+=min(targetTextureHistogram[i][j][c],databaseTextureHistogram[i][j][c]);              
                        }
                    }
                }
                // Call function to generate orientation histogram
                generateOrientationHistogram( databaseImage, databaseOrientationHistogram);
                // Initialize distance variable
                float oSum = 0;
                // Iterate through all bins, and add the min of target and database images
                for(int i=0;i<databaseOrientationHistogram.size();i++)
                {
                    oSum+=min(targetOrientationHistogram[i],databaseOrientationHistogram[i]);              
                }
                // Push the image name and distance to vector
                vectOfImages.push_back(make_pair(((1-mSum)*25+(1-sum)*50+(1-oSum)*25), buffer));
                printf("processing: %s\n", buffer);
                waitKey(1);
                //append_image_data_csv("example.csv",buffer,databaseVectors,0);
                //databaseVectors.clear();   
            }
        }
        
        // Sort the vector in ascending order, ordered by the float object 
        sort(vectOfImages.begin(), vectOfImages.end());
        // Output first numberOfMatches in the vector
        cout<<endl;
        cout<<" TOP MATCHES"<<endl;
        cout<<"----------------------------------"<<endl;
        for (int index=0; index<numberOfMatches; index++)
        {
            // Out the image name and distance
            cout <<" "<<index<<": "<< vectOfImages[index].second << " "
                << vectOfImages[index].first << endl;
        }
        cout<<"----------------------------------"<<endl;
        waitKey(1);
    }
    else if((strcmp(featureType, "Custom") == 0) && (strcmp(matchingMethod, "Custom") == 0))
    {
        // Initialize variable to denote number of bins
        int numberOfBins = 32;
        // Initialize vector variables to hold database image names and its distance value
        vector< pair <float, string> > vectOfImages;
        //append_image_data_csv("example.csv","flower.jpg",targetVectors,1);
        // Initialize 32x32x32 3D histogram for database color & texture, whole image and center only
        vector<vector<vector<float>>> targetColorHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
        vector<vector<vector<float>>> targetTextureHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
        vector<vector<vector<float>>> targetColorCenterHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
        vector<vector<vector<float>>> targetTextureCenterHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
        // get the target path
        strcpy(targetname, argv[1]);
        printf("Processing target image %s\n", targetname );
        // Get the image based on the path
        Mat targetImage = imread(targetname);

        //If image does not exist, output an error message and terminate the program
        if (!targetImage.data) 
        {
            printf("Does not exist %s\n", targetname );
            return 1;
        }
        // Calculate numberOfPixels the image contains, needed to normalize the histogram probability
        int numberOfPixels = targetImage.rows*targetImage.cols;
        // Iterate through all pixels and add each color value to its histogram bin
        for(int i=0;i<targetImage.rows;i++) 
        {
            // src pointer
            cv::Vec3b *rptr = targetImage.ptr<cv::Vec3b>(i);
            // for each column
            for(int j=0;j<targetImage.cols;j++) 
            {
                int b = rptr[j][0]/(256/numberOfBins);
                int g = rptr[j][1]/(256/numberOfBins);
                int r = rptr[j][2]/(256/numberOfBins);
                // Increment corresponding histogram bin value by normalized 1 
                targetColorHistogram[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // Call utility functions to generate gradient magnitude mat
        cv::Mat targetX;
        cv::Mat targetY;
        cv::Mat targetM;
        sobelX3x3(targetImage,targetX);
        sobelY3x3(targetImage,targetY);
        magnitude(targetX,targetY,targetM);
        // Iterate through all pixels and add each gradient magnitude value to its histogram bin
        for(int i=0;i<targetM.rows;i++)
        {
            cv::Vec3b *trptr = targetM.ptr<cv::Vec3b>(i);
            for(int j=0;j<targetM.cols;j++)
            {
                int b = trptr[j][0]/(256/numberOfBins);
                int g = trptr[j][1]/(256/numberOfBins);
                int r = trptr[j][2]/(256/numberOfBins);
                // Increment corresponding histogram bin value by normalized 1 
                targetTextureHistogram[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // Calculate numberOfPixels center of the image contains, needed to normalize the histogram probability
        numberOfPixels = (targetImage.rows/2)*(targetImage.cols/2);
        // Iterate through all pixels in the center of the image
        // and add each color value to its histogram bin
        for(int i=(targetImage.rows/4);i<(targetImage.rows/4*3);i++) 
        {
            // src pointer
            cv::Vec3b *rptr = targetImage.ptr<cv::Vec3b>(i);
            // for each column
            for(int j=targetImage.cols/4;j<(targetImage.cols/4*3);j++) 
            {
                int b = rptr[j][0]/(256/numberOfBins);
                int g = rptr[j][1]/(256/numberOfBins);
                int r = rptr[j][2]/(256/numberOfBins);
                // Increment corresponding histogram bin value by normalized 1 
                targetColorCenterHistogram[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // Iterate through all pixels in the center of the image
        // and add each graident magnitude value to its histogram bin
        for(int i=targetM.rows/4;i<(targetM.rows/4*3);i++)
        {
            cv::Vec3b *trptr = targetM.ptr<cv::Vec3b>(i);
            for(int j=targetM.cols/4;j<(targetM.cols/4*3);j++)
            {
                int b = trptr[j][0]/(256/numberOfBins);
                int g = trptr[j][1]/(256/numberOfBins);
                int r = trptr[j][2]/(256/numberOfBins);
                // Increment corresponding histogram bin value by normalized 1 
                targetTextureCenterHistogram[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // get the image database path
        strcpy(dirname, argv[2]);
        printf("Processing directory %s\n", dirname );

        // open the directory
        dirp = opendir( dirname );
        if( dirp == NULL) {
            printf("Cannot open directory %s\n", dirname);
            exit(-1);
        }
         // loop over all the files in the image file listing
        while( (dp = readdir(dirp)) != NULL ) 
        {
            // Initialize distance variables
            float sum = 0.0;
            float sumCenter = 0.0;
            // Initialize numberOfBinsxnumberOfBinsxnumberOfBins 3D histogram for database color & texture, whole and center only
            vector<vector<vector<float>>> databaseColorHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
            vector<vector<vector<float>>> databaseColorCenterHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
            vector<vector<vector<float>>> databaseTextureHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
            vector<vector<vector<float>>> databaseTextureCenterHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
            // check if the file is an image
            if( strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif") ) 
            {
                printf("processing image file: %s\n", dp->d_name);

                // build the overall filename
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);
                printf("full path name: %s\n", buffer);
                // Get the image based on the path
                Mat databaseImage = imread(buffer);
                // Calculate numberOfPixels the image contains, needed to normalize the histogram probability
                int numberOfPixels = databaseImage.rows*databaseImage.cols;

                //If image does not exist, output an error message and terminate the program
                if (!databaseImage.data) 
                {
                    printf("Does not exist %s\n", buffer);
                    return 1;
                }
                // Iterate through all pixels and add each color value to its histogram bin
                for(int i=0;i<databaseImage.rows;i++) 
                {
                    // src pointer
                    cv::Vec3b *dptr = databaseImage.ptr<cv::Vec3b>(i);
                    // for each column
                    for(int j=0;j<databaseImage.cols;j++) 
                    {
                        int b = dptr[j][0]/(256/numberOfBins);
                        int g = dptr[j][1]/(256/numberOfBins);
                        int r = dptr[j][2]/(256/numberOfBins);
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseColorHistogram[b][g][r]+=(1.0/numberOfPixels);
                    }
                }
                // Iterate through all bins, and add the min of target and database images
                for(int i=0;i<databaseColorHistogram.size();i++)
                {
                    for(int j=0;j<databaseColorHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseColorHistogram[i][j].size();c++)
                        {
                            sum+=min(targetColorHistogram[i][j][c],databaseColorHistogram[i][j][c]);              
                        }
                    }
                }
                // Call utility functions to generate gradient magnitude for database images
                cv::Mat databaseX;
                cv::Mat databaseY;
                cv::Mat databaseM;
                sobelX3x3(databaseImage,databaseX);
                sobelY3x3(databaseImage,databaseY);
                magnitude(databaseX,databaseY,databaseM);
                // Initialize distance variable
                float mSum=0;
                // Iterate through all pixels and add each gradient magnitude value to its histogram bin
                for(int i=0;i<targetM.rows;i++)
                {
                    cv::Vec3b *trptr = targetM.ptr<cv::Vec3b>(i);
                    cv::Vec3b *drptr = databaseM.ptr<cv::Vec3b>(i);
                    for(int j=0;j<targetM.cols;j++)
                    {
                        int b = drptr[j][0]/(256/numberOfBins);
                        int g = drptr[j][1]/(256/numberOfBins);
                        int r = drptr[j][2]/(256/numberOfBins);
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseTextureHistogram[b][g][r]+=(1.0/numberOfPixels);
                    }
                }
                // Iterate through all bins, and add the min of target and database images
                for(int i=0;i<databaseTextureHistogram.size();i++)
                {
                    for(int j=0;j<databaseTextureHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseTextureHistogram[i][j].size();c++)
                        {
                            mSum+=min(targetTextureHistogram[i][j][c],databaseTextureHistogram[i][j][c]);              
                        }
                    }
                }
                // Calculate numberOfPixels center of the image contains, needed to normalize the histogram probability
                numberOfPixels = (databaseImage.rows/2) * (databaseImage.cols/2);
                // Iterate through all pixels in the center of the image
                // and add each color value to its histogram bin
                for(int i=databaseImage.rows/4;i<(databaseImage.rows/4*3);i++) 
                {
                    // src pointer
                    cv::Vec3b *dptr = databaseImage.ptr<cv::Vec3b>(i);
                    // for each column
                    for(int j=databaseImage.cols/4;j<(databaseImage.cols/4*3);j++) 
                    {
                        int b = dptr[j][0]/(256/numberOfBins);
                        int g = dptr[j][1]/(256/numberOfBins);
                        int r = dptr[j][2]/(256/numberOfBins);
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseColorCenterHistogram[b][g][r]+=(1.0/numberOfPixels);
                    }
                }
                // Iterate through all bins, and add the min of target and database images
                for(int i=0;i<databaseColorCenterHistogram.size();i++)
                {
                    for(int j=0;j<databaseColorCenterHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseColorCenterHistogram[i][j].size();c++)
                        {
                            sumCenter+=min(targetColorCenterHistogram[i][j][c],databaseColorCenterHistogram[i][j][c]);              
                        }
                    }
                }
                // Initialize distance variable
                float mSumCenter=0;
                // Iterate through all pixels in the center of the image 
                // and add each gradient magnitude value to its histogram 
                for(int i=targetM.rows/4;i<(targetM.rows/4*3);i++)
                {
                    cv::Vec3b *trptr = targetM.ptr<cv::Vec3b>(i);
                    cv::Vec3b *drptr = databaseM.ptr<cv::Vec3b>(i);
                    for(int j=targetM.cols/4;j<(targetM.cols/4*3);j++)
                    {
                        int b = drptr[j][0]/(256/numberOfBins);
                        int g = drptr[j][1]/(256/numberOfBins);
                        int r = drptr[j][2]/(256/numberOfBins);
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseTextureCenterHistogram[b][g][r]+=(1.0/numberOfPixels);
                    }
                }
                // Iterate through all bins, and add the min of target and database images
                for(int i=0;i<databaseTextureHistogram.size();i++)
                {
                    for(int j=0;j<databaseTextureHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseTextureHistogram[i][j].size();c++)
                        {
                            mSumCenter+=min(targetTextureCenterHistogram[i][j][c],databaseTextureCenterHistogram[i][j][c]);              
                        }
                    }
                }
                //vectOfImages.push_back(make_pair((mSum*50+sum/maxColor*50), buffer));
                double numberOfBaselinePixels = 100*100*3*255*255.0;
                // Push the image name and distance to vector
                vectOfImages.push_back(make_pair((1-sum)*20+(1-sumCenter)*30+(1-mSum)*20+(1-mSumCenter)*30, buffer));
                printf("processing: %s\n", buffer);
                waitKey(1);
            }
        }
        
        // Sort the vector in ascending order, ordered by the float object 
        sort(vectOfImages.begin(), vectOfImages.end());
        // Output first numberOfMatches in the vector
        cout<<endl;
        cout<<" TOP MATCHES"<<endl;
        cout<<"----------------------------------"<<endl;
        for (int index=0; index<numberOfMatches; index++)
        {
            // Out the image name and distance
            cout <<" "<<index<<": "<< vectOfImages[index].second << " "
                << vectOfImages[index].first << endl;
        }
        cout<<"----------------------------------"<<endl;
        waitKey(1);
    }
    else if((strcmp(featureType, "Texture-Color-Entropy") == 0) && (strcmp(matchingMethod, "ScaledStandardDeviation") == 0))
    {
        // Initialize target entropy variable
        double targetColorEntropy = 0.0;
        // Initialize variable to denote number of bin in a histogram
        int numberOfBins = 32;
        // Initialize vector variables to hold database image names and its distance value
        vector< pair <float, string> > vectOfImages;
        // Initialize map for all the vector distances, <nameOfTheImage, distance>
        map<string, float> mapOfColorSum;
        map<string, float> mapOfColorSumCenter;
        map<string, float> mapOfGradientSum;
        map<string, float> mapOfGradientSumCenter;
        map<string, float> mapOfEntropy;
        // Initialize a vector to hold all the names of the database images
        vector<string> pics;
        // Initialize numberOfBinsxnumberOfBinsxnumberOfBins 3D histogram for target color and texture, whole and center only
        vector<vector<vector<float>>> targetColorHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
        vector<vector<vector<float>>> targetTextureHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
        vector<vector<vector<float>>> targetColorCenterHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
        vector<vector<vector<float>>> targetTextureCenterHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
        // get the target path
        strcpy(targetname, argv[1]);
        printf("Processing target image %s\n", targetname );
        // Get the image based on the path
        Mat targetImage = imread(targetname);

        //If image does not exist, output an error message and terminate the program
        if (!targetImage.data) 
        {
            printf("Does not exist %s\n", targetname );
            return 1;
        }
        // Calculate numberOfPixels the image contains, needed to normalize the histogram probability
        int numberOfPixels = targetImage.rows*targetImage.cols;
        // Iterate through all pixels and add each color value to its histogram bin
        for(int i=0;i<targetImage.rows;i++) 
        {
            // src pointer
            cv::Vec3b *rptr = targetImage.ptr<cv::Vec3b>(i);
            // for each column
            for(int j=0;j<targetImage.cols;j++) 
            {
                int b = rptr[j][0]/(256/numberOfBins);
                int g = rptr[j][1]/(256/numberOfBins);
                int r = rptr[j][2]/(256/numberOfBins);
                // Increment corresponding histogram bin value by normalized 1 
                targetColorHistogram[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // Call utility functions to generate gradient magnitude mat
        cv::Mat targetX;
        cv::Mat targetY;
        cv::Mat targetM;
        sobelX3x3(targetImage,targetX);
        sobelY3x3(targetImage,targetY);
        magnitude(targetX,targetY,targetM);
        // Iterate through all pixels and add each gradient magnitude value to its histogram bin
        for(int i=0;i<targetM.rows;i++)
        {
            cv::Vec3b *trptr = targetM.ptr<cv::Vec3b>(i);
            for(int j=0;j<targetM.cols;j++)
            {
                int b = trptr[j][0]/(256/numberOfBins);
                int g = trptr[j][1]/(256/numberOfBins);
                int r = trptr[j][2]/(256/numberOfBins);
                // Increment corresponding histogram bin value by normalized 1 
                targetTextureHistogram[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // Calculate numberOfPixels center of the image contains, needed to normalize the histogram probability
        numberOfPixels = (targetImage.rows/2)*(targetImage.cols/2);
        // Iterate through all pixels in the center of the image, 
        // and add each color value to its histogram bin
        for(int i=(targetImage.rows/4);i<(targetImage.rows/4*3);i++) 
        {
            // src pointer
            cv::Vec3b *rptr = targetImage.ptr<cv::Vec3b>(i);
            // for each column
            for(int j=targetImage.cols/4;j<(targetImage.cols/4*3);j++) 
            {
                int b = rptr[j][0]/(256/numberOfBins);
                int g = rptr[j][1]/(256/numberOfBins);
                int r = rptr[j][2]/(256/numberOfBins);
                // Increment corresponding histogram bin value by normalized 1
                targetColorCenterHistogram[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // Iterate through all pixels in the center of the image, 
        // and add each graident magnitude value to its histogram bin
        for(int i=targetM.rows/4;i<(targetM.rows/4*3);i++)
        {
            cv::Vec3b *trptr = targetM.ptr<cv::Vec3b>(i);
            for(int j=targetM.cols/4;j<(targetM.cols/4*3);j++)
            {
                int b = trptr[j][0]/(256/numberOfBins);
                int g = trptr[j][1]/(256/numberOfBins);
                int r = trptr[j][2]/(256/numberOfBins);
                // Increment corresponding histogram bin value by normalized 1
                targetTextureCenterHistogram[b][g][r]+=(1.0/numberOfPixels);
            }
        }
        // get the image database path
        strcpy(dirname, argv[2]);
        printf("Processing directory %s\n", dirname );

        // open the directory
        dirp = opendir( dirname );
        if( dirp == NULL) {
            printf("Cannot open directory %s\n", dirname);
            exit(-1);
        }
         // loop over all the files in the image file listing
        while( (dp = readdir(dirp)) != NULL ) 
        {
            // Set target color entropy value to 0 again
            targetColorEntropy = 0.0;
            // Initialize variables for distances
            float sum = 0.0;
            float sumCenter = 0.0;
            // Initialize numberOfBinsxnumberOfBinsxnumberOfBins 3D histogram for database color & texture, whole and center only
            vector<vector<vector<float>>> databaseColorHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
            vector<vector<vector<float>>> databaseColorCenterHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
            vector<vector<vector<float>>> databaseTextureHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
            vector<vector<vector<float>>> databaseTextureCenterHistogram(numberOfBins, vector<vector<float>>(numberOfBins, vector<float>(numberOfBins, 0)));
            // check if the file is an image
            if( strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif") ) 
            {
                printf("processing image file: %s\n", dp->d_name);

                // build the overall filename
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);
                printf("full path name: %s\n", buffer);
                // Get the image based on the path
                Mat databaseImage = imread(buffer);
                // Calculate numberOfPixels the image contains, needed to normalize the histogram probability
                int numberOfPixels = databaseImage.rows*databaseImage.cols;

                //If image does not exist, output an error message and terminate the program
                if (!databaseImage.data) 
                {
                    printf("Does not exist %s\n", buffer);
                    return 1;
                }
                // Initialize variable for database color entropy
                double databaseColorEntropy = 0.0;
                // Iterate through all pixels and add each color value to its histogram bin
                for(int i=0;i<databaseImage.rows;i++) 
                {
                    // src pointer
                    cv::Vec3b *dptr = databaseImage.ptr<cv::Vec3b>(i);
                    // for each column
                    for(int j=0;j<databaseImage.cols;j++) 
                    {
                        int b = dptr[j][0]/(256/numberOfBins);
                        int g = dptr[j][1]/(256/numberOfBins);
                        int r = dptr[j][2]/(256/numberOfBins);
                        databaseColorHistogram[b][g][r]+=(1.0/numberOfPixels);
                    }
                }
                // Iterate through all bins, and add the min of target and database images
                // and use entropy formula to calculate entropy sun for each image, target and database
                for(int i=0;i<databaseColorHistogram.size();i++)
                {
                    for(int j=0;j<databaseColorHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseColorHistogram[i][j].size();c++)
                        {
                            sum+=min(targetColorHistogram[i][j][c],databaseColorHistogram[i][j][c]);    
                            float p1 = targetColorHistogram[i][j][c];
                            float p2 = databaseColorHistogram[i][j][c];
                            // Only calculate when value is greater than 0,
                            // else it will throw an error
                            if(p1>0)
                                targetColorEntropy+=(p1*log10(p1)); 
                            if(p2>0)
                                databaseColorEntropy+=(p2*log10(p2));        
                        }
                    }
                }
                targetColorEntropy *= -1;
                databaseColorEntropy *= -1;
                // Insert the color distance and entropy value to its map
                mapOfColorSum.insert({buffer,1-sum});
                mapOfEntropy.insert({buffer,databaseColorEntropy});
                // Call the utility functions to generate gradient magnitude mat
                cv::Mat databaseX;
                cv::Mat databaseY;
                cv::Mat databaseM;
                sobelX3x3(databaseImage,databaseX);
                sobelY3x3(databaseImage,databaseY);
                magnitude(databaseX,databaseY,databaseM);
                // Initialize a variable for distance
                float mSum=0;
                // Iterate through all pixels and add each gradient magnitude value to its histogram bin
                for(int i=0;i<targetM.rows;i++)
                {
                    cv::Vec3b *trptr = targetM.ptr<cv::Vec3b>(i);
                    cv::Vec3b *drptr = databaseM.ptr<cv::Vec3b>(i);
                    for(int j=0;j<targetM.cols;j++)
                    {
                        int b = drptr[j][0]/(256/numberOfBins);
                        int g = drptr[j][1]/(256/numberOfBins);
                        int r = drptr[j][2]/(256/numberOfBins);
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseTextureHistogram[b][g][r]+=(1.0/numberOfPixels);
                    }
                }
                // Iterate through all bins, and add the min of target and database images
                for(int i=0;i<databaseTextureHistogram.size();i++)
                {
                    for(int j=0;j<databaseTextureHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseTextureHistogram[i][j].size();c++)
                        {
                            mSum+=min(targetTextureHistogram[i][j][c],databaseTextureHistogram[i][j][c]);              
                        }
                    }
                }
                // Insert the gradient magnitude distance to its map
                mapOfGradientSum.insert({buffer,1-mSum});
                // Calculate numberOfPixels the image contains, needed to normalize the histogram probability
                numberOfPixels = (databaseImage.rows/2) * (databaseImage.cols/2);
                // Iterate through all pixels in the center of the image
                //and add each color value to its histogram bin
                for(int i=databaseImage.rows/4;i<(databaseImage.rows/4*3);i++) 
                {
                    // src pointer
                    cv::Vec3b *dptr = databaseImage.ptr<cv::Vec3b>(i);
                    // for each column
                    for(int j=databaseImage.cols/4;j<(databaseImage.cols/4*3);j++) 
                    {
                        int b = dptr[j][0]/(256/numberOfBins);
                        int g = dptr[j][1]/(256/numberOfBins);
                        int r = dptr[j][2]/(256/numberOfBins);
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseColorCenterHistogram[b][g][r]+=(1.0/numberOfPixels);
                    }
                }
                // Iterate through all bins, and add the min of target and database images
                for(int i=0;i<databaseColorCenterHistogram.size();i++)
                {
                    for(int j=0;j<databaseColorCenterHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseColorCenterHistogram[i][j].size();c++)
                        {
                            sumCenter+=min(targetColorCenterHistogram[i][j][c],databaseColorCenterHistogram[i][j][c]);              
                        }
                    }
                }
                // Insert the gradient magnitude distance for the center of the image to its map
                mapOfColorSumCenter.insert({buffer,1-sumCenter});
                // Initialize a variable for distance
                float mSumCenter=0;
                // Iterate through all pixels in the center of the image
                // and add each graident magnitude value to its histogram bin
                for(int i=targetM.rows/4;i<(targetM.rows/4*3);i++)
                {
                    cv::Vec3b *trptr = targetM.ptr<cv::Vec3b>(i);
                    cv::Vec3b *drptr = databaseM.ptr<cv::Vec3b>(i);
                    for(int j=targetM.cols/4;j<(targetM.cols/4*3);j++)
                    {
                        int b = drptr[j][0]/(256/numberOfBins);
                        int g = drptr[j][1]/(256/numberOfBins);
                        int r = drptr[j][2]/(256/numberOfBins);
                        // Increment corresponding histogram bin value by normalized 1 
                        databaseTextureCenterHistogram[b][g][r]+=(1.0/numberOfPixels);
                    }
                }
                // Iterate through all bins, and add the min of target and database images
                for(int i=0;i<databaseTextureHistogram.size();i++)
                {
                    for(int j=0;j<databaseTextureHistogram[i].size();j++)
                    {
                        for(int c=0;c<databaseTextureHistogram[i][j].size();c++)
                        {
                            mSumCenter+=min(targetTextureCenterHistogram[i][j][c],databaseTextureCenterHistogram[i][j][c]);              
                        }
                    }
                }
                // Insert the color distance for the center of the image to its map
                mapOfGradientSumCenter.insert({buffer,1-mSumCenter});
                // Insert each database image's name to the array
                pics.push_back(buffer);   
            }
        }
        // Initialize standard deviation variables for all the feature types
        double sdGradient=0.0;
        double sdGraidentCenter=0.0;
        double sdSum=0.0;
        double sdSumCenter=0.0;
        double sdEntropy=0.0;
        // Call utility function to generate standard deviations for all the feature types
        calculateStandardDeviation(mapOfGradientSum,sdGradient);
        calculateStandardDeviation(mapOfGradientSumCenter,sdGraidentCenter);
        calculateStandardDeviation(mapOfColorSum,sdSum);
        calculateStandardDeviation(mapOfColorSumCenter,sdSumCenter);
        calculateStandardDeviation(mapOfEntropy,sdEntropy);
        // For each database image, calculate its number of standard deviations away from the target
        // and all it all up with a weight and push the image name and distance to vector
        for (int i=0;i<pics.size();i++) 
        {
            double numOfSDGradientSum = mapOfGradientSum[pics[i]]/sdGradient;
            double numOfSDGraidentSumCenter = mapOfGradientSumCenter[pics[i]]/sdGraidentCenter;
            double numOfSDColorSum = mapOfColorSum[pics[i]]/sdSum;
            double numOfSDColorSumCenter = mapOfColorSumCenter[pics[i]]/sdSumCenter;
            double numOfSDEntropy = abs((mapOfEntropy[pics[i]]-targetColorEntropy))/sdEntropy;
            vectOfImages.push_back(make_pair(numOfSDColorSum*30+numOfSDColorSumCenter*15+numOfSDGradientSum*30+numOfSDGraidentSumCenter*15+numOfSDEntropy*10, pics[i]));
        }
        
        printf("processing: %s\n", buffer);
        waitKey(1);

        // Sort the vector in ascending order, ordered by the float object 
        sort(vectOfImages.begin(), vectOfImages.end());
        // Output first numberOfMatches in the vector
        cout<<endl;
        cout<<" TOP MATCHES"<<endl;
        cout<<"----------------------------------"<<endl;
        for (int index=0; index<numberOfMatches; index++)
        {
            // Out the image name and distance
            cout <<" "<<index<<": "<< vectOfImages[index].second << " "
                << vectOfImages[index].first << endl;
        }
        cout<<"----------------------------------"<<endl;
        waitKey(1);
    }
    
    printf("Terminating\n");
    return(0);
    
}
