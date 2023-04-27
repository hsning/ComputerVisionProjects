#include <iostream>

#include <opencv2/opencv.hpp>
#include "include/filter.h"
#include <chrono>
#include <fstream>
#include <string>
#include <ctime>
using namespace cv;
using namespace std;

//Initialize all the boolean variables determining state of the program
bool grayscale = false;
bool alternativeGrayScale = false;
bool blurFiveByFive = false;
bool sobelX = false;
bool sobelY = false;
bool magnitudeGradient = false;
bool blurQuantizeBoolean = false;
bool cartoonBoolean = false;
bool saveBoolean = false;
bool medianFilterBoolean = false;
bool addCaptionBoolean = false;
std::string x; 
std::string getTimestamp();
void reset();
int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

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
                cv::Mat dst;
                cv::Mat dstX;
                cv::Mat dstY;
                cv::Mat dstM;
                cv::Mat gray_image;
                // see if there is a waiting keystroke
                char key = cv::waitKey(10);
                if( key == 'q') 
                {
                    break;
                }
                else if(key == 's')
                {
                    saveBoolean = true;
                }
                else if(key == 'g')
                {
                    reset();
                    grayscale = true;   
                }
                else if(key == 'h')
                {
                    reset();
                    grayscale = true;
                    alternativeGrayScale = true;   
                }
                else if(key == 'b')
                {
                    reset();
                    blurFiveByFive = true;
                }
                else if(key == 'x')
                {
                    reset();
                    sobelX = true;
                }
                else if(key == 'y')
                {
                    reset();
                    sobelY = true;
                }
                else if(key == 'm')
                {
                    reset();
                    magnitudeGradient = true;
                }
                else if(key == 'l')
                {
                    reset();
                    blurQuantizeBoolean = true;
                }
                else if(key == 'c')
                {
                    reset();
                    cartoonBoolean = true;
                }
                else if(key=='i')
                {
                    reset();
                    medianFilterBoolean = true;
                    x="";
                }
                else if(key=='e')
                {
                    reset();
                    addCaptionBoolean = true;
                    x="";
                }
                if(grayscale)
                {
                    if(!alternativeGrayScale)
                    {
                        cvtColor( frame, dst, cv::COLOR_RGB2GRAY );
                        cv::imshow("Video", dst);
                    }     
                    else
                    {
                        greyscale(frame, dst);
                        cv::imshow("Video", dst);
                    }                   
                }
                else if(blurFiveByFive)
                {
                    blur5x5(frame,dst); 
                    cv::imshow("Video", dst);
                }
                else if(sobelX)
                {
                    sobelX3x3(frame,dstX);
                    cv::Mat displaysrc;
                    cv::convertScaleAbs( dstX, displaysrc, 2 );
                    displaysrc.copyTo(dst);
                    cv::imshow("Video", dst);
                }
                else if(sobelY)
                {
                    sobelY3x3(frame,dstY);
                    cv::Mat displaysrc;
                    cv::convertScaleAbs( dstY, displaysrc, 2 );
                    displaysrc.copyTo(dst);
                    cv::imshow("Video", dst);
                }
                else if(magnitudeGradient)
                {
                    sobelX3x3(frame,dstX);
                    sobelY3x3(frame,dstY);
                    magnitude(dstX,dstY,dstM);
                    dstM.copyTo(dst);
                    cv::imshow("Video", dst);
                }
                else if(blurQuantizeBoolean)
                {
                    blurQuantize(frame,dst,15);
                    cv::imshow("Video", dst);
                }
                else if(cartoonBoolean)
                {
                    cartoon(frame,dst,15,15);
                    cv::imshow("Video", dst);
                }
                else if(medianFilterBoolean)
                {      
                    medianFilter(frame,dst);
                    cv::imshow("Video", dst);
                }
                else if(addCaptionBoolean)
                {
                    if(x.size()==0)
                    {
                        cout << "Please enter the caption here: "; // Type a number and press enter
                        cin >> x; // Get user input from the keyboard
                    }          
                    addCaption(frame,dst,x);
                    cv::imshow("Video", dst);

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
        return(0);
}

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

void reset()
{
    grayscale = false;
    alternativeGrayScale = false;
    blurFiveByFive = false;
    sobelX = false;
    sobelY = false;
    magnitudeGradient = false;
    blurQuantizeBoolean = false;
    cartoonBoolean = false;
    saveBoolean = false;
    medianFilterBoolean = false;
    addCaptionBoolean = false;
} 