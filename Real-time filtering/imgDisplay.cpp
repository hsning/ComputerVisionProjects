#include <iostream>

#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <string>
#include <ctime>
using namespace cv;
using namespace std;
int main( int argc, char** argv )
{
     // code block to be executed
    
        std::string img = "000000009600.tif";
        Mat srcImage = imread(img);
        if (!srcImage.data) 
        {
            return 1;
        }
        std::cout<<srcImage.channels()<<endl;
        while(true)
        {
        imshow("srcImage", srcImage);
        waitKey(1);
        // see if there is a waiting keystroke
        char key = cv::waitKey(10);
        if( key == 'q') 
        {
            break;
        }
        }
    return 0;
    //std::cout << "aa" << std::endl;    
    
}