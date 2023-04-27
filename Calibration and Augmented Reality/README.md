# Operating System and IDE
- Windows 10
- Visual Studio Code, need to install C/C++ extension (C/C++ IntelliSense, debugging, and code browsing)
- Need to have g++ compiler on your machine 
    - It can be installed from https://www.msys2.org/ as a part of MSYS2 software package
    - Once downloaded, need to add the bin path to environment variables
- Need to have OpenCV libraries 
    - It can be downloaded from https://github.com/hsning/OpenCVLibraries 
    - Once downloaded, unzip it, and need to add the bin path to environment variables



# Instructions for Building Executable
- Create a .vscode folder in root, and add the 4 configuration files submitted with the assignment: c_cpp_properties.json, launch.json, settings.json, tasks.json
- For c_cpp_properties.json, need to modify settings on line 7 and 14 to point to compiler path on your local desktop
- For launch.json, need to modify settings on line 16 to point to compiler path on your local desktop
- For tasks.json, need to modify settings on line 6 to point to compiler path on your local desktop
- For tasks.json, also need to modify settings on line 14&15 to point to OpenCV path on your local desktop
- Graders are welcome to train the images from scratch, however if they want to start with a already trained DB file, objects.csv is attached.
- Once the environment is set up, drop the DB file objects.cvs in the same folder as main.exe
- Go to the main file, main.cpp, and build by press <CTRL+SHIFT+B>
- Go to file, surf.cpp, and build by press <CTRL+SHIFT+B>
- Go to file, imageAR.cpp, and build by press <CTRL+SHIFT+B>
- Once the build is finished, 3 executables in the name of the cpp file should appear in the root folder

# Instruction for running commands

There are 3 programs in this assignments:

1.  The main program is interacted through command line interfaces with the following format: <br />
````main.exe <actionType>```` <br />
For example: <br />
Task 1-3:<br /> ````main.exe c````: used to calibarte the camera<br />
    - Press 'e' for corner extraction<br/>
    - Press 's' for saving the image<br/>
    - Press 'c' for calibration, a prompt will appear on the command ask you whether you want to save the matrix and coefficient to a file called intrinsicParameters.txt or not, <br/>
    - Press 'q' for quit<br/>

    Task 4-6:<br /> ````main.exe v````<br />
    - Place the image in front of the camera, once the corners are detected, the virtual object will appear<br/>
    - Press 'o' for corners only<br/>
    - Press 'q' for quit<br/>
 <br />
2.   The surf program is interacted through command line interfaces with the following format:<br />
````surf.exe v```` <br />
            - Opened a camera to capture pattern , and video will show the SURF key-points<br /> 
and <br />
````surf.exe c <image1> <image2>````<br/>
For example:<br/>
````surf.exe c tree.jpg and treeR.jpg````<br/>
** tree.jpg and treeR.jpg are bundled with the submission, and they can be used as a testing base
treeR.jpg is of the exact scene as the tree.jpg but in upside down rotation<br />
    - Compare 2 images and show matching SURF key-points 

3.  The imageAR program is interacted through command line interfaces with the following format:<br/>
````imageAR.exe <g for generate or c for calibrate or v for video> <image path, needed only if actionType is g for generate>````<br/>
For example:<br/>
````imageAR.exe g tree.jpg```` : can be used to generated Aruco marker cornered image<br />
````imageAR.exe c````: used to calibrate the camera, press e for corner extraction, press s for saving the point set to the list, and press c for calibration<br />
    - Press 'e' for corner extraction
    - Press 's' for saving the image
    - Press 'c' for calibration, a prompt will appear on the command ask you whether you want to save the matrix and coefficient to a file called intrinsicParametersArucoImage.txt or not
    - Press 'q' for quit<br/>

    ````imageAR.exe v````: used to project virtual object, place image in front of the camera, and virtual object will appear on the video

# Instructions for testing extension
The extension is about generating Aruco marker cornered images, and use the resulting image to calibrate the camera to compute camera matrix and distortion coefficient used for the final projection step. Once calibrated, camera will capture the resulting image, extract the Aruco markers to compute its pose and project a virtual object on to that image. The virtual object will move with the image. Steps for testing are as follows:
1. To generate image with Aruco marker corners: ````imageAR.exe g <input image>````
2. Calibrate: run  ````imageAR.exe c```` place an image in front of the camera and press 'e' to start extract corners, and press 's' to save at least images, when at least 5 captured, press 'c' to calibrate
2. Virtual Object Projection: run  ````imageAR.exe v```` place an image in front of the camera and once the corners are detected, a virtual object will appear


