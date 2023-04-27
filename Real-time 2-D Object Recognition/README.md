# 2 Travel Days Used
- I was sick, and I talked to Professor and he was ok with me using 2 of my travel days to submit the assignment late.
# Operating System and IDE
- Windows 10
- Visual Studio Code, need to install C/C++ extension (C/C++ IntelliSense, debugging, and code browsing)
- Need to have g++ compiler on your machine 
    - It can be installed from https://www.msys2.org/ as a part of MSYS2 software package
    - Once downloaded, need to add the bin path to environment variables
- Need to have OpenCV libraries 
    - It can be downloaded from https://opencv.org/opencv-4-1-1/ 
    - Once downloaded, need to add the bin path to environment variables



# Instructions for Building Executable
- Create a .vscode folder in root, and add the 4 configuration files submitted with the assignment: c_cpp_properties.json, launch.json, settings.json, tasks.json
- For c_cpp_properties.json, need to modify settings on line 7 and 14 to point to compiler path on your local desktop
- For launch.json, need to modify settings on line 16 to point to compiler path on your local desktop
- For tasks.json, need to modify settings on line 6 to point to compiler path on your local desktop
- For tasks.json, also need to modify settings on line 14&15 to point to OpenCV path on your local desktop
- Graders are welcome to train the images from scratch, however if they want to start with a already trained DB file, objects.csv is attached.
- Once the environment is set up, drop the DB file objects.cvs in the same folder as main.exe
- Go to the main file, main.cpp, and build by press <CTRL+SHIFT+B>
- Once the build is finished, an executable in the name of the cpp file should appear in the root folder

# Instruction for running commands

- The main program is interacted through command line interfaces with the following format: <br />
````main.exe <actionType>```` <br />
For example: <br />
Task 1-4:<br /> ```` main.exe v ```` <br />
Press 't' for threshold<br/>
Press 'm' for clean-up<br/>
Press 'e' for segmentation<br/>
Press 'f' for feature extraction<br/>
Press 's' for save<br/>
Press 'q' for quit<br/>
Task 5:<br /> ```` main.exe t ```` <br />
Task 6:<br /> ```` main.exe ck1 ```` <br />
Task 7:<br /> ```` main.exe ck2 ```` <br />
Task 9 Video: <br /> https://www.youtube.com/watch?v=MColu6D7Dec  <br />
Extension: <br />```` main.exe olympus/pic.1082.jpg olympus Texture-Color-Entropy ScaledStandardDeviation 4```` <br />

# Instructions for testing extension
- The extension is on the training part: 
    - Used more than required 10 objects
    - Give users options to train objects by directories, faster and easier. Program will ask the user if they want to enter the images individually (i) or by directory (d). 


