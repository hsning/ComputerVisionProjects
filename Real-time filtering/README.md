# Operating System and IDE
- Windows 10
- Visual Studio Code, need to install C/C++ extension (C/C++ IntelliSense, debugging, and code browsing)
- Need to have g++ compiler on your machine 
    - It can be installed from https://www.msys2.org/ as a part of MSYS2 software package
    - Once downloaded, need to add the bin path to environment variables
- Need to have OpenCV libraries 
    - It can be downloaded from https://opencv.org/opencv-4-1-1/ 
    - Once downloaded, need to add the bin path to environment variables



# Instructions for Running Executable
- Create a .vscode folder in root, and add the 4 configuration files submitted with the assignment: c_cpp_properties.json, launch.json, settings.json, tasks.json
- For c_cpp_properties.json, need to modify settings on line 7 and 14 to point to compiler path on your local desktop
- For launch.json, need to modify settings on line 16 to point to compiler path on your local desktop
- For tasks.json, need to modify settings on line 6 to point to compiler path on your local desktop
- For tasks.json, also need to modify settings on line 14&15 to point to OpenCV path on your local desktop
- Once the environment is set up, go to the main file, whether that be imgDisplay.cpp or vidDisplay.cpp and build by press <CTRL+SHIFT+B>
- Once the build is finished, an executable in the name of the cpp file should appear in the root folder
- Run the executable by double clicking it

# Instructions for testing extension
- Have focus on the video window, and press 'e', the video should have stopped until a caption is entered
- Command prompt on the side should ask you to enter the caption
- Once entered, press enter, and the video should resume with the caption on the left hand corner
- Can press 's' anytime to save the captioned image


