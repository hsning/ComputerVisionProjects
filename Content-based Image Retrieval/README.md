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
- Once the environment is set up, drop the database image directory (olympus) in the root folder
- Go to the main file, main.cpp, and build by press <CTRL+SHIFT+B>
- Once the build is finished, an executable in the name of the cpp file should appear in the root folder

# Instruction for running commands

- The main program is interacted through command line interfaces with the following format: <br />
````main.exe <target path> <image database directory path> <feature type> <matching method> <numberOfMatches>```` <br />
For example: <br />
Task 1:<br /> ```` main.exe olympus/pic.1016.jpg olympus 9x9 Euclidean 4 ```` <br />
Task 2:<br /> ```` main.exe olympus/pic.0164.jpg olympus Histogram Histogram-Intersection 4 ```` <br />
Task 3:<br /> ```` main.exe olympus/pic.0274.jpg olympus Multi-Histogram Histogram-Intersection 4 ```` <br />
Task 4:<br /> ```` main.exe olympus/pic.0535.jpg olympus TextureAndColor Histogram-Intersection 4 ```` <br />
Task 5: <br />```` main.exe olympus/pic.0013.jpg olympus Custom Custom 10 ```` <br />
Extension: <br />```` main.exe olympus/pic.1082.jpg olympus Texture-Color-Entropy ScaledStandardDeviation 4```` <br />

# Instructions for testing extension
- Use Command Prompt or Powershell or any other interfaces to run command in the following format:

    ```` main.exe olympus/pic.1082.jpg olympus Texture-Color-Entropy ScaledStandardDeviation <numberOfMatches> ````

    For example: <br />
    ```` main.exe olympus/pic.1082.jpg olympus Texture-Color-Entropy ScaledStandardDeviation 4 ````


