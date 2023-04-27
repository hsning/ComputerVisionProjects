By: Hao Sheng Ning
# 3 Travel Days Used
- I had time conflict, and I talked to Professor and he was ok with me using 3 of my travel days to submit the assignment late.
# Operating System and IDE
- Windows 10
- Visual Studio Code, need to install Code Runner & Python extension (IntelliSense (Pylance), Linting, Debugging (multi-threaded, remote), Jupyter Notebooks, code formatting, ...)
- Need to have python on your machine 
    - The installer can be downloaded from https://www.python.org/downloads/windows/ 
    - Once downloaded, need to run the installer, and REMEMBER to add python to your PATH (inside environment variables)
- Need to have pip on your machine 
    - Need to download the script from https://bootstrap.pypa.io/get-pip.py.
    - Once downloaded, Open a terminal/command prompt, cd to the folder containing the get-pip.py file and run:</br>
    ````py get-pip.py````
    - For detailed documentation, please refer https://pip.pypa.io/en/stable/installation/
- Need to have matplotlib modules
    - Open a terminal/command prompt, and run:</br>
    ````python -m pip install -U pip````
    ````python -m pip install -U matplotlib````
    - For detailed documentation, please refer https://matplotlib.org/stable/users/installing/index.html
- Need to have pytorch modules
    - Open a terminal/command prompt, and run: </br>
    ````pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 ````
    - For detailed documentation, please refer https://pytorch.org/get-started/locally/
- Need to have numpy modules
    - Open a terminal/command prompt, and run: </br>
    ````pip install numpy````
    - For detailed documentation, please refer https://numpy.org/install/


# Instruction for Running the program (extensions not included)

There are in total 10 programs for this assignment, we will break them down into Tasks:

#### Task 1 (A-E): 
- Create an empty folder inside the root called "data"
- Go to "train.py" file and press <CTRL+ALT+N> to run the program, this program will first show you the ground truth labels of first 6 images, close them will prompt the program to train the model and it will output the progress and the running accuracy and loss in the terminal. Once the training is done, a plot of the training and testing accuracy graph will pop up, close it will save the model to the "result/model.pth" file.

#### Task 1 (F-G): 
- Unzip the testData.zip to root folder. Feel free to add your own test data here.
- Go to "execute.py" file and press <CTRL+ALT+N> to run the program, this program load the model from the "result/model.pth" file created from task above. Then it will test the images and output the resulting values in the terminal and the images with predictions in the plot. Close the plot will prompt the program to test the handwritten images saved inside "testData/test" folder. If testers want to add their own handwritten images for testing purposes, images can be added in this path: "testData/test". The test result (images with the classification) will be outputted in the plot.

#### Task 2: 
- Go to "analyze.py" file and press <CTRL+ALT+N> to run the program. This program will first show the different filters in the plot, then closing the plot window will prompt the program to show the filters and corresponding filtered images in another plot.

#### Task 3: 
- Unzip the greek_train-3.zip and testDataGreek-3.zip to root folder. The greek_train-3.zip is the same one as given.
- Go to "executeGreek.py" file and press <CTRL+ALT+N> to run the program. This program will first show you the ground truth labels of first 3 Greek Letter images, close them will prompt the program to train the model. After model is fully trained, the corresponding training loss rate graph will show up in a plot, closing it will prompt the program to show the three handwritten greek letters and their classifications in another plot. 

#### Task 4:
- Optimize Number of Filters:</br>
  Go to "optimizeNumberOfFilters.py" and press <CTRL+ALT+N> to run the program. This program will run the training using 12 different number of filter and output each's accuracy. This process will do the training for 12 times, hence will take more than 30 minutes to complete. 
- Optimize Number of Hidden Nodes:</br>
  Go to "optimizeNumberOfHiddenNodes.py" and press <CTRL+ALT+N> to run the program. This program will run the training using 12 different number of hidden nodes and output each's accuracy. This process will do the training for 12 times, hence will take more than 30 minutes to complete. 
- Optimize Dropout rates:</br>
  Go to "optimizeDropoutRate.py" and press <CTRL+ALT+N> to run the program. This program will run the training using 13 different dropout rates and output each's accuracy. This process will do the training for 13 times, hence will take more than 30 minutes to complete.
- Optimize Batch Sizes:</br>
  Go to "optimizeBatchSize.py" and press <CTRL+ALT+N> to run the program. This program will run the training using 11 different batch sizes and output each's accuracy. This process will do the training for 11 times, hence will take more than 30 minutes to complete.  
- Optimize Activation Functions:</br>
  Go to "optimizeActivationFunction.py" and press <CTRL+ALT+N> to run the program. This program will run the training using 12 different activation functions and output each's accuracy. This process will do the training for 12 times, hence will take more than 30 minutes to complete.  
- Optimize Everything combined:</br>
  Go to "optimizecombined.py" and press <CTRL+ALT+N> to run the program. This program will run the training using all the optimized variable. This process will do the training for only 1 time, hence will take at most 5 minutes.

# Instructions for testing extension
The extension part is divided into 2 parts:</br>
1. Test 6 Greek Letters and try different dimensions to improve the accuracy:</br>
    - Unzip the greek_train-6.zip and testDataGreek-6.zip to root folder. Feel free to add extra training/testing files here.
    - Go to "Extention_executeGreek.py" file and press <CTRL+ALT+N> to run the program. This program will first show you the ground truth labels of first 3 Greek Letter images, close them will prompt the program to train the model. After model is fully trained, a plot of six handwritten greek letters and their classifications will show up.
2. Load Resnet50 and analyze its first conv1 layer filters and its filtered images:
    - Go to "Extension_analyze.py" file and press <CTRL+ALT+N> to run the program. This program will show you the filters and corresponding filtered images in another plot.


