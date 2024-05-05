# Face-Recognition-Based-Attendance-System-
I have created a face recognition based attendance system where the attendance will be marked if the face is recognized from the webcam or from the uploaded photo
To Run this project first download necessary Libraries 
Then Run Main.py file in the terminal 

**How to Use this ? **
1. First upon running the script you will get the interface which was created using tkinter GUI
2. Then You need to create the datset from webcam or From Uploaded pic
3. If you select create dataset from webcam then the webcam will start running and asks the name and id of the person and it will take 100 images and store in seperate folder
4. If you select create dataset from uploaded pic then if you upload group picture each face will be detected and asks fro the name and id and it will generate 100 images from that one face
5. After this step you need to enter the details of the person in the excel sheet so that the data will be retreived from the excels sheet and display over the image 
6. Then you need to train the model
7. After that you can recognize from webcam
8. Also you can recognize from the uploaded picture 

Also i have Tried to implement the Anti spoofing technique using Blink detection. 
To checkout that run the EAR.py script for that you need to download shape_predictor_68_face_landmarks.dat file from internet 
