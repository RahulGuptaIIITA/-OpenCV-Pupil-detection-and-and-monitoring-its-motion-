# Pupil Detection And Monitoring Its Motion To Control Application (OpenCV)
This was an independent Project aimed to work on 'Assistive Technology' to help the physically challenged fit in the general populace' 

The project is developed in MATLAB. The module will detects pupil and tracks its motion to take the decision. It continously record video, extract alternate frames, process them, and detect pupil using 'Viola Jones' algorithm for 'Object Detection'. 

Project mainly has two component.
a) Detecting Pupil
b) Monitoring its motion


For Pupil Detection, steps involved are as follows:-

1) Used Viola Jones for eye detection. EyeDetection() function in code does this job.
2) After successfully detecting eyes,  I did some pruning to remove irrelevant part (Eyebrows) from the detected part.
3) Now After Detecting Eye, I tried to extract pupil out of it. PupilDetection() function in code does this job. 
    -First convert the image into grayScale.   
    -We know that in our eyes/eyeball, pupil has less intensity pixels (Black in GrayScale image). 
     So I converted the grayscale image into binary image. For threshold, if pixel's intensity value is within the minIntensity pixel + K (where K is some factor like 4, 5) then I would include this pixel into ROI.
    -After having Binary image, I used 'Contour Detection'to get all objects in a binary Image.
    -From all detected object, eliminated unwanted one on the basis of certain parameters. I used 'area' to eliminate undesirable thing.
4) After successfully detecting pupil, just displayed it on the original image by drawing a circle around it.  


For Monitoring Its Motion, steps involved are as follows:-

1) Used the first frame of the vido as the frame of reference (assuming the person is looking straight, and eyes are in middle), also validated it against the check that position of both the pupil would be almost equidistant from the center of nose.
2) Now, processed each frame, calculated the postion of pupil and distance it has away from the center of nose. If left pupil traversed more, then person is looking in right direction If right pupil has traversed more then in right direction. If the position doesn't change, he is looking straight and if fails to find pupil then eyes are closed. Considering this pattern, rules were made to control a application.

Overall, I was able to achieve the accuracy of 82% on the self made dataset ( created by taking all the batchmates ) 
Contact GitHub API Training Shop Blog About
