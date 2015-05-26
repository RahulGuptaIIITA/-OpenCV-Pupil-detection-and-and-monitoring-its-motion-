#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace cv;
using namespace std;


void EyeDetection(Mat);								//Eyes Detection Function prototype
void PupilDetection( Mat, Mat, int, int);					//Pupil Detection Function prototype
string eye_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";               	//Pre Trained classifiers for eyes
CascadeClassifier eye_cascade;							//CascadeClassifier class for detecting object(eyes) in video

int main()
{
	int t_f;
	Mat frame;
	int num = 0;

	if (!eye_cascade.load(eye_cascade_name))				//Load the Classifier
	{
		cout << "Error in loading the Classifier" << endl;
		return -1;
	};

	VideoCapture cap("vid9.mp4"); 						// open the video file for reading
	if ( !cap.isOpened() )  						// if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	t_f = cap.get(CV_CAP_PROP_FRAME_COUNT); 				//get the total number of frames in video
	cout << "Total Number of Frames : " << t_f << endl;

	while(1)
	{
		bool bSuccess = cap.read(frame); 				// read a new frame from video into Mat frame 
		if (!bSuccess) 							//if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}
		cout << "Frame No : " << ++num << endl;

		EyeDetection (frame);						//Calling function for detecting Eyes

		if(waitKey(30) == 27) 						//wait for'esc'key press for 30 ms.If'esc'key is prsd,breakloop
		{
			cout << "'esc' key is pressed by user" << endl;
			break; 
		}
	}
	return 0;
}


void EyeDetection (Mat Orig_frame)
{
	Mat L_crop;								
	Mat R_crop;
	Mat Gray_frame;
	Mat Res_frame;
	vector<Rect>eyes;							

	cvtColor(Orig_frame, Gray_frame, CV_BGR2GRAY);				//Converts RGB to GrayScale
	equalizeHist(Gray_frame, Gray_frame);					//Using histogram Equalization tech for improving contrast

	eye_cascade.detectMultiScale(Gray_frame, eyes, 1.15, 4, 0 | CASCADE_SCALE_IMAGE,Size(10, 10)); //Detect Eyes

	Rect L_roi;								//region of interest
	Rect R_roi;	

	int x1, y1;								//(x1,y1) is indx of left detected eye
	int w1, h1;								//width and height of detected eye

	int x2, y2;								//(x2,y2) is indx of right detected eye
	int w2, h2;								

	int e_x1, e_y1;								//(e_x1,e_y1) is indx of left eye after pruning
	int e_w1, e_h1;								//width and height of eye after pruning

	int e_x2, e_y2;								//(e_x2,e_y2) is indx of right eye after pruning
	int e_w2, e_h2;

	if ( !eyes.empty() ) {

		if ( eyes[0].width > 0 && eyes[0].height > 0) {			//First Detected eyes
			x1 = eyes[0].x;						//Dimesnions of Left Detected eye in frame
			y1 = eyes[0].y;
			w1 = eyes[0].width;
			h1 = eyes[0].height;

			L_roi.x = e_x1 = x1 + .11*w1;                           //pruning Left eye to eliminate unwanted pixels (resizing)
			L_roi.y = e_y1 = y1 + .15*h1;
			L_roi.width = e_w1 = .8*w1;
			L_roi.height = e_h1 = .65*h1;

			Point L_pt1(e_x1,e_y1);
			Point L_pt2(e_x1 + e_w1, e_y1 + e_h1);

			L_crop = Gray_frame(L_roi);
			rectangle(Orig_frame, L_pt1, L_pt2, Scalar(0, 255, 0), 2, 8, 0);

			PupilDetection(L_crop, Orig_frame, L_roi.x, L_roi.y);	//Calling PupilDetection method
		}
		

		if ( eyes[1].width > 0 && eyes[1].height > 0) {			//Second Detected eyes
			x2 = eyes[1].x;						//Dimension of Right Detected eye in frame
			y2 = eyes[1].y;
			w2 = eyes[1].width;
			h2 = eyes[1].height;

			R_roi.x = e_x2 = x2 + .11*w2;                           //pruning Right eye to eliminate unwanted pixels (resizing)
			R_roi.y = e_y2 = y2 + .15*h2;
			R_roi.width = e_w2 = .8*w2;
			R_roi.height = e_h2 = .65*h2;

			Point R_pt1(e_x2, e_y2);
			Point R_pt2(e_x2 + e_w2, e_y2 + e_h2);

			R_crop = Gray_frame(R_roi);
			rectangle(Orig_frame, R_pt1, R_pt2, Scalar(0, 255, 0), 2, 8, 0);

			PupilDetection(R_crop, Orig_frame, R_roi.x, R_roi.y);	//Calling PupilDetection method 
		}
	}
}

void PupilDetection( Mat frame,  Mat Orig_frame, int px, int py)
{
	int val;
	int rows;
	int cols;
	int min_val = 270;

	Size s = frame.size();
	rows = s.height;
	cols = s.width;
	
	//GaussianBlur(frame, frame, Size(11, 11), 5, 5);
	for ( int i = 0; i < rows; i++ ) {					//Iterating over image to find out min intesity pixel
		for ( int j = 0; j < cols; j++ ) {
			val = frame.at<uchar>(i,j);
			if ( val < min_val) {
				min_val = val;
			}
		}
	}
	//cout << min_val << endl;
	
	for ( int i = 0; i < rows; i++ ) {					//Performing Binarization (Pupil has min intensity pixels)
		for ( int j = 0; j < cols; j++ ) {
			val = frame.at<uchar>(i,j);
			if ((abs)(val - min_val) < 4 ) {
				frame.at<uchar>(i,j) = 255;
			
			} else {
				frame.at<uchar>(i,j) = 0;
			}
		}
	}

	dilate (frame, frame, Mat(), Point(-1, -1), 1, 1, 1);			//Morphology's  Operations to remove unwanted pixels 
	erode (frame, frame, Mat(), Point(-1, -1), 2, 1, 1);

	//imshow("image", frame);

	vector<vector<Point> >contours;						//Contour Detection 
	findContours(frame.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(frame, contours, -1, CV_RGB(255,255,255), -1);		//Fill hole in each contour

	int flag = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);    			//Area of Detected object
		Rect rect = boundingRect(contours[i]); 				//Bounding box 
		int radius = rect.width/2;                     			//Radius
		//cout << "radius " << radius << endl;
		//cout << "area  " << area << endl;
		if ( radius < 7 ) {
			radius = 7;
		
		} else if (radius > 20 ) {
			radius = 20;
		
		} if (( area >= 10 && area <= 550) && flag == 0) {		//Eliminating non desirable contours  

			circle(Orig_frame, Point(px + rect.x + radius, py + rect.y + radius), radius, CV_RGB(255,0,0), 2);
			flag = 1;
		}
	}
	imshow("Eye Tracking", Orig_frame);					//Displaying Detected Pupil
}
