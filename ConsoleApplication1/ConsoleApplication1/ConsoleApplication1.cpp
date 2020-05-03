// ConsoleApplication1.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//

#include "pch.h"

// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace cv;
using namespace std;

int thresh = 50;
const char* wndname = "Square Detection Demo";

static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
}
enum METHOD { MEAN, GAUSS, MEDIAN };
Mat adaptiveThresh(Mat I, int radius, float ratio, METHOD method = MEAN)
{
	Mat smooth;
	switch (method)
	{
	case MEAN:
		boxFilter(I, smooth, -1, Size(2 * radius + 1, 2 * radius + 1));
		break;
	case GAUSS:
		GaussianBlur(I, smooth, Size(2 * radius + 1, 2 * radius + 1), 0, 0);
		break;

	default:
		break;
	}
	I.convertTo(I, CV_32FC1);
	smooth.convertTo(smooth, CV_32FC1);
	Mat diff = I - (1.0 - ratio)*smooth;

	Mat out = Mat::zeros(diff.size().height, diff.size().width, CV_8UC1);

	for (int r = 0; r < out.rows; r++)
	{
		for (int c = 0; c < out.cols; c++)
		{
			if (diff.at<float>(r, c) >= 0)
				out.at<uchar>(r, c) = 255;
		}
	}

	return out;
}




// returns sequence of squares detected on the image.
static void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
	squares.clear();

	Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	image.copyTo(timg);
	// down-scale and upscale the image to filter out the noise
	//pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
	//pyrUp(pyr, timg, image.size());
	vector<vector<Point> > contours;

	int ch[] = { 0, 0 };
	//mixChannels(&timg, 1, &gray0, 1, ch, 1);
		
	Mat image_out;
		
	cvtColor(timg, image_out, CV_BGR2Lab);//將bgr to Lab
	mixChannels(&image_out, 1, &gray0, 1, ch, 1);//根據L灰階

	gray0 = adaptiveThresh(gray0,50,0.15, MEAN);//影像平滑 使用快速平均值平滑


	//這兩個是我看OPENCV 官方文檔用CANNY的方式 不太了解
	Canny(gray0, gray, 0, thresh, 5);// canny 邊緣檢測

	dilate(gray, gray, Mat(), Point(-1, -1));//膨胀



	// find contours and store them all as a list
	findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);//找出圖像中所有輪廓的座標集合
	vector<Point> approx;
	// test each contour

	for (size_t i = 0; i < contours.size(); i++)
	{
		// approximate contour with accuracy proportional
		// to the contour perimeter
		approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);//连续光滑曲线折线化 利入類似方形就會變方形
		// square contours should have 4 vertices after approximation
		// relatively large area (to filter out noisy contours)
		// and be convex.
		// Note: absolute value of an area is used because
		// area may be positive or negative - in accordance with the
		// contour orientation
		if (approx.size() == 4 &&// 符合 4頂點 且面積 1000~100000
			fabs(contourArea(approx)) > 1000 &&
			fabs(contourArea(approx)) < 100000 &&
			isContourConvex(approx))
		{
			double maxCosine = 0;
			for (int j = 2; j < 5; j++)//找出最小的角度
			{
				// find the maximum cosine of the angle between joint edges
				double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
				maxCosine = MAX(maxCosine, cosine);
			}
			// if cosines of all angles are small
			// (all angles are ~90 degree) then write quandrange
			// vertices to resultant sequence
			if (maxCosine < 0.2)// 約等於78度 而正方形90度
			{
				RotatedRect rect = minAreaRect(contours[i]);//该函数计算并返回指定点集的最小面积边界矩形（可能旋转）
				Mat vertices;

				boxPoints(rect, vertices);//vertices 是四頂點的座標
				
				//bottom left:0, top left:1, top right:2, bottom right:3
				cv::Point2f p0(vertices.row(0));
				cv::Point2f p1(vertices.row(1));
				cv::Point2f p2(vertices.row(2));
				cv::Point2f p3(vertices.row(3));

				double perimeter = sqrt(pow((p2.x-p3.x),2) + pow((p2.y-p3.y),2));//周長


				Point2f src[] = {p1,p2,p0,p3};//0>top left; 1>top right; 2>bottom left; 3>bottom right <<index
				Point2f dst[] = {Point2f(0,0),Point2f(perimeter,0),Point2f(0,perimeter),Point2f(perimeter,perimeter) };//這裡是給定將原始圖片方形四頂點轉正後要出現在新的圖像的座標點

				//使用投影轉換 更正4頂點
				Mat P = getPerspectiveTransform(src, dst);//投影轉換矩陣
				Mat result;
				warpPerspective(timg,result,P,timg.size());
				Mat correct = result(Rect(Point2f(0, 0), Point2f(perimeter, perimeter)));//這裡是選取轉正後的圖片的區域

				imshow("result", correct);
						
				squares.push_back(approx);						
			}
						
		}
	}
}
// the function draws all the squares in the image
static void drawSquares(Mat& image, const vector<vector<Point> >& squares)
{
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];
		int n = (int)squares[i].size();
		polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
	}
	imshow(wndname, image);
}
int main(int argc, char** argv)
{
	VideoCapture capture;
	capture.open(0); 

	//VideoCapture capture("D:\\Videos\\30s.mp4");// 影片輸入方式

	vector<vector<Point> > squares;

	cvNamedWindow(wndname, 1);
	while (capture.isOpened())
	{
		double start = (double)getTickCount();		

		Mat image;
		capture.read(image);
		findSquares(image, squares);
		drawSquares(image, squares);
		int c = waitKey(10);
		if (c == 27)
			break;

		start = ((double)cv::getTickCount() - start) / getTickFrequency();
		double fps = 1.0 / start;
		cout <<"fps:"<< setprecision(5) << fps<<endl;
	}
	return 0;
}