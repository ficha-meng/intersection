// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU 

#include "../include/Header.h"
#include "../include/gms_matcher.h"
typedef struct histinfo
{
	int idx;
	int value;
};
void GmsMatch(Mat &img1, Mat &img2);

void runImagePair(){
	Mat img1 = imread("C:\\Users\\hu\\Desktop\\denoise\\1.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	Mat img2 = imread("C:\\Users\\hu\\Desktop\\denoise\\2.jpg", CV_LOAD_IMAGE_ANYCOLOR);

	imresize(img1, 480);
	imresize(img2, 480);

	GmsMatch(img1, img2);
}
void sobelgradient(Mat mv, Mat &grad,int kernel_size, int scale, int delta, int ddepth)
{
	char* window_name = "Sobel Demo - Simple Edge Detector";

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(mv, grad_x, ddepth, 1, 0, kernel_size, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(mv, grad_y, ddepth, 0, 1, kernel_size, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
}
int histgram3(Mat src)
{
	//Mat src, dst;

	/// Load image
	//src = imread(argv[1], 1);

	//if( !src.data )
	//{ return -1; }

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 128, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 600;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImageb(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImageg(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImager(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);

		line(histImageb, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImageg, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImager, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	imshow("calcHist Demo", histImage);
	imshow("calcHist B", histImageb);
	imshow("calcHist G", histImageg);
	imshow("calcHist R", histImager);

	waitKey(0);

	return 0;
}

int histgram(vector<Mat> bgr_planes, vector<Mat>&hist, vector<histinfo> &redATT, vector<histinfo> &greenATT)
{
	//Mat src, dst;

	/// Load image
	//src = imread(argv[1], 1);

	//if( !src.data )
	//{ return -1; }

	/// Separate the image in 3 places ( B, G and R )
	//vector<Mat> bgr_planes;
	//split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 255 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	//Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), hist[0], 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), hist[1], 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), hist[2], 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 600;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImageb(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImageg(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImager(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	//normalize(hist[0], hist[0], 0, 255, NORM_MINMAX, -1, Mat());
	//normalize(hist[1], hist[1], 0, 255, NORM_MINMAX, -1, Mat());
	//normalize(hist[2], hist[2], 0, 255, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		//line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist[0].at<float>(i - 1))),
		//	Point(bin_w*(i), hist_h - cvRound(hist[0].at<float>(i))),
		//	Scalar(255, 0, 0), 2, 8, 0);
		//line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist[1].at<float>(i - 1))),
		//	Point(bin_w*(i), hist_h - cvRound(hist[1].at<float>(i))),
		//	Scalar(0, 255, 0), 2, 8, 0);
		//line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist[2].at<float>(i - 1))),
		//	Point(bin_w*(i), hist_h - cvRound(hist[2].at<float>(i))),
		//	Scalar(0, 0, 255), 2, 8, 0);

		line(histImageb, Point(bin_w*(i - 1), hist_h - cvRound(hist[0].at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist[0].at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImageg, Point(bin_w*(i - 1), hist_h - cvRound(hist[1].at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist[1].at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImager, Point(bin_w*(i - 1), hist_h - cvRound(hist[2].at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist[2].at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);

		line(histImage, Point2f(bin_w*(i - 1), hist_h - (cvRound(hist[2].at<float>(i - 1)) - cvRound(hist[0].at<float>(i - 1))) / (hist[2].at<float>(i - i))),
			Point2f(bin_w*(i), hist_h - (cvRound(hist[2].at<float>(i)) - cvRound(hist[0].at<float>(i))) / 1),
			Scalar(0, 0, 255), 2, 8, 0);
		

		line(histImage, Point2f(bin_w*(i - 1), hist_h - (cvRound(hist[1].at<float>(i - 1)) - cvRound(hist[0].at<float>(i - 1))) / (hist[1].at<float>(i - i))),
			Point2f(bin_w*(i), hist_h - (cvRound(hist[1].at<float>(i)) - cvRound(hist[0].at<float>(i))) / 1),
			Scalar(0, 255, 0), 2, 8, 0);
	}
	for (int i = 0; i < histSize; i++)
	{
		histinfo tmp1,tmp2;
		tmp1.idx = i; tmp2.idx = i;
		tmp1.value = ((hist[1].at<float>(i))) / (1);
		greenATT.push_back(tmp1);
		tmp2.value = ((hist[2].at<float>(i))) / (1);
		//tmp2.value = tmp2.value - tmp1.value;
		redATT.push_back(tmp2);
	}
	/// Display
	imshow("calcHist", histImage);
	//imshow("calcHist B", histImageb);
	//imshow("calcHist G", histImageg);
	//imshow("calcHist R", histImager);
	return 0;
}
bool myfunction(histinfo i, histinfo j) { return (i.value < j.value); };
void CalcDarkChannel(Mat& darkchannel, Mat& brightchannel, Mat&input, int radius)
{
	int height = input.rows;
	int width = input.cols;


	int st_row, ed_row;
	int st_col, ed_col;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			st_row = i - radius, ed_row = i + radius;
			st_col = j - radius, ed_col = j + radius;

			st_row = st_row < 0 ? 0 : st_row;
			ed_row = ed_row >= height ? (height - 1) : ed_row;
			st_col = st_col < 0 ? 0 : st_col;
			ed_col = ed_col >= width ? (width - 1) : ed_col;

			int cur = 0;
			int min = 300;
			int max = 0;
			for (int m = st_row; m <= ed_row; m++)
			{
				for (int n = st_col; n <= ed_col; n++)
				{
					for (int k = 0; k < 3; k++)
					{
						cur = input.at<Vec3b>(m, n)[k];
						if (cur < min) min = cur;
						if (cur > max) max = cur;
					}
				}
			}

			darkchannel.at<uchar>(i, j) = min;
			brightchannel.at<uchar>(i, j) = max;

		}
	}
}
int main()
{
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0){ cuda::setDevice(0); }
#endif // USE_GPU

	//runImagePair();
	
	Mat src = imread("C:\\Users\\hu\\Desktop\\denoise\\7.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	Mat srccp;
	src.copyTo(srccp);
	vector<Mat> mv, grad,hist;
	split(src, mv);
	split(src, grad);
	split(src, hist);
	Mat darkchannel = Mat::ones(src.cols,src.rows,CV_8U);
	Mat brightchannel = Mat::ones(src.cols, src.rows, CV_8U);
	cvtColor(src,darkchannel,CV_BGR2GRAY);
	cvtColor(src, brightchannel, CV_BGR2GRAY);
	CalcDarkChannel(darkchannel, brightchannel, src, 0);
	grad[2] = grad[2]-darkchannel;
	grad[0] = brightchannel-grad[0];
	grad[1] = grad[2] + grad[0];
	normalize(grad[2], grad[2],255,0,CV_MINMAX);
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	//sobelgradient(mv[0], grad[0], kernel_size, scale, delta, ddepth);
	//sobelgradient(mv[1], grad[1], kernel_size, scale, delta, ddepth);
	//sobelgradient(mv[2], grad[2], kernel_size, scale, delta, ddepth);


	//Laplacian(mv[0], grad[0], ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	//Laplacian(mv[1], grad[1], ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	//Laplacian(mv[2], grad[2], ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	//convertScaleAbs(grad[0], grad[0]);
	//convertScaleAbs(grad[1], grad[1]);
	//convertScaleAbs(grad[2], grad[2]);

	//normalize(grad[0], grad[0], 0, 255, NORM_MINMAX, -1, Mat());
	//normalize(grad[1], grad[1], 0, 255, NORM_MINMAX, -1, Mat());
	//normalize(grad[2], grad[2], 0, 255, NORM_MINMAX, -1, Mat());

	//Mat bgr;
	//merge(grad, bgr);
	//histgram3(bgr);
	vector<histinfo> redATT, greenATT;
	//histgram(grad, hist, redATT, greenATT);
	//std::sort(redATT.begin(), redATT.end(), myfunction);
	//std::sort(greenATT.begin(), greenATT.end(), myfunction);
	//for (int s = 50; s < 100; s++)
	//{
	//	if (redATT[s].value == 0) continue;
	//	for (int i = 0; i < grad[2].cols; i++)
	//	{
	//		for (int j = 0; j < grad[2].rows; j++)
	//		{
	//			if (grad[1].at<uchar>(j, i) == greenATT[s].idx)
	//			{
	//				//circle(srccp, Point(i, j), 1, Scalar(0, 255, 0), 1);
	//			}

	//			if (grad[2].at<uchar>(j, i) == redATT[s].idx)
	//			{
	//				//circle(srccp, Point(i, j), 1, Scalar(0, 0, 255), 1);
	//			}
	//		}
	//	}
	//}
	imshow("srccp",srccp);
	//imshow("src", src);
	imshow("sobelb", grad[0]);
	imshow("sobelg", grad[1]);
	imshow("sobelr", grad[2]);

	waitKey(0);
	return 0;
}


void GmsMatch(Mat &img1, Mat &img2){
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;

	//ORB orb;
	//orb.setAlgori
	//orb.setFastThreshold(0);
	//orb.detectAndCompute(img1, Mat(), kp1, d1);
	//orb.detectAndCompute(img2, Mat(), kp2, d2);

	ORB orb1(10000, 1.1f, 15, 15, 0, 3, ORB::FAST_SCORE, 31);
	ORB orb2(10000, 1.1f, 15, 15, 0, 3, ORB::FAST_SCORE, 31);
	orb1(img1, Mat(), kp1, d1, false);
	orb2(img2, Mat(), kp2, d2, false);
	

#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
#endif

	// GMS filter
	int num_inliers = 0;
	std::vector<bool> vbInliers;
	gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
	num_inliers = gms.GetInlierMask(vbInliers, false, false);

	cout << "Get total " << num_inliers << " matches." << endl;

	// draw matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}

	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
	imshow("show", show);
	waitKey();
}


