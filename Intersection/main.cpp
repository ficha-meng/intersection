#include <opencv.hpp>
#include <stdio.h>
#include <string.h>
#include "math.h"
#include <vector>
#include <time.h>
#include "fitline.h"
using namespace std;
using namespace cv;
struct RectPoint
{
    cv::Point p[4];
};
struct Line
{
    float theta;
    float k;
    int endp1;
    int endp2;
};


bool ifIntersection(RectPoint ps,float theta,float k)
{  
    bool ret=false;
    int f0=theta*ps.p[0].x+k-ps.p[0].y;
    
    if(f0==0)
    {
        ret=true;
    }
    else if(f0<0)
    {
        if(theta<0)
        {
            ret=false;
        }
        else if(theta>0)
        {
            int f1=theta*ps.p[1].x+k-ps.p[1].y;
            if(f1>0)
            {
                ret=true;
            }
            else if(f1<0)
            {
                ret=false;
            }
        }
    }
    else if(f0>0)
    {
        if(theta<0)
        {
            int f2=theta*ps.p[2].x+k-ps.p[2].y;
            if(f2>0)
            {
                ret=false;
            }
            else
            {
                ret=true;
            }
        }
        else if(theta>0)
        {
            int f3=theta*ps.p[3].x+k-ps.p[3].y;
            if(f3>0)
            {
                ret=false;
            }
            else
            {
                ret=true;
            }
        }
    }
    return ret;
}
//#define INPUT1
//#define INPUT2
#define INPUT3
#define FIT
#ifdef INPUT2
#define SHOW
#endif
int main(int argc, char * argv[])
{
#ifdef INPUT1
    FILE *file=fopen("/Users/mengyang/Desktop/input1.txt", "r");
#endif
#ifdef INPUT2
    FILE *file=fopen("C:\\Users\\hu\\Desktop\\rectangle\\input2.txt", "r");
#endif
#ifdef INPUT3
    FILE *file=fopen("C:\\Users\\hu\\Desktop\\rectangle\\input3.txt", "r");
#endif
    
    if(file==NULL)
    {
        printf("Error!\n");
    }
    
    int numRect = 0;
    fscanf(file, "%d",&numRect);
    
  
#ifdef FIT
	vector<double> X,Y;
	vector<RectPoint> pointSet;
	vector<Line> linePara;
	cv::Mat canvas = cv::Mat(200,200,2);
	for (int r = 0; r<numRect; r++)
	{
		cv::Rect tmp;
		fscanf(file, "%d %d %d %d\n", &tmp.x, &tmp.y, &tmp.width, &tmp.height);
#ifdef SHOW
		tmp = Rect(tmp.x * 10, tmp.y * 10, tmp.width * 10, tmp.height * 10);
#endif
		X.push_back(cv::Point2d(tmp.x, tmp.y).x);
		Y.push_back(cv::Point2d(tmp.x, tmp.y).y);

		X.push_back(cv::Point2d(tmp.x + tmp.width, tmp.y).x);
		Y.push_back(cv::Point2d(tmp.x + tmp.width, tmp.y).y);

		X.push_back(cv::Point2d(tmp.x + tmp.width, tmp.y + tmp.height).x);
		Y.push_back(cv::Point2d(tmp.x + tmp.width, tmp.y + tmp.height).y);

		X.push_back(cv::Point2d(tmp.x, tmp.y + tmp.height).x);
		Y.push_back(cv::Point2d(tmp.x, tmp.y + tmp.height).y);

		RectPoint ps;
		ps.p[0] = cv::Point(tmp.x, tmp.y);
		ps.p[1] = cv::Point(tmp.x + tmp.width, tmp.y);
		ps.p[2] = cv::Point(tmp.x + tmp.width, tmp.y + tmp.height);
		ps.p[3] = cv::Point(tmp.x, tmp.y + tmp.height);
		pointSet.push_back(ps);
#ifdef SHOW
		circle(canvas, cv::Point2d(tmp.x, tmp.y), 1, cv::Scalar(255, 255, 255), 2);
		circle(canvas, cv::Point2d(tmp.x + tmp.width, tmp.y), 1, cv::Scalar(255, 255, 255), 2);
		circle(canvas, cv::Point2d(tmp.x + tmp.width, tmp.y + tmp.height), 1, cv::Scalar(255, 255, 255), 2);
		circle(canvas, cv::Point2d(tmp.x, tmp.y + tmp.height), 1, cv::Scalar(255, 255, 255), 2);
		cv::rectangle(canvas,tmp,cv::Scalar(255,255,255),2);
#endif
	}
	
	czy::Fit fit;
	fit.linearFit(X, Y, true);
	std::printf("y=%fx+%f\r\nssr:%f,sse=%f,rmse:%f,factor:%f\n", fit.getSlope(), fit.getIntercept()
		,fit.getSSR(),fit.getSSE(),fit.getRMSE(),fit.getR_square());
#ifdef SHOW
	line(canvas, Point(0, fit.getIntercept()), Point(-fit.getIntercept() / fit.getSlope(), 0), cv::Scalar(255, 255, 255), 2);
	imshow("line", canvas);
	waitKey(-1);
#endif
	for(int i=0;i<numRect;i++)
	{
		RectPoint ps_i=pointSet[i];

		for(int j=i+1;j<numRect;j++)
		{
			RectPoint ps_j=pointSet[j];

			for(int m=0;m<4;m++)
			{
				for(int n=0;n<4;n++)
				{
					if(ps_j.p[n].x==ps_i.p[m].x||ps_j.p[n].y==ps_i.p[m].y) continue;

					float theta=0.0f;
					float k=0.0f;
					theta=((float)ps_i.p[m].y-(float)ps_j.p[n].y)/((float)ps_i.p[m].x-(float)ps_j.p[n].x);
					k=float(ps_i.p[m].x*ps_j.p[n].y-ps_i.p[m].y*ps_j.p[n].x)/float(ps_i.p[m].x-ps_j.p[n].x);

					Line l;
					l.theta=theta;
					l.k=k;
					l.endp1=i;
					l.endp2=j;
					//if (fabs(theta-fit.getSlope()) <0.5)
					if (l.theta*fit.getSlope()>0)
					{
						linePara.push_back(l);
					}
				}
			}
		}
	}

	int maxCount=0;
	size_t linesize=linePara.size();
	size_t rectsize=pointSet.size();

	for(int l=0;l<linesize;l++)
	{
		int count=2;
		for(int r=0;r<rectsize;r++)
		{
			if(r==linePara[l].endp1||r==linePara[l].endp2)continue;

			if(ifIntersection(pointSet[r],linePara[l].theta,linePara[l].k))
			{
				count++;
			}
		}

		if(count>maxCount)
			maxCount=count;
	}
	linePara.clear();
	pointSet.clear();
	fclose(file);
	printf("result=%d\n",maxCount);
	system("pause");
#else
	vector<RectPoint> pointSet;
	for (int r = 0; r<numRect; r++)
	{
		cv::Rect tmp;
		fscanf(file, "%d %d %d %d\n", &tmp.x, &tmp.y, &tmp.width, &tmp.height);

		RectPoint ps;
		ps.p[0] = cv::Point(tmp.x, tmp.y);
		ps.p[1] = cv::Point(tmp.x + tmp.width, tmp.y);
		ps.p[2] = cv::Point(tmp.x + tmp.width, tmp.y + tmp.height);
		ps.p[3]=cv::Point(tmp.x,tmp.y+tmp.height);
		pointSet.push_back(ps);
	}
    vector<Line> linePara;
    
    for(int i=0;i<numRect;i++)
    {
        RectPoint ps_i=pointSet[i];
        
        for(int j=i+1;j<numRect;j++)
        {
            RectPoint ps_j=pointSet[j];
          
            for(int m=0;m<4;m++)
            {
                for(int n=0;n<4;n++)
                {
                    if(ps_j.p[n].x==ps_i.p[m].x||ps_j.p[n].y==ps_i.p[m].y) continue;
                    
                    float theta=0.0f;
                    float k=0.0f;
                    theta=((float)ps_i.p[m].y-(float)ps_j.p[n].y)/((float)ps_i.p[m].x-(float)ps_j.p[n].x);
                    k=float(ps_i.p[m].x*ps_j.p[n].y-ps_i.p[m].y*ps_j.p[n].x)/float(ps_i.p[m].x-ps_j.p[n].x);
                  
                    Line l;
                    l.theta=theta;
                    l.k=k;
                    l.endp1=i;
                    l.endp2=j;
                    linePara.push_back(l);
                }
            }
        }
    }
 
    int maxCount=0;
    size_t linesize=linePara.size();
    size_t rectsize=pointSet.size();
    
    for(int l=0;l<linesize;l++)
    {
        int count=2;
        for(int r=0;r<rectsize;r++)
        {
            if(r==linePara[l].endp1||r==linePara[l].endp2)continue;
    
            if(ifIntersection(pointSet[r],linePara[l].theta,linePara[l].k))
            {
                count++;
            }
        }

       if(count>maxCount)
           maxCount=count;
    }
    linePara.clear();
    pointSet.clear();
    fclose(file);
    printf("result=%d\n",maxCount);
#endif
    return 1;
}


