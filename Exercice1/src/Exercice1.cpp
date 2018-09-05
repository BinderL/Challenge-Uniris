#include <typeinfo>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// static void help()
// {
//     cout << "\nThis program demonstrates unsupervised kmeans clustering for find circle on image .\n"
//             "input : an image with circle, then find cluster by considering the pixel position \n
//             "Optimised the number of cluster with a local criteria\n"
//			   "and draw the number of circle find in the last steps\n"
//			   "output : Images with draw circle"
//             "Call\n"
//             "./Exercice1\n" << endl;
// }

int main( int argc, char** argv )
{
	int e = 3; //3
	int clusterMax = 15; // optim rank
	Point ipt; double dx, dy; // global variable used in the 2 main loop
	Mat image = imread("/home/binderl/eclipseworkspace/Exercice1/src/opencv2.png", IMREAD_GRAYSCALE);
	uint8_t* pixelPtr = (uint8_t*)image.data;
	int cn = image.channels();
	cout <<"Résolution image : " << image.rows <<"*"<< image.cols << endl;
	Mat points(46881, 2, CV_32F); // for database storage
	imshow("image", image); //display input image
    Scalar colorTab[] = // color displaying
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255),
		Scalar(255,0,0),
		Scalar(255,255,255),
		Scalar(100,100,100),
		Scalar(255,100,255)
    };
    int pointCount = 0;
    for (int i = 0; i < image.rows; ++i) // build database
    {
    	for (int j = 0; j < image.cols; ++j)
    	{
    		if(pixelPtr[i*image.cols*cn + j*cn + 0] != 255)
    		{
    			points.at<float>(pointCount,0) = j;
    			points.at<float>(pointCount,1) = i;
    			pointCount++;
    		}
    	}
    }
    cout <<"DataBase sample = "<< pointCount << endl;
    cout << "DataBase built" << endl;
    Mat img(image.rows, image.cols, CV_8UC3);
    for(;;)
    {
		Mat labels; // to store database label
    	int ncircle = 0;
		int n_optim = 0; //optim result
		Point sum =  Point(0, 0);
		int vmax =  0;
		int clusterSample = 0;
		int sampleCount = pointCount;
		double vmaxf = 0.0;
		Point2f iptmax;
    	for(int n = 2; n < clusterMax; n++) //optim loop : find right number of cluster.
    	{
			int clusterCount = n;
			Point2f pointClusterTab[clusterCount];
			Point2f pointMaxCoordTab [clusterCount];
			double distTab [clusterCount];
			Point2f iptmaxTab [clusterCount];
			int i;
	        kmeans(points, clusterCount, labels,
	            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
	               3, KMEANS_PP_CENTERS);

			for(int k = 0; k < clusterCount; k++) //cluster loop
			{
				sum = Point(0,0);
				clusterSample = 0;
				for( i = 0; i < sampleCount; i++ ) //find cluster centroid position
				{
					int clusterIdx = labels.at<int>(i);
					if(clusterIdx == k)
					{
						ipt = points.at<Point2f>(i);
						sum.x = sum.x + ipt.x;
						sum.y = sum.y + ipt.y;
						clusterSample++;
						vmax = std::max(vmax, ipt.y);
					}
				}
				Point2f average = sum / clusterSample;
				pointClusterTab[k] = average;
				Point2f idMax = Point2f(average.x,vmax);
				pointMaxCoordTab[k] = idMax;
				double sum = 0;
				clusterSample = 0;
				for (int j=0;j < sampleCount; j++) // compute average distance between points and cluster centroid
				{
					if(labels.at<int>(j) == k)
					{
						ipt = points.at<Point2f>(j);
						dx = ipt.x - pointClusterTab[k].x;
						dy = ipt.y - pointClusterTab[k].y;
						double dist2 = dx * dx + dy * dy;
						double dist = std::sqrt(dist2);
						sum = sum + dist;
						clusterSample++;
					}
				}
				double averageDist = sum/clusterSample;
				for (int j=0;j < sampleCount; j++) // compute metrics :  on clustering set linearized with threshold e
				{
					sum = 0;
					if(labels.at<int>(j) == k)
					{
						ipt = points.at<Point2f>(j);
						dx = ipt.x - pointClusterTab[k].x;
						dy = ipt.y - pointClusterTab[k].y;
						double dist2 = dx * dx + dy * dy;
						double dist = std::sqrt(dist2);
						sum = sum + dist;
						if(dist > vmaxf && dist < e * averageDist )
						{
							iptmax = ipt;
							vmaxf = dist;
						}
						clusterSample++;
					}
				}

				iptmaxTab[k] = iptmax;
				distTab[k] = vmaxf;
				vmaxf = 0;

			}
			int critere = 0;
			for(int k = 0; k < clusterCount; k++) // compute criteria to find the right number of cluster
			{
				for(int j = k+1; j < clusterCount; j++)
				{
					dx = pointClusterTab[k].x - pointClusterTab[j].x;
					dy = pointClusterTab[k].y - pointClusterTab[j].y;
					double distCluster = std::sqrt( dx * dx + dy * dy );
					double sumDist = distTab[j] + distTab[k];
					//cout <<"n"<<n<<"dx"<<dx<<"dy"<<dy<<"pointClusterTab[k]"<<pointClusterTab[k]<<"pointClusterTab[j]"<<pointClusterTab[j]<< "distCluster" << distCluster <<"sumDist"<<sumDist<<"distTab[j]"<<distTab[j]<<"distTab[k]"<<distTab[k]<<"iptmaxTab[k]"<<iptmaxTab[k]<<"iptmaxTab[j]"<<iptmaxTab[j]<<endl;
					if(distCluster > sumDist )
					{
						critere++;
					}
				}
			}

			if (critere == (n*n-n)/2)
			{
				critere = 0;
				n_optim = n;
			}
    	}
    	cout<<"optimisation result : number of cluster = "<<n_optim<<endl;
    	cout<<"begin visualisation"<<endl;
		kmeans(points, n_optim, labels,
			TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
			   3, KMEANS_PP_CENTERS);
		for(int k = 0; k < n_optim; k++)
		{
			sum = Point(0,0);
			clusterSample = 0;
			for(int i = 0; i < sampleCount; i++ ) //find cluster centroid position for special case n = nmax
			{
				int clusterIdx = labels.at<int>(i);
				if(clusterIdx == k)
				{
					ipt = points.at<Point2f>(i);
					sum.x = sum.x + ipt.x;
					sum.y = sum.y + ipt.y;
					clusterSample++;
					vmax = std::max(vmax, ipt.y);
				}
			}
			Point2f average = sum / clusterSample;
			double sum = 0;
			for (int j=0;j < sampleCount; j++) // compute average distance between points and cluster centroid for special case
			{
				if(labels.at<int>(j) == k)
				{
					ipt = points.at<Point2f>(j);
					dx = ipt.x - average.x;
					dy = ipt.y - average.y;
					double dist2 = dx * dx + dy * dy;
					double dist = std::sqrt(dist2);
					sum = sum + dist;
					clusterSample++;
				}
			}
			double averageDist = sum/clusterSample;
			vmaxf = 0;
			for (int j=0;j < sampleCount; j++) // compute metrics :  on clustering set linearized (radius)
			{
				sum = 0;
				if(labels.at<int>(j) == k)
				{
					ipt = points.at<Point2f>(j);
					dx = ipt.x - average.x;
					dy = ipt.y - average.y;
					double dist2 = dx * dx + dy * dy;
					double dist = std::sqrt(dist2);
					sum = sum + dist;
					if(dist > vmaxf && dist < e * averageDist )
					{
						iptmax = ipt;
						vmaxf = dist;
					}
					clusterSample++;
				}
			}
			circle(img, average, vmaxf, colorTab[k], FILLED, LINE_AA ); // draw circle
			ncircle++;
			cout<<"circle and cluster number : "<<k<<" radius = "<<vmaxf<<endl;
			line(img, average, iptmax,Scalar(0, 0, 255)); //draw radius
			circle(img, average, 3, Scalar(255, 255, 255), FILLED, LINE_AA ); // draw cluster position
			//
		}
        imshow("clusters", img); //display output image
        cout<<"Number of circle draw "<<ncircle<<endl;
        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }
    return 0;
}


