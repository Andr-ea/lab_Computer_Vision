#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdlib.h>

using namespace cv;

// Global variables
cv::Mat src, src_gray;
cv::Mat dst, detected_edges, dst_gray, standard_hough, circle_hough;

//Variable for Canny
int edgeThresh = 1;
int lowThreshold = 100;
int const max_lowThreshold = 200;
int ratio;
int max_ratio = 10;
int kernel_size = 3;

//Variable for HoughLine
int houghthreshold = 100;
int rho = 1;
std::vector<Vec2f> lines;
std::vector<std::vector<Point2f> > linesForIntersection;

//Variable for HpughCircle
int dp = 1;
int minDist;
int param1;
int param2 = 1, minRadius = 1, maxRadius = 1;
Point centerC;
int radiusC;

// For Callback Canny
void CannyEdgeDetector(int, void* )
{
    // Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3) );
    
    // Canny detector
    cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    
    // Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);
    
    src.copyTo( dst, detected_edges );
    imshow( "EDGE MAP ", detected_edges );
    
}

// For Callback HoughLine
void HoughLine(int, void* ){
    
     //Vector to store the lines found by HoughLines
    std::vector<Vec2f> lines;
    
    cvtColor( detected_edges, standard_hough, COLOR_GRAY2BGR );
    
    HoughLines( detected_edges, lines, rho, CV_PI/180, houghthreshold, 0, 0 );
    
    //Compute point of lines and draw it
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float r = lines[i][0], t = lines[i][1];
        double cos_t = cos(t), sin_t = sin(t);
        double x0 = r*cos_t, y0 = r*sin_t;
        double alpha = 1000;
        
        Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
        Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
        
        std::vector<Point2f> points;
        points.push_back(pt1);
        points.push_back(pt2);
        
        line( standard_hough, pt1, pt2, Scalar(255,0,0), 3, LINE_AA);
        
        // Store two lines with opposite slope in sign to find the intersection
        if( r > 0 ){
            linesForIntersection.push_back(points);
        }
        if( r < 0 ){
            linesForIntersection.push_back(points);
        }
    }
    
    imshow( "HOUGH MAP ", standard_hough );
    
}

// For Callback HoughCircle
void HoughCircle(int, void* ){
    
    cvtColor( detected_edges, circle_hough, COLOR_GRAY2BGR );
    
    //Vector to store the circles found by HoughCircles
    std::vector<Vec3f> circles;
    HoughCircles(detected_edges, circles, CV_HOUGH_GRADIENT, dp, detected_edges.rows, 200, param2, minRadius, maxRadius );
    
    //Compute center and radius of circles and draw it
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        
        // Draw the circle center
        circle( circle_hough, center, 3, Scalar(0,255,0), -1, 5, 0 );
        // Draw the circle outline
        circle( circle_hough, center, radius, Scalar(0,255,0), 3, 8, 0 );
        
        centerC = center;
        radiusC = radius;
    }
    
    imshow( "HOUGH CIRCLE MAP ", circle_hough );
}

//Method for compute the intersection between two line give four point: A and B for line 1, C and D for line 2
Point2f intersection(Point2f A, Point2f B, Point2f C, Point2f D) {
   
    double a = B.y - A.y;
    double b = A.x - B.x;
    double c = a*(A.x) + b*(A.y);
    
    double a1 = D.y - C.y;
    double b1 = C.x - D.x;
    double c1 = a1*(C.x)+ b1*(C.y);
    
    double det = a*b1 - a1*b;
    if (det == 0) {
        return Point2f(FLT_MAX, FLT_MAX);
    } else {
        double x = (b1*c - b*c1)/det;
        double y = (a*c1 - a1*c)/det;
        return Point2f(x, y);
    }
}

int main(int argc, char **argv){
    
    std::string path;
    std::cout << "Insert path image: ";
    std::getline(std::cin,path);
    
    src = cv::imread(path);
    
    // Create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );
    
    // Convert the image to grayscale
    cvtColor( src, src_gray, CV_BGR2GRAY );
    
    // Create a window
    namedWindow( "EDGE MAP ", CV_WINDOW_AUTOSIZE );
    
    // Create a Trackbar for user to enter threshold
    createTrackbar( "Min Threshold:", "EDGE MAP ", &lowThreshold, max_lowThreshold, CannyEdgeDetector );
    createTrackbar( "Ratio:", "EDGE MAP ", &ratio, max_ratio, CannyEdgeDetector );
    
    // Show the image
    CannyEdgeDetector(0, 0);
    //Press key to continue
    waitKey();
    
    //Results of Canny edge detector
    std::cout << "_________________________________________________ " << std::endl;
    std::cout << "VALUES OF CANNY: " << std::endl;
    int valueMinThreshold = getTrackbarPos( "Min Threshold:", "EDGE MAP ");
    std::cout << "VALUE OF LOWTHRESHOLD: " << valueMinThreshold << std::endl;
    int valueRatio = getTrackbarPos( "Ratio:", "EDGE MAP ");
    std::cout << "VALUE OF RATIO: " << valueRatio << std::endl;
    
    cv::destroyWindow("EDGE MAP ");
    
    namedWindow( "HOUGH MAP ", CV_WINDOW_AUTOSIZE );
    createTrackbar( "Houghthreshold:", "HOUGH MAP ", &houghthreshold, 200, HoughLine );
    
    // Show the image
    HoughLine(0, 0);
    //Press key to continue
    waitKey();
    
    std::cout << "_________________________________________________ " << std::endl;
    std::cout << "VALUES OF HOUGH LINES: " << std::endl;
    int valueHoughthreshold = getTrackbarPos( "Houghthreshold:", "HOUGH MAP ");
    std::cout << "VALUE OF Hough Threshold: " << valueHoughthreshold << std::endl;
   
    
    cv::destroyWindow("HOUGH MAP ");
    
    //Vector for store the intersections points
    std::vector<cv::Point> intersections;
    
    //Interception between two lines with opposite slope in sign
    Point p = intersection( linesForIntersection[0][0],  linesForIntersection[0][1],  linesForIntersection[1][0], linesForIntersection[1][1]);
    //circle( src, p, 4, Scalar(0,255,255), 3, 8, 0 ); //For view interception point
    intersections.push_back(p);
    
    //Interception between first oblique line and the line on the bottom edge of the image
    Point p1 = intersection( linesForIntersection[0][0],  linesForIntersection[0][1], Point2f(1, src.rows-1 ), Point2f(src.cols-1, src.rows-1 ) );
    //circle( src, p1, 4, Scalar(0,255,255), 3, 8, 0 ); //For view interception point
    intersections.push_back(p1);
    
    //Interception between second oblique line and the line on the bottom edge of the image
    Point p2 = intersection( linesForIntersection[1][0], linesForIntersection[1][1] , Point2f(1, src.rows-1 ), Point2f(src.cols-1, src.rows-1 ) );
    //circle( src, p2, 4, Scalar(0,255,255), 3, 8, 0 );  //For view interception point
    intersections.push_back(p2);
    
    //Draw the polygon delimited by the intersection points
    cv::fillConvexPoly(src, intersections, Scalar(0,0,255), CV_AA);
    
    /*
    //Hough Circles
    namedWindow( "HOUGH CIRCLE MAP ", CV_WINDOW_AUTOSIZE );
    createTrackbar( "Dp:", "HOUGH CIRCLE MAP ", &dp, 20, HoughCircle );
    createTrackbar( "Param2:", "HOUGH CIRCLE MAP ", &param2, 20, HoughCircle );
    createTrackbar( "MinRadius:", "HOUGH CIRCLE MAP ", &minRadius, 20, HoughCircle );
    createTrackbar( "MaxRadius:", "HOUGH CIRCLE MAP ", &maxRadius, 20, HoughCircle );
    
    // Show the image
    HoughCircle(0, 0);
    //Press key to continue
    waitKey(0);
    
    std::cout << "_________________________________________________ " << std::endl;
    std::cout << "VALUES OF HOUGH CIRCLES: " << std::endl;
    int valueDp = getTrackbarPos( "Dp:", "HOUGH CIRCLE MAP ");
    std::cout << "VALUE OF DP: " << valueDp << std::endl;
    int valueParam2 = getTrackbarPos( "Param2:", "HOUGH CIRCLE MAP ");
    std::cout << "VALUE OF Param2: " << valueParam2 << std::endl;
    int valueMinRadius = getTrackbarPos( "MinRadius:", "HOUGH CIRCLE MAP ");
    std::cout << "VALUE OF MinRadius: " << valueMinRadius << std::endl;
    int valueMaxRadius = getTrackbarPos( "MaxRadius:", "HOUGH CIRCLE MAP ");
    std::cout << "VALUE OF MaxRadius: " << valueMaxRadius << std::endl;
    
    cv::destroyWindow("HOUGH CIRCLE MAP ");
    
    //Draw the circle of sign
    cv::circle(src, centerC, radiusC, Scalar(0,255,0), CV_FILLED);
   */
    namedWindow( "Result", 1 );
    imshow( "Result", src );
    
    //Press key to finish
    waitKey();
    
    return 0;
}

