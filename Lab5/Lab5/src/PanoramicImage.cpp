#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/features2d.hpp>
#include "opencv2/core/utils/filesystem.hpp"
#include "PanoramicImage.h"
#include "panoramic_utils.h"

using namespace cv;
using namespace std;
 
PanoramicImage::PanoramicImage(string pathFile, string extention, int fieldOfView) {
    FoV = fieldOfView;
    
    path_file = pathFile;
    extension = extention;
    vector<cv::String> fileNames;
    cv::utils::fs::glob(path_file, "i**." + extension, fileNames);
    
    //Read images
    for (int i = 0; i < fileNames.size(); i++){
        Mat img = imread(fileNames[i]);
        inputImgs.push_back(img);
    }
    
    // Projection in the cylinder
    for (int i = 0; i < inputImgs.size(); i++){
        Mat img = inputImgs[i];
        img = PanoramicUtils::cylindricalProj(img, FoV );
        cylImgs.push_back(img);
    }
}

// Compute projection
void PanoramicImage::computeProjection(Mat img1, Mat img2, double distThreshold, vector<double>* dx, vector<double>* dy){
    
    //SIFT detector
    Ptr<Feature2D> sift1 = xfeatures2d::SIFT::create();
    Ptr<Feature2D> sift2 = xfeatures2d::SIFT::create();
    vector<KeyPoint> keyPoints1, keyPoints2;
    Mat descriptor1, descriptor2;
    
    //Detection keypoints and descriptors
    sift1->detectAndCompute(img1, Mat(), keyPoints1, descriptor1);
    sift2->detectAndCompute(img2, Mat(), keyPoints2, descriptor2);
    
    BFMatcher bf(NORM_L2, true);
    vector<DMatch> matches;
    bf.match(descriptor1, descriptor2, matches, Mat());
    
    // Keep only good matches
    Mat index;
    int nMatch = int(matches.size());
    double min = matches[0].distance;
    
    for (int i = 0; i < nMatch; i++) {
        if (matches[i].distance < min)
            min = matches[i].distance;
    }
   
    vector<DMatch> bestMatches;
    for (int i = 0; i < nMatch; i++) {
        if (matches[i].distance < distThreshold * min)
            bestMatches.push_back(matches[i]);
    }
    

    vector<Point2f> destination;            //First image
    vector<Point2f> source;                //Second image
    
    for (vector<DMatch>::iterator i = bestMatches.begin(); i != bestMatches.end(); i++){
        //Get the keypoints from the good matches
        destination.push_back(keyPoints1[i->queryIdx].pt); // position of the keypoint on the query
        source.push_back(keyPoints2[i->trainIdx].pt); // position of the keypoint on the train
    }
    
    //Draw Best Match
    Mat match_img;
   // drawMatches(img1, keyPoints1, img2, keyPoints2, bestMatches, match_img, Scalar(0,0,255), Scalar(0, 0, 255),Mat(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //imshow("BestMatches", match_img);
   
    //Compute average displacement between the two images, using only good matches
    double myDx = 0, myDy = 0;
    int c = 0;
    Mat mask;
    Mat H = findHomography(source, destination, RANSAC, 5.0, mask);
    
    vector<DMatch> inliersMatches;
    for (int i = 0; i < bestMatches.size(); i++){
        if ((int)mask.at<uchar>(i)){
            myDx += source[i].x - destination[i].x + img1.cols;
            myDy += source[i].y - destination[i].y;
            c++;
            inliersMatches.push_back(bestMatches[i]);
        }
    }
    
    //Draw inliers Match
  // drawMatches(img1, keyPoints1, img2, keyPoints2, inliersMatches, match_img, Scalar(0, 255, 0), Scalar(0, 0, 255), Mat(), DrawMatchesFlags::DRAW_OVER_OUTIMG);
   // imshow("Inliers", match_img);
    
    dx->push_back(myDx / c);
    dy->push_back(myDy / c);
    
}

// Find the minimum and the maximum distances of the matches
vector<int> PanoramicImage::findMinMax(vector<double> distances){
    double max = 0;
    double min = 0;
    for (int i = 1; i < distances.size(); i++){
        if (max < distances[i]) {  max = distances[i]; }
        if (min > distances[i]) {  min = distances[i]; }
    }
    
    vector<int> minMax;
    minMax.push_back((int)min);
    minMax.push_back((int)max);
    return minMax;
}

Mat PanoramicImage::getPanoramicImage(double threshold){
    
    vector<double> dx, dy; // Translation dx dy between two images
    vector<Mat> Hv;
    for (int i = 0; i < cylImgs.size()-1; i++) {
        Mat img1 = cylImgs[i].clone();
        Mat img2 = cylImgs[i+1].clone();
        PanoramicImage::computeProjection(img1, img2, threshold, &dx, &dy);
       // waitKey();
    }
    
    double s_x = 0;
    for (int k = 0; k < dx.size(); k++) {
        s_x += dx[k];
    }
    int width = cylImgs[0].cols;
    int height = cylImgs[0].rows;
    vector<int> minMax = findMinMax(dy);
    
    // Panoramic image
    Mat panoramicImg = Mat::zeros(Size((cylImgs.size()+1)*width-s_x, (int)(height + dy.size()*abs(minMax[0]) +dy.size()*abs(minMax[1]))), CV_8UC1);
    int s_p_x = 0;
    int s_y = 0;
    int maxX = 0;
    int minY = (int)dy.size()*abs(minMax[0]);
    int maxY = (int)dy.size()*abs(minMax[0])+height;
    
    cylImgs[0].copyTo(panoramicImg(Rect(0, (int)dy.size()*abs(minMax[0]), width, height))); // Insert each image in the panoramic image
    for (int i = 1; i <= dx.size(); i++) {
       
        Mat img_t = cylImgs[i].clone();
        equalizeHist(img_t, img_t);  // Equalization
        s_p_x += dx[i-1];
        
        int y = dy.size() * abs(minMax[0])+(dy[i-1]+s_y);
        s_y = dy[i-1];
        img_t.copyTo(panoramicImg(Rect(i*width-s_p_x, y, width, height)));
        
        if (y + height > maxY){
            maxY = y + height;
        }
        if (y < minY){
            minY = y;
        }
        maxX = (i + 1) * width - s_p_x;
    }
    panoramicImg = panoramicImg(Rect(0, minY, maxX, maxY-minY));
   
    resize(panoramicImg, panoramicImg, Size(panoramicImg.cols, panoramicImg.rows));
    return panoramicImg;
}
