 
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include <opencv2/calib3d.hpp>
#include <string>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

// Computes the starting keypoints and starting corners
vector<vector<vector<Point2f> > >  computeInitialState(vector<Mat> video, vector<Mat> objects, double distThreshold){
    
    vector<vector<Point2f> > points;
    vector<vector<Point2f> > corners;
    
    //SIFT Detector
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
    vector<KeyPoint> keyPoint1, keyPoint2;
    Mat descript1, descript2;
    
    Mat matches_img = video[0];
    
    for(int i = 0; i < objects.size(); i++){
        cv::Mat obj_img = objects[i];
        cv::Mat scene_img = video[0];

        detector->detectAndCompute(obj_img, noArray(), keyPoint1, descript1);
        detector->detectAndCompute(scene_img, noArray(), keyPoint2, descript2);
        
        BFMatcher bf(NORM_L2, false); 
        vector<DMatch> matches;

        bf.match(descript1, descript2, matches);
        
        //Keep best matches
        vector<DMatch> bestMatches;
        
        Mat index;
        int nbMatch = int(matches.size());
        double minim = matches[0].distance;
        for (int i = 0; i < nbMatch; i++) {
            if (matches[i].distance < minim)
                minim = matches[i].distance;
        }
        for (int i = 0; i < nbMatch; i++) {
            if (matches[i].distance< distThreshold*minim)
                bestMatches.push_back(matches[i]);
        }
    
        // Localization of the object
        vector<Point2f> src_points, dst_points;
        for(size_t i = 0; i < bestMatches.size(); i++){
            dst_points.push_back(keyPoint1[bestMatches[i].queryIdx].pt);
            src_points.push_back(keyPoint2[bestMatches[i].trainIdx].pt);
        }
            
        vector<KeyPoint> keyPoints_best_temp;
        
        for (vector<DMatch>::iterator it = bestMatches.begin(); it != bestMatches.end(); ++it) {
            keyPoints_best_temp.push_back(keyPoint2[it->trainIdx]);
        }
        
        Mat mask;
        Mat h = findHomography(dst_points, src_points, RANSAC, 5, mask);
        
        vector<Point2f> scene_pts;
        vector<DMatch> inliers_matches;
        for (int i = 0; i < bestMatches.size(); i++){
            if ((int)mask.at<uchar>(i)){
                inliers_matches.push_back(bestMatches[i]);
                scene_pts.push_back(keyPoints_best_temp[i].pt);
            }
        }
        
        points.push_back(scene_pts);
    
        //Get the corners from the image1
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f( (float)obj_img.cols, 0 );
        obj_corners[2] = Point2f( (float)obj_img.cols, (float)obj_img.rows );
        obj_corners[3] = Point2f( 0, (float)obj_img.rows );
        std::vector<Point2f> scene_corns(4);
        perspectiveTransform( obj_corners, scene_corns, h);
        
        corners.push_back(scene_corns);
        
    }
    vector<vector<vector<Point2f> > > initialSate;
    initialSate.push_back(points);
    initialSate.push_back(corners);
    
    return initialSate;
}

void drawRect(Mat img, vector<vector<Point2f> > corners){
    
    Mat rect = img.clone();
    
    for (int i = 0; i < corners.size(); i++) {
        int b, g, r;
        if(i == 0){
            b = 0; g = 0; r = 255; //red
        }
        else if (i == 1){
            b = 0; g = 255; r = 0; // green
        }
        else if (i == 2){
            b = 255; g = 0; r = 0; // blue
        }
        else{
            b = 255; g = 200; r = 0; //light blue
        }
        
        // Draw lines between the corners 
        line(rect, corners[i][0], corners[i][1], Scalar(b, g, r), 4);
        line(rect, corners[i][1], corners[i][2], Scalar(b, g, r), 4);
        line(rect, corners[i][2], corners[i][3], Scalar(b, g, r), 4);
        line(rect, corners[i][3], corners[i][0], Scalar(b, g, r), 4);
    
    }
    imshow("Object Detection", rect);
    cv::waitKey(10);
    
}

int main(){
    
    string video_path, obj_path;
    
    cout << "Insert video path: ";
    cin >> video_path;
    cout << "Insert objects images folder path: ";
    cin >> obj_path;
    
    VideoCapture cap(video_path);
    vector<Mat> video_frames, objects;
    
    if(cap.isOpened())
        for(;;){
            cv::Mat frame;
            cap >> frame;
            
            if(frame.empty()==1){  break; }
           
            video_frames.push_back(frame);
        }
    
    vector<cv::String> fileNames;
    cv::utils::fs::glob(obj_path, "i*.png" , fileNames);
    for(int i = 0; i < fileNames.size(); i++){
        Mat current_img = imread(fileNames[i]);
        objects.push_back(current_img);
    }
    
    // Locate objects into the first image of the video
    vector<vector<vector<Point2f> > > tmp_a = computeInitialState(video_frames, objects, 4);
    
    vector<vector<Point2f> > startPoints = tmp_a[0];
    vector<vector<Point2f> > startCorners = tmp_a[1];
    vector<unsigned char> status;
    
    for (int i = 1; i < video_frames.size(); i++) {
    
        vector<vector<Point2f> > frameCorners;
        for (int k = 0; k < objects.size(); k++) {
            vector<Point2f> endPoints, sp, ep;
            vector<Point2f> endCorners;
            cv::calcOpticalFlowPyrLK(video_frames.at(i - 1), video_frames.at(i), startPoints.at(k), endPoints, status, noArray(), Size(13, 13));
            
            //Keep inliers
            for (int j = 0; j < status.size(); j++)
                if (status.at(j) == 1) {
                    sp.push_back(startPoints.at(k).at(j));
                    ep.push_back(endPoints.at(j));
                }
            startPoints.at(k) = ep;
            
            //Compute the translation
            Mat homography = findHomography(sp, ep, RANSAC, 5);
            perspectiveTransform(startCorners.at(k), endCorners, homography);
            startCorners.at(k) = endCorners;
            frameCorners.push_back(endCorners);
        }
        Mat img_rect = video_frames.at(i).clone();
        drawRect(img_rect, frameCorners);
        
    }
    return 0;
}
