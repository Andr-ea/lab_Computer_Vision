
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

// Defining the dimensions of checkerboard
Size patternSize(6,5);
float SIZE_CELL = 0.11;

/*
//Auxiliary method for print Rotation and Transaction vector
void printVEC_MAT(std::vector<Mat> vec){
    for (size_t i = 0; i < vec.size() ; i++) {
        std::cout <<vec[i] << '-';
    }
}
*/
int main(int argc, char **argv){
    
    // Creating vector for 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f> > objpoints;
    // Creating vector for 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f> > imgpoints;
    
    // Compute the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for(int i=0; i<patternSize.height; i++){
        for(int j=0; j<patternSize.width; j++){
            objp.push_back(cv::Point3f(i*SIZE_CELL,j*SIZE_CELL,0.0f));
        }
    }
    
    // Extracting path of individual image stored in a given directory
    std::vector<cv::String> images;
    
    // Path of the folder containing checkerboard images
    std::string path;
    std::cout << "Insert path folder of images: ";
    std::getline(std::cin,path);
    
    cv::utils::fs::glob(path,"00**_color.png", images);
    
    cv::Mat frame;
    // vector to store the pixel coordinates of detected checkerboard corners
    std::vector<cv::Point2f> corner_pts;
    bool found;
    
    for(int i=0; i < images.size(); i++){
        frame = cv::imread(images[i]);
        
        found = cv::findChessboardCorners(frame, patternSize, corner_pts, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE );
        
        if(found){
            
            //cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
            // refining pixel coordinates for given 2d points.
            // cv::cornerSubPix(frame,corner_pts,cv::Size(724,964), cv::Size(-1,-1),criteria);
            
            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(frame,patternSize , corner_pts, found);
            
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
        //To visualize detected corners
        // cv::imshow("Image",frame);
        // cv::waitKey(0);
    }
    
    cv::destroyAllWindows();
    
    cv::Mat cameraMatrix = Mat(3, 3, CV_32FC1);
    cv::Mat distCoeffs;
    std::vector<Mat> R;
    std::vector<Mat> T;
    
    // Compute cameraMatrix, distCoeffs, Rotation vector and Transaction vector (Intrinsic and Extrinsic parameters )
    cv::calibrateCamera(objpoints, imgpoints, patternSize, cameraMatrix, distCoeffs, R, T);
    
    /*
     //To visualize cameraMatrix, distCoeffs, Rotation vector and Transaction vector
     std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
     std::cout << "_____________________________________________________________ : " << std::endl;
     std::cout << "distCoeffs : " << distCoeffs << std::endl;
     std::cout << "_____________________________________________________________ : " << std::endl;
     std::cout << "Rotation vector : "; printVEC_MAT(R); std::cout <<  std::endl;
     std::cout << "_____________________________________________________________ : " << std::endl;
     std::cout << "Transaction vector : "; printVEC_MAT(T);std::cout <<  std::endl;
     
     std::cout << "_____________________________________________________________ : " << std::endl;
     */
    
    std::cout << "Intrinsic Parameters: " << std::endl;
    std::cout << "Fx: " << cameraMatrix.at<double>(0,0) << std::endl;
    std::cout << "Fy: " << cameraMatrix.at<double>(1,1) << std::endl;
    std::cout << "Uc: " << cameraMatrix.at<double>(0,2) << std::endl;
    std::cout << "Vc: " << cameraMatrix.at<double>(1,2) << std::endl;
    std::cout << "_________________________________________________: " << std::endl;
    
    //Variable for compute reprojection Error
    std::vector<cv::Point2f> projectedPoints;
    std::vector<float> perViewError;
    perViewError.resize(objpoints.size());
    
    double error, totalErr=0;
    double totalPoints = 0;
    double maxError = 0;
    int i_Max = 0;
    int i_Min = 0;
    double minError = UINT_MAX;
    
    for( int i = 0; i < objpoints.size(); i++ ){
        
        cv::projectPoints(Mat(objpoints[i]) , R[i], T[i], cameraMatrix,  distCoeffs, projectedPoints);
        
        error = norm(Mat(imgpoints[i]), Mat(projectedPoints), NORM_L2);
        int n = static_cast<int>(objpoints.size());
        perViewError[i] = (float) std::sqrt(error*error/n);
        
        if( perViewError[i] > maxError ){
            maxError = perViewError[i];
            i_Max = i;
        }
        else if( perViewError[i] <= minError ){
            minError = perViewError[i];
            i_Min = i;
        }
        
        totalErr += error*error;
        totalPoints += n;
        
    }
    
    std::cout << "Best Error : " <<  minError << " of Image[" <<i_Min<<"]" << std::endl;
    std::cout << "_________________________________________________: " << std::endl;
    std::cout << "Worst Error : " <<  maxError << " of Image[" <<i_Max<<"]" << std::endl;
    std::cout << "_________________________________________________: " << std::endl;
    std::cout << "Mean Reprojection error : " <<  std::sqrt(totalErr/totalPoints) << std::endl;
    std::cout << "_________________________________________________: " << std::endl;
    
    
    //Path of test image to undistort
    std::string path2;
    std::cout << "Insert path of test image: ";
    std::getline(std::cin,path2);
    
    cv::Mat image;
    cv::Mat image0 = cv::imread(path2);
    
    // Build the undistort map
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, cv::Size(image0.rows, image0.cols) , 0);

    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, cv::Size(image0.cols, image0.rows), CV_32FC1, map1, map2);
    
    cv::remap(image0, image, map1, map2, cv::INTER_LINEAR);
    cv::imshow("Original", image0);
    cv::imshow("Undistorted", image);
    cv::waitKey(0);
    return 0;
}
