//
//  main.cpp
//  OpenCVTest
//
//  Created by Andrea Oriolo on 25/03/2020.
//  Copyright © 2020 Andrea Oriolo. All rights reserved.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//#include <opencv2/opencv.hpp>

#define NEIGHBORHOOD_Y 9
#define NEIGHBORHOOD_X 9

double  MAX_B_CHANNEL = 0;
double  MAX_G_CHANNEL = 0;
double  MAX_R_CHANNEL = 0;

double  MIN_B_CHANNEL = 255;
double  MIN_G_CHANNEL = 255;
double  MIN_R_CHANNEL = 255;

std::vector<double> minTh = {0,0,0};
std::vector<double> maxTh = {0,0,0};
int cont = 0;

using namespace cv;

void printMinThreshold(std::vector <double> const &a) {
   std::cout << "MIN Threshold: : ";
   
   for(int i=0; i < a.size(); i++)
      std::cout << a.at(i) << ' ';
    std::cout << ' ' << std::endl;
}
void printMaxThreshold(std::vector <double> const &a) {
   std::cout << "MAX Threshold: ";
   
   for(int i=0; i < a.size(); i++)
      std::cout << a.at(i) << ' ';
    std::cout << ' ' << std::endl;
}

Mat createNew (cv::Mat input_img ){
    
    Mat output_img (input_img.rows, input_img.cols, CV_8UC3);
    cv::resize(input_img, output_img, cv::Size(input_img.cols, input_img.rows ));
    
    for ( int i = 0; i < input_img.rows; i++){
        for( int j = 0; j < input_img.cols; j++){
            Vec3b bgrPixel = input_img.at<Vec3b>(i, j);
            
            if( bgrPixel[0] >= MIN_B_CHANNEL && bgrPixel[0] <= MAX_B_CHANNEL &&
                bgrPixel[1] >= MIN_G_CHANNEL && bgrPixel[1] <= MAX_G_CHANNEL &&
                bgrPixel[2] >= MIN_R_CHANNEL && bgrPixel[2] <= MAX_R_CHANNEL
               )
                output_img.at<Vec3b>(i, j)[0] = 201;
                output_img.at<Vec3b>(i, j)[1] = 37;
                output_img.at<Vec3b>(i, j)[1] = 93;
        }
    }
    
    return output_img;
    
}

void onMouse(int event, int x, int y, int f, void* userdata){
    //If the left button is pressed
    if( event == cv::EVENT_LBUTTONDOWN ){
        
        //Retrieving the image from the main
        cv::Mat image = *(cv::Mat*) userdata;
        
       // cv::Mat image_out = image.clone();
       // Mat output_img (image.rows, image.cols, CV_8UC3);
        //cv::resize(output_img, image, cv::Size(image.cols /2.0, image.rows /2.0 ));
        
       
        
        //Preventing segfaults for looking over the image boundaries
        if( y + NEIGHBORHOOD_Y > image.rows ||
            x + NEIGHBORHOOD_X > image.cols     )
            return;
        
        //Mean on the neighborhood
        cv:Rect rect(x,y, NEIGHBORHOOD_X, NEIGHBORHOOD_Y);
        cv::Scalar mean = cv::mean(image(rect));
        
        std::cout << ' ' << std::endl;
        std::cout << "Mean: " << mean << std::endl;
        
        
        //Aggiorno MIN E MAX
        if( mean[0] < MIN_B_CHANNEL ){
            MIN_B_CHANNEL = mean[0];
            minTh[0] = mean[0];
        }
        else if( mean[0] > MAX_B_CHANNEL ){
            MAX_B_CHANNEL = mean[0];
            maxTh[0] = mean[0];
        }
        
        if( mean[1] < MIN_G_CHANNEL ){
            MIN_G_CHANNEL = mean[1];
            minTh[1] = mean[1];
            
        }
        else if( mean[1] > MAX_G_CHANNEL ){
            MAX_G_CHANNEL = mean[1];
            maxTh[1] = mean[1];
        }
        
        if( mean[2] < MIN_R_CHANNEL ){
            MIN_R_CHANNEL = mean[2];
            minTh[2] = mean[2];
            
        }
        else if( mean[2] > MAX_R_CHANNEL ){
            MAX_R_CHANNEL = mean[2];
            maxTh[2] = mean[2];
        }
    
        printMinThreshold(minTh);
        printMaxThreshold(maxTh);
        
        if( cont > 3 ){
        Mat output_img = createNew(image);
        cv::imshow( "img_out" , output_img);
        }
        cont++;
        
    }
}





int main(int argc, char **argv)
{
       
    cv::Mat input_img = cv::imread("/Users/andreaoriolo/Desktop/Lab1/Data/robocup.jpg");
    cv::resize(input_img, input_img, cv::Size(input_img.cols /2.0, input_img.rows /2.0 ));
    cv::imshow("img", input_img);
    cv::setMouseCallback("img", onMouse, (void*)&input_img);
    
    
  //  Mat output_img (input_img.rows, input_img.cols, CV_8UC3);
   // cv::resize(output_img, input_img, cv::Size(input_img.cols /2.0, input_img.rows /2.0 ));
    //cv::imshow("img_out", output_img);
   
    cv::waitKey(0);
    return 0;
}


