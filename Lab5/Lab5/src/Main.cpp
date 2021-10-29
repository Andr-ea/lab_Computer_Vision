#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/utils/filesystem.hpp"
#include <opencv2/calib3d.hpp>
#include "PanoramicImage.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    
    string imagesPath, extention;
    int FoV;
	
    cout << "Insert the folder path of images: " ;
    cin >> imagesPath;
    cout << endl;
    
    cout << "Insert the file extention: ";
    cin >> extention;
    cout << endl;
   
    cout << "Insert the half of Field of View: " ;
    cin >> FoV;
    cout << endl;
    
	PanoramicImage panoramic_img(imagesPath, extention, FoV);
	Mat my_panoramic_img = panoramic_img.getPanoramicImage(10);
	imshow("Panoramic image", my_panoramic_img);

	waitKey();
	return 0;

}



