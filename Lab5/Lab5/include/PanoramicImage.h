#include <memory>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>

using namespace std;
using namespace cv;

class PanoramicImage{

public:

	PanoramicImage(string Path, string extension, int fieldOfView);

	Mat getPanoramicImage(double threshold);	

	
protected:

	void computeProjection(Mat i1, Mat i2, double r, vector<double>* a_t_x, vector<double>* a_t_y);

	vector<int> findMinMax(vector<double> re);

	// Data
	// path
	string path_file;
    string extension;
	
	//Field of View
	int FoV;

	// Original images
	vector<Mat> inputImgs;

	// Cylindrical images
	vector<Mat> cylImgs;



};
