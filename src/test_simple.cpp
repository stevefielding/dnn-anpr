/* Test code to help figure out how to read from video file, 
and how to to convert dlib image format to from OpenCV image format
*/
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

using namespace std;
using namespace dlib;


// ----------------------------------------------------------------------------------------

int main(int argc, char **argv) try
{
    if (argc != 2) {
      cout << "Uage: dnn_mmod_find_cars_ex  mmod_cars_test_image.jpg\n";
      return -1;
    }

    // Load video file
    cv::VideoCapture cap(argv[1]);
    //cv::VideoCapture cap;
    //cap.open(argv[2]);
    //cap.open(0);
    if (!cap.isOpened())
    {
        cerr << "Unable to open video file: " << argv[2] << endl;
        return 1;
    }

    image_window win2;
    rectangle rect1, rect2;
    cv::RNG& rng = cv::theRNG();
    while(!win2.is_closed()) {

      cv::Mat temp;
      if (!cap.read(temp))
      {
          break;
      }
      cv::Mat clone = temp.clone();
      cv_image<bgr_pixel> image(temp);
      matrix<rgb_pixel> cimg;
      assign_image(cimg, image);
      
      rect1 = rectangle(rng.uniform(10,100),rng.uniform(10,100),rng.uniform(101,500),rng.uniform(101,500));
      auto myUnion = (rect1 + rect2).area();
      auto intersect = (rect1.intersect(rect2)).area();
      float iou = (float) intersect / myUnion;
      cout << "iou: " << iou;
      rect2 = rect1;
      matrix<rgb_pixel> roi;
      roi = subm(cimg, rect2);
      win2.set_image(roi);
      cv::Mat car;
      car = clone(cv::Rect(rect1.left(), rect1.top(), rect1.width(), rect1.height()));
      cv::imshow("Car",car);
      cv::waitKey(0);
      cout<<"Hit any key";
      //int ch = std::cin.get();
      //string strInput=getch();
    }

    cout << "Hit enter to end program" << endl;
    cin.get();
}
catch(image_load_error& e)
{
    cout << e.what() << endl;
    cout << "The test image is located in the examples folder.  So you should run this program from a sub folder so that the relative path is correct." << endl;
}
catch(serialization_error& e)
{
    cout << e.what() << endl;
    cout << "The correct model file can be obtained from: http://dlib.net/files/mmod_rear_end_vehicle_detector.dat.bz2" << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




