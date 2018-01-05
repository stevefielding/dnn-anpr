// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    Based on examples/dnn_mmod_find_cars_ex.cpp
    Modified to read video file, and save the cropped cars to file
*/
// Usage: gdb --args ./dnn_mmod_find_cars_save_to_file ../datasets/mmod_rear_end_vehicle_detector.dat ../video/101634AA.MP4



#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

using namespace std;
using namespace dlib;



// The rear view vehicle detector network
// con5d num_filter convolutions, 5x5 filter size, 2x2 stride
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
// con5 num_filter convolutions, 5x5 filter size, 1x1 stride
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
// I believe that the affine is a replacement for the batch normalization layers used during training.
// I think affine simply adjusts the levels of the signal by applying the average mean and stndard deviation as corrections.
// Activations leaving a batch normalization layer will have
// approximately zero mean and unit variance (i.e., zero-centered).
// See Starter Bundle 11.2.6
template <typename SUBNET> using downsampler  = relu<affine<con5d<32,
                                                relu<affine<con5d<32, 
                                                relu<affine<con5d<16,
                                                SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<55,SUBNET>>>;
using net_type = loss_mmod<
                 con<1,9,9,1,1,
                 rcon5<
                 rcon5<
                 rcon5<
                 downsampler<
                 input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char **argv) try
{
    if (argc != 3) {
      cout << "Uage: dnn_mmod_find_cars_ex mmod_rear_end_vehicle_detector.dat mmod_cars_test_image.jpg\n";
      return -1;
    }

    // Load video file
    cv::VideoCapture cap(argv[2]);
    //cv::VideoCapture cap;
    //cap.open(argv[2]);
    //cap.open(0);
    if (!cap.isOpened())
    {
        cerr << "Unable to open video file: " << argv[2] << endl;
        return 1;
    }

#ifdef SIMPLE_VIDEO_PLAY
    image_window win2;
    while(!win2.is_closed()) {

      cv::Mat temp;
      if (!cap.read(temp))
      {
          break;
      }
      cv_image<bgr_pixel> image(temp);
      matrix<rgb_pixel> cimg;
      assign_image(cimg, image);
      win2.set_image(cimg);
    }
#endif

    net_type net;
    shape_predictor sp;
    // You can get this file from http://dlib.net/files/mmod_rear_end_vehicle_detector.dat.bz2
    // This network was produced by the dnn_mmod_train_find_cars_ex.cpp example program.
    // As you can see, the file also includes a separately trained shape_predictor.  To see
    // a generic example of how to train those refer to train_shape_predictor_ex.cpp.
    //deserialize("mmod_rear_end_vehicle_detector.dat") >> net >> sp;
    cout << "[INFO] Loading the convNet and shape predictor" << endl;
    deserialize(argv[1]) >> net >> sp;



    //matrix<rgb_pixel> img;
    // load_image(img, "../mmod_cars_test_image.jpg");
    //load_image(img, argv[2]);
    image_window win;

    // Grab and process frames until the main window is closed by the user.
    cout << "[INFO] Processing the video frames" << endl;
    int carCnt=0;
    int frameCnt=0;
    while(!win.is_closed()) {

      cv::Mat temp;
      if (!cap.read(temp))
      {
          break;
      }
      frameCnt++;
      if (frameCnt % 10 == 0)
        cout << "[INFO] Processed " << frameCnt << " frames. Found " << carCnt << " cars" << endl;
 
      if (frameCnt > 1090) {
        // cap >> temp;
        // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
        // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
        // long as temp is valid.  Also don't do anything to temp that would cause it
        // to reallocate the memory which stores the image as that will make cimg
        // contain dangling pointers.  This basically means you shouldn't modify temp
        // while using cimg.
        //cv_image<bgr_pixel> cimg(temp);
  
        cv_image<bgr_pixel> image(temp);
        matrix<rgb_pixel> cimg;
        assign_image(cimg, image);
  
         //dlib::array2d<bgr_pixel> dlibImage;
         //ib::assign_image(dlibImage, dlib::cv_image<bgr_pixel>(cvMatImage));
  
  
  
        // Run the detector on the image and show us the output.
        int j=0;

        for (auto&& d : net(cimg))
        {
          // We use a shape_predictor to refine the exact shape and location of the detection
          // box.  This shape_predictor is trained to simply output the 4 corner points of
          // the box.  So all we do is make a rectangle that tightly contains those 4 points
          // and that rectangle is our refined detection position.
          auto fd = sp(cimg,d);
          rectangle rect2;
          matrix<rgb_pixel> roi;
          rect2 = fd.get_rect();
          string fname = "../out/car_" + to_string(frameCnt) + "_" + to_string(j++) + ".jpg";
          roi = subm(cimg, rect2);
          save_jpeg(roi,fname);
          //win.set_image(cimg);
          //win.add_overlay(rect2, rgb_pixel(255,0,0));
          carCnt++;
        }
      }
    }

    cout << "Found " << carCnt << " cars" << endl;
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




