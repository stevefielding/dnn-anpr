// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    Based on dnn_mmod_find_cars_ex.cpp
    Removed the shape predictor, and hard coded reference to mmod_lplate_detector.dat

    You can also see some videos of this vehicle detector running on YouTube:
        https://www.youtube.com/watch?v=4B3bzmxMAZU
        https://www.youtube.com/watch?v=bP2SUo5vSlc
*/
// Build: cmake --build . --config Debug

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

using namespace std;
using namespace dlib;



// The rear view vehicle detector network
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<55,SUBNET>>>;
using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char **argv) try
{
	int plateCnt = 0;
    net_type net;
    // Load NN model and weights from file. Read image from file, detect plates, crop the plates 
    // from the image and save to file

    if (argc != 4) {
      cout << "ERROR: Must supply path to model file, input image filename, and output image filename prefix\n";
      cout << "Eg:  ..datasets/mmod_lplate_detector.dat ../images/myImage.jpg  ../out/myImage_" << endl;
      return -1;
    }

//#define INCLUDE_SHAPE_PRED
#ifdef INCLUDE_SHAPE_PRED
    shape_predictor sp;
    deserialize("../datasets/mmod_rear_end_vehicle_detector.dat") >> net >> sp;
#else
    deserialize(argv[1]) >> net;
#endif


    cout << "Loading: " << argv[2];
    matrix<rgb_pixel> img;
    load_image(img, argv[2]);

//#define DISPLAY_IMAGE
#ifdef DISPLAY_IMAGE
    image_window win;
    win.set_image(img);
#endif

    // Run the detector on the image and show us the output.
    for (auto&& d : net(img))
    {
        rectangle rect = d.rect;
        //cout << "d: " << rect << endl;
#ifdef DISPLAY_IMAGE
        win.add_overlay(rect, rgb_pixel(255,0,0));
#endif
        matrix<rgb_pixel> roi;
        string fname = argv[3] + to_string(plateCnt++) + ".jpg";
        roi = subm(img, rect);
        save_jpeg(roi,fname);
    }

    cout << ", found " << plateCnt << " plates" << endl;
    //cout << "Hit enter to end program" << endl;
    //cin.get();
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




