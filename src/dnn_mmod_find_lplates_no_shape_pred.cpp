// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    Based on dnn_mmod_find_cars_ex.cpp
    Removed the shape predictor, and hard coded reference to mmod_lplate_detector.dat

    You can also see some videos of this vehicle detector running on YouTube:
        https://www.youtube.com/watch?v=4B3bzmxMAZU
        https://www.youtube.com/watch?v=bP2SUo5vSlc
*/
// Build: cmake --build . --config Debug
// Load NN weights from file. Read an image from file, detect the location of plates in the image using NN, 
// and then crop the plates from the image and save the plate images to file

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>

using namespace std;
using namespace dlib;

// Lplate neywork, same as dlib example rear view vehicle detector network.
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<55,SUBNET>>>;
using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char **argv) try
{
    int plateCnt = 0;
    if (argc != 4) {
      cout << "ERROR: Must supply path to model file, input image filename, and output image filename prefix\n";
      cout << "output image filename prefix will be suffixed with plate number and .jpg, eg 0.jpg" << endl;
      cout << "USAGE:  ..datasets/mmod_lplate_detector.dat ../images/myImage.jpg  ../out/myImage_" << endl;
      return -1;
    }

    // Create net model and load the weights
    net_type net;
    deserialize(argv[1]) >> net;

    // Load the image
    cout << "Loading: " << argv[2];
    matrix<rgb_pixel> img;
    load_image(img, argv[2]);

//#define DISPLAY_IMAGE
#ifdef DISPLAY_IMAGE
    image_window win;
    win.set_image(img);
#endif

    // Run the detector on the image, show us the output, and save the cropped image to file.
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
    cout << "The image failed to load." << endl;
}
catch(serialization_error& e)
{
    cout << e.what() << endl;
    cout << "The model file is not correct" << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




