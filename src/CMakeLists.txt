# This is a CMake makefile.  CMake is a tool that helps you build C++ programs.
# You can download CMake from http://www.cmake.org.  This CMakeLists.txt file
# you are reading builds dlib's example programs. 
#


cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(dnn-anpr)


# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.
add_subdirectory(../../../dlib/dlib-19.8 dlib_build)


# The next thing we need to do is tell CMake about the code you want to
# compile.  We do this with the add_executable() statement which takes the name
# of the output executable and then a list of .cpp files to compile.  Here we
# are going to compile one of the dlib example programs which has only one .cpp
# file, assignment_learning_ex.cpp.  If your program consisted of multiple .cpp
# files you would simply list them here in the add_executable() statement.  
#add_executable(dnn_mmod_find_cars_ex dnn_mmod_find_cars_ex.cpp)

# Finally, you need to tell CMake that this program, assignment_learning_ex,
# depends on dlib.  You do that with this statement: 
#target_link_libraries(dnn_mmod_find_cars_ex dlib::dlib)




# To compile this program all you need to do is ask cmake.  You would type
# these commands from within the directory containing this CMakeLists.txt
# file:
# mkdir build
# cd build
# cmake  -DCMAKE_BUILD_TYPE=Debug  ../src
# cmake --build . --config Debug
#
# The cmake .. command looks in the parent folder for a file named
# CMakeLists.txt, reads it, sets up everything needed to build program.  Also,
# note that CMake can generate Visual Studio or XCode project files.  So if
# instead you had written:
#   mkdir build
#   cmake .. -G Xcode
#
# You would be able to open the resulting Xcode project and compile and edit
# the example programs within the Xcode IDE.  CMake can generate a lot of
# different types of IDE projects.  Run the cmake -h command to see a list of
# arguments to -G to see what kinds of projects cmake can generate for you.  It
# probably includes your favorite IDE in the list.

if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the webcam_face_pose_ex example.")
else()
   # find_package(OpenCV QUIET)
   find_package(OpenCV)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

      #add_executable(webcam_face_pose_ex webcam_face_pose_ex.cpp)
      add_executable(dnn_mmod_find_cars_save_to_file dnn_mmod_find_cars_save_to_file.cpp)
      add_executable(dnn_mmod_find_lplates_save_to_file dnn_mmod_find_lplates_save_to_file.cpp)
      add_executable(test_simple test_simple.cpp)
      add_executable(dnn_mmod_train_find_lplates dnn_mmod_train_find_lplates.cpp)
      add_executable(dnn_mmod_simple_train_and_test dnn_mmod_simple_train_and_test.cpp)
      add_executable(dnn_mmod_find_lplates_no_shape_pred dnn_mmod_find_lplates_no_shape_pred.cpp)

      #target_link_libraries(webcam_face_pose_ex dlib::dlib ${OpenCV_LIBS} )
      target_link_libraries(dnn_mmod_find_cars_save_to_file dlib::dlib ${OpenCV_LIBS} )
      target_link_libraries(dnn_mmod_find_lplates_save_to_file dlib::dlib ${OpenCV_LIBS} )
      target_link_libraries(test_simple dlib::dlib ${OpenCV_LIBS} )
      target_link_libraries(dnn_mmod_train_find_lplates dlib::dlib ${OpenCV_LIBS} )
      target_link_libraries(dnn_mmod_simple_train_and_test dlib::dlib ${OpenCV_LIBS} )
      target_link_libraries(dnn_mmod_find_lplates_no_shape_pred dlib::dlib ${OpenCV_LIBS} )
   else()
      message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()
endif()



