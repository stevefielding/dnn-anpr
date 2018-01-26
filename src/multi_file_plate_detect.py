# USAGE: python multi_file_plate_detect.py --app ../build/dnn_mmod_find_lplates_no_shape_pred --inputPath ../images --outputPath ../out

import os
from imutils import paths
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputPath", required=True,
  help="path to input images")
ap.add_argument("-a", "--app", required=True,
  help="path to application")
ap.add_argument("-o", "--outputPath", required=True,
	help="path to output images")
ap.add_argument("-m", "--model", required=True,
	help="path to model")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["inputPath"]))
for imagePath in imagePaths:
  outPath = imagePath.split('/') [-1]
  outPath = args["outputPath"] + '/' + ''.join(outPath.split('.')[:-1])
  #print (outPath)
  os.system("{} {} {} {}".format(args["app"], args["model"], imagePath, outPath))

