//
//  main.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/3.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <string.h>
#include <vector>
#include <regex>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "pre_process.hpp"
using namespace std;
using namespace cv;


int main(int argc, const char * argv[]) {
    //读取图片文件夹下的jpg图片和box.txt中对应的裁剪框坐标，进行裁剪
    string ImagePath = "samples/landmark/landmark/image";
    string netOutPath = "samples/landmark/landmark/network_output";
    string postPath = "samples/landmark/landmark/post";
    string boxPath = "samples/landmark/landmark/box.txt";
    string faceIndex = "samples/landmark/landmark/face_ind.txt";
    string uv_kpt_ind = "samples/landmark/landmark/uv_kpt_ind.txt";
    string savePath = "samples/landmark/landmark/crop_image";
    int resolution = 256;
    pre_process(ImagePath, boxPath, netOutPath, postPath, uv_kpt_ind, faceIndex, savePath, resolution);
    //network output
    waitKey();
    return 0;
}
