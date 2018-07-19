//
//  pre_process.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//
#pragma once
#ifndef pre_process_hpp
#define pre_process_hpp
#include <stdio.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
struct Affine_Matrix
{
    string name;
    Mat affine_mat(2,3,CV_32F);
};

void pre_process(string ImagePath, string boxPath, string netOutPath, string postPath, string uv_kpt_ind, string faceIndex, string savePath, int resolution, vector<Affine_Matrix> &affine_matrix);
#endif /* pre_process_hpp */
