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
using namespace std;

struct Affine_Matrix
{
    string name;
    Mat affine_mat;
	Mat crop_img;
};

void pre_process(string ImagePath, string boxPath, string netOutPath, string postPath, string uv_kpt_ind, string faceIndex, string savePath, int resolution, vector<Affine_Matrix> &affine_matrix);
#endif /* pre_process_hpp */
