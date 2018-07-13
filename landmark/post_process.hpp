//
//  post_process.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef post_process_hpp
#define post_process_hpp
#include <stdio.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
void post_process(string filePath, string name, string save_path, string faceIndex, string uv_kpt_ind, int resolution, Mat affine_mat, Mat affine_mat_inv);
#endif /* post_process_hpp */
