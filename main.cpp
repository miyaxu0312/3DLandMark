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
#include "post_process.hpp"
#include "inference.hpp"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    //读取图片文件夹下的jpg图片和box.txt中对应的裁剪框坐标，进行裁剪
    string ImagePath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/landmark_Vc-/image";
    string netOutPath="/workspace/run/xyx/TensorRT-4.0.1.6/samples/landmark_Vc-/network_output";
    string postPath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/landmark_Vc-/post";
    string boxPath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/landmark_Vc-/box_api.txt";
    string faceIndex = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/landmark_Vc-/face_ind.txt";
    string uv_kpt_ind = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/landmark_Vc-/uv_kpt_ind.txt";
    string savePath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/landmark_Vc-/crop_image";
    string pose_save = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/landmark_Vc-/pose.txt";
    int resolution = 256;
    /*pre-process the input image*/
    vector<Affine_Matrix> affine_matrix;
    pre_process(ImagePath, boxPath, netOutPath, postPath, uv_kpt_ind, faceIndex, savePath, resolution, affine_matrix);
    cout<<"----------Pre-process Completed----------"<<endl;
    inference(savePath, netOutPath, affine_matrix); //use tensorRT network
    cout<<"----------Network Completed----------"<<endl;
    post_process(ImagePath, netOutPath, postPath,pose_save,faceIndex, uv_kpt_ind, resolution, affine_matrix);
    cout<<"----------Post-process Completed----------"<<endl;
   // waitKey();
    return 0;
}
