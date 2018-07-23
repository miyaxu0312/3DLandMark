//
//  get_results.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef get_results_hpp
#define get_results_hpp
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
vector<vector<float>> get_vertices(Mat pos, vector<float> face_ind, int resolution);
vector<vector<float>> get_landmark(Mat pos, vector<float> uv_kpt_ind_0, vector<float> uv_kpt_ind_1);
vector<float> estimate_pose(vector<vector<float>> vertices, string canonical_vertices_path);
Mat P2sRt(Mat p);
vector<float> matrix2angle(Mat p);
#endif /* get_results_hpp */
