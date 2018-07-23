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
#include "pre_process.hpp"
using namespace std;
void post_process(string ori_path, string filePath, string save_path, string pose_save, string canonical_vertices, string faceIndex, string uv_kpt_ind, int resolution, vector<Affine_Matrix> &affine_matrix);
#endif /* post_process_hpp */
