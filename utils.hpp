//
//  utils.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef utils_hpp
#define utils_hpp

#include <stdio.h>
#include <string>
#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
vector<string> get_all_files(string path, string suffix);
vector<string> my_split(string my_str,string seperate);
bool searchkey(vector<int> a, int value);
void getFromText(String nameStr, Mat &myMat);
#endif /* utils_hpp */
