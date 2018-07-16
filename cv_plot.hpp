//
//  cv_plot.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef cv_plot_hpp
#define cv_plot_hpp
#include <stdio.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
void plot_landmark(Mat image, string name, vector<vector<float>> kpt);
#endif /* cv_plot_hpp */
