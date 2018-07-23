//
//  cv_plot.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "cv_plot.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

void plot_landmark(Mat img, string name, vector<vector<float>> kpt)
{
    Mat image = img.clone();
    vector<int> end_list = {17-1, 22-1, 27-1, 42-1, 48-1, 31-1, 36-1, 68-1};
    for(int i=0;i<68;i++)
    {
        int start_point_x, start_point_y, end_point_x, end_point_y;
        start_point_x = int(round(kpt[i][0]));
        start_point_y = int(round(kpt[i][1]));
        Point center1(start_point_x,start_point_y);
        circle(image, center1, 2, Scalar(0,0,255));
        if (searchkey(end_list,i))
            continue;
        end_point_x = int(round(kpt[i+1][0]));
        end_point_y = int(round(kpt[i+1][1]));
        Point center2(end_point_x,end_point_y);
        line(image, center1, center2, Scalar(0,255,0));
    }
    //imshow("Landmark",image);
    string path = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/landmark_Vc-/plot_kpt";
    if (access(path.c_str(),6)==-1)
    {
        mkdir(path.c_str(), S_IRWXU);
    }
    imwrite(path + "/" + name, image);
    cout<<"-----plot-kpt-completed-----"<<endl;
}

