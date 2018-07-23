
//  pre_process.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "pre_process.hpp"
//
//  pre_process.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <string.h>
#include <vector>
#include <fstream>
#include <io.h>
#include <unistd.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"
#include "post_process.hpp"
#include "inference.hpp"
using namespace std;
using namespace cv;

/*get the face detection result from txt file*/
vector<int> get_box(string path, string name)
{
    vector<string> box;
    vector<int> box_int;
    ifstream f;
    f.open(path);
    string line;               //保存读入的每一行
    while(getline(f,line))
    {
        if (line+".jpg" == name)
        {
            getline(f,line);
            box = my_split(line,",");
            vector<string>::iterator iter;
            for(iter=box.begin();iter!=box.end();++iter)
            {
                box_int.push_back(stoi(*iter));
            }
        }
        else{
            getline(f,line);
        }
    }
    return box_int;
}

void pre_process(string filePath, string boxPath, string netOutPath, string postPath, string uv_kpt_ind,string faceIndex, string savePath, int resolution, vector<Affine_Matrix> &affine_matrix)
{
    vector<string> files;
    vector<string> split_result;
    vector<int> box;
    string  suffix = ".*.jpg";
    Mat img;
    files = get_all_files(filePath, suffix);
    cout<<"-----image num:-----"<<files.size()<<endl;
    Affine_Matrix tmp_affine_mat;
    if(files.size() == 0)
    {
	    cerr<<"-----no image data-----"<<endl;
	    exit(1);
    }
    for(int i=0;i<files.size();++i)
    {
        split_result = my_split(files[i],"/");
        string name = split_result[split_result.size()-1];
        img = imread(files[i], CV_LOAD_IMAGE_UNCHANGED); // 读取每一张图片
      
        Mat similar_img;
        box = get_box(boxPath, name);
        int old_size = (box[1] - box[0] + box[3] - box[2])/2;
        int size = old_size * 1.58;
        float center_x=0.0, center_y=0.0;
	box[3] = box[3]- old_size * 0.3;
	box[1] = box[1] - old_size * 0.25;
	box[0] = box[0] + old_size * 0.2;
        center_x = box[1] - (box[1] - box[0]) / 2.0;
        center_y = box[3] - (box[3] - box[2]) / 2.0 + old_size * 0.14;
        
        float temp_src[3][2] = {{center_x-size/2, center_y-size/2},{center_x - size/2, center_y + size/2},{center_x+size/2, center_y-size/2}};
        
        Mat srcMat(3, 2, CV_32F,temp_src);
        float temp_dest[3][2] = {{0, 0},{0, static_cast<float>(resolution-1)},{static_cast<float>(resolution-1), 0}};
        Mat destMat(3, 2, CV_32F,temp_dest);
        Mat affine_mat = getAffineTransform(srcMat, destMat);
        img.convertTo(img,CV_32FC3);
        img = img/255.;
        
        warpAffine(img, similar_img, affine_mat,  similar_img.size());
        /*save pre-processed image for the network*/
	if (access(savePath.c_str(),6)==-1)
   	{
    	 	mkdir(savePath.c_str(), S_IRWXU);
    	}
        imwrite(savePath+"/" + name,similar_img);
        tmp_affine_mat.name = name;
        tmp_affine_mat.affine_mat = affine_mat;
	tmp_affine_mat.crop_img = similar_img;
        affine_matrix.push_back(tmp_affine_mat);
    }
    
}
