//
//  post_process.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "post_process.hpp"
#include "pre_process.hpp"
#include "utils.hpp"
#include "get_results.hpp"
#include "cv_plot.hpp"
#include "utils.hpp"
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <string>
#include <string.h>
#include <vector>
#include <regex>
#include <cassert>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

void post_process(string ori_path, string filePath, string save_path, string pose_save, string faceIndex, string uv_kpt_ind_path, int resolution, vector<Affine_Matrix> affine_matrix)
{
    vector<string> files;
    vector<string> split_result;
    vector<float> face_ind;
    bool isfind = false;
    vector<float> uv_kpt_ind1,uv_kpt_ind2;
    Mat img, ori_img, z, vertices_T, stacked_vertices, affine_mat_stack;
    Mat pos(resolution,resolution,CV_8UC3);
    Mat affine_mat, affine_mat_inv;
    string name;
    
    files = get_all_files(filePath, suffix);
    cout<<"-----image num:-----"<<files.size()<<endl;
    for(uint i=0;i<files.size();++i)
    {
        string tmp;
        isfind = false;
        img = imread(files[i], IMREAD_UNCHANGED); // 读取每一张图片
        split_result = my_split(files[i],"/");
        name = split_result[split_result.size()-1]; //获取图片名
        
        vector<Affine_Matrix>::iterator iter;
        for(iter = affine_matrix.begin(); iter!= affine_matrix.end(); ++iter)
        {
            if(*iter.name == name)
            {
                affine_mat =  *iter.affine_mat;
                invertAffineTransform(affine_mat, affine_mat_inv);
                isfind = true;
            }
        }
        if( !isfind )
            continue;
        cout<<"----------img: "<<name<<" loaded---------"<<endl;
        ori_img = imread(ori_path+"/"+name, IMREAD_UNCHANGED);    //加载原始图片，方便画landmark
        Mat cropped_vertices(resolution*resolution,3,img.type()), cropped_vertices_T(3,resolution*resolution,img.type());
        
        cropped_vertices = img.reshape(1, resolution*resolution);
        Mat cropped_vertices_swap(resolution*resolution,3,cropped_vertices.type());
        
        cropped_vertices.col(0).copyTo(cropped_vertices_swap.col(2));
        cropped_vertices.col(1).copyTo(cropped_vertices_swap.col(1));
        cropped_vertices.col(2).copyTo(cropped_vertices_swap.col(0));
        
        transpose(cropped_vertices_swap, cropped_vertices_T);
        cropped_vertices_T.convertTo(cropped_vertices_T, affine_mat.type());
        z = cropped_vertices_T.row(2).clone() / affine_mat.at<double>(0,0);
     
        Mat ones_mat(1,resolution*resolution,cropped_vertices_T.type(),Scalar(1));
        ones_mat.copyTo(cropped_vertices_T.row(2));
        
        cropped_vertices_T.convertTo(cropped_vertices_T, affine_mat.type());
        Mat vertices, affine_mat_inv;
      
        vertices =  affine_mat_inv * cropped_vertices_T;
        z.convertTo(z, vertices.type());
        
        vconcat(vertices.rowRange(0, 2), z, stacked_vertices);
        transpose(stacked_vertices, vertices_T);
        pos = vertices_T.reshape(3,resolution);
        Mat pos2(resolution,resolution,CV_64FC3);
        
        for (int i=0; i<pos.rows;++i)
        {
            for (int j=0; j<pos.cols;++j)
            {
                pos2.at<Vec3d>(i,j)[0] = pos.at<Vec3d>(i,j)[2];
                pos2.at<Vec3d>(i,j)[1] = pos.at<Vec3d>(i,j)[1];
                pos2.at<Vec3d>(i,j)[2] = pos.at<Vec3d>(i,j)[0];
            }
            
        }
        imwrite(save_path+"/"+name,pos);
        //cout<<"----------position map saved---------"<<endl;
        ifstream f;
        f.open(faceIndex);
        assert(f.is_open());
        
        while(getline(f, tmp))
        {
            istringstream iss(tmp);
            float num;
            iss >> num;
            face_ind.push_back(num);
        }
        //cout<<"----------face index data loaded---------"<<endl;
        f.close();
        
        f.open(uv_kpt_ind_path);
        assert(f.is_open());
        getline(f, tmp);
        int i = 1;
        vector<string> all_uv = my_split(tmp, " ");
        vector<string>::iterator uv_iter;
        for (uv_iter=all_uv.begin();uv_iter!=all_uv.end();++uv_iter,++i)
        {
            istringstream iss(*uv_iter);
            float num;
            iss >> num;
            if (i<=68 && i>0)
                uv_kpt_ind1.push_back(num);
            else if (i>68 && i<=68*2)
                uv_kpt_ind2.push_back(num);
        }
        //cout<<"----------kpt index data loaded---------"<<endl;
        f.close();
        vector<vector<float>> all_vertices = get_vertices(pos2, face_ind, resolution);
        vector<vector<float>> landmark = get_landmark(pos2, uv_kpt_ind1, uv_kpt_ind2);
        //cout<<"----------get landmark Completed----------"<<endl;
        plot_landmark(ori_img, name, landmark);
        vector<float> pose = estimate_pose(all_vertices);
        //cout<<"----------estimate pose Completed----------"<<endl;
        ofstream outfile(pose_save, ios::app);
        outfile<<"name:"<<name<<"\n";
        vector<float>::iterator iter;
        outfile<<"pose: ";
        for(iter=pose.begin();iter!=pose.end();iter++)
        {
            outfile<<*iter<<",";
        }
        outfile<<"\n";
        outfile.close();
    }
}
