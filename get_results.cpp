
//
//  get_results.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "get_results.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
using namespace cv;


vector<vector<float>> get_vertices(Mat pos, vector<float> face_ind, int resolution)
{
    Mat all_vertices = pos.reshape(1,resolution*resolution);
    vector<vector<float>> result(face_ind.size(),vector<float>(3,0));
    vector<float>::iterator iter;
    int i=0;
    for (iter=face_ind.begin();iter!=face_ind.end();iter++)
    {
        result[i][0] = all_vertices.at<double>(int(*iter),2);
        result[i][1] = all_vertices.at<double>(int(*iter),1);
        result[i][2] = all_vertices.at<double>(int(*iter),0);
        i++;
    }
    
    return result;
}

vector<vector<float>> get_landmark(Mat pos, vector<float> uv_kpt_ind_0,vector<float> uv_kpt_ind_1)
{
    vector<vector<float>> landmark(68,vector<float>(3,0));
    for (int i=0; i<uv_kpt_ind_0.size();++i)
    {
        landmark[i][0] = pos.at<Vec3d>(uv_kpt_ind_1[i],uv_kpt_ind_0[i])[2];
        landmark[i][1] = pos.at<Vec3d>(uv_kpt_ind_1[i],uv_kpt_ind_0[i])[1];
        landmark[i][2] = pos.at<Vec3d>(uv_kpt_ind_1[i],uv_kpt_ind_0[i])[0];
    }
    
    return landmark;
}

vector<float> estimate_pose(vector<vector<float>> vertices)
{
    string path = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/landmark_Vc-/canonical_vertices.txt";
    Mat canonical_vertices_homo;
    Mat canonical_vertices = Mat::zeros(131601/3, 3, CV_32FC1);
    getFromText(path, canonical_vertices);
    
    Mat ones_mat(131601/3,1,canonical_vertices.type(),Scalar(1));
    ones_mat.convertTo(ones_mat, CV_32F);
    hconcat(canonical_vertices, ones_mat, canonical_vertices_homo);
    //cout<<ones_mat;
    //cout<<canonical_vertices_homo;
    Mat canonical_vertices_homo_T, vertices_T;
    CvMat *canonical_vertices_homo_T_pointer=cvCreateMat(43867, 4,CV_32FC1);
    CvMat *vertices_T_pointer=cvCreateMat(43867, 3,CV_32FC1);
    CvMat *P_pointer=cvCreateMat(4, 3,CV_32FC1);
    
    cvSetData(canonical_vertices_homo_T_pointer, canonical_vertices_homo.data, CV_AUTOSTEP);
    //cvSetData(vertices_T_pointer, vertices.data(), CV_AUTOSTEP);
    for (int i=0;i<43867;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvmSet(vertices_T_pointer, i, j, vertices[i][j]);
        }
    }
    //cout<<vertices.data();
    cvSolve(canonical_vertices_homo_T_pointer,vertices_T_pointer, P_pointer);
    
    Mat P(P_pointer->rows,P_pointer->cols,P_pointer->type,P_pointer->data.fl);
    Mat P_T;
    transpose(P, P_T);
    //cout<<P_T;
    Mat rotation_matrix = P2sRt(P_T);
    cout<<"-------P2sRt completed-------"<<endl;
    vector<float> pose = matrix2angle(rotation_matrix);
    cout<<"-------matrix2angle completed-------"<<endl;
    //save pos
    return pose;
}

//p 3*4
Mat P2sRt(Mat P)
{
    Mat t2d(2,1, P.type());
    Mat R1(1,3, P.type());
    Mat R2(1,3, P.type());
    Mat R(3,3, P.type());
    //t2d
    Mat P_row0 = P.rowRange(0,1).clone();
    R1 = P_row0.colRange(0, 3).clone();
    Mat P_row1 = P.row(1).clone();
    P_row1.colRange(0, 3).copyTo(R2);
    Mat r1 = R1 / norm(R1);
    Mat r2 = R2 / norm(R2);
    CvMat *r1_pointer=cvCreateMat(1, 3,CV_32FC1);
    cvSetData(r1_pointer, r1.data, CV_AUTOSTEP);
    CvMat *r2_pointer=cvCreateMat(1, 3,CV_32FC1);
    cvSetData(r2_pointer, r2.data, CV_AUTOSTEP);
    CvMat *r3_pointer=cvCreateMat(1, 3,CV_32FC1);
    cvCrossProduct(r1_pointer,r2_pointer,r3_pointer);
    Mat r3(r3_pointer->rows,r3_pointer->cols,r3_pointer->type,r3_pointer->data.fl);
    vconcat(r1, r2, R);
    vconcat(R, r3, R);
    return R;
}

//r 3*3
vector<float> matrix2angle(Mat R)
{
    vector<float> pose_angle;
    float x,y,z;
    if (R.at<float>(2,0) != 1 || R.at<float>(2,0) != -1)
    {
        x =asin(R.at<float>(2,0));
        y = atan2(R.at<float>(2,1)/cos(x), R.at<float>(2,2)/cos(x));
        z = atan2(R.at<float>(1,0)/cos(x), R.at<float>(0,0)/cos(x));
    }
    else{
        z = 0;
        if (R.at<float>(2,0) == -1)
        {
            x = M_PI / 2;
            y = z +atan2(-R.at<float>(0,1), -R.at<float>(0,2));
        }
    }
    pose_angle.push_back(x);
    pose_angle.push_back(y);
    pose_angle.push_back(z);
    return pose_angle;
}

