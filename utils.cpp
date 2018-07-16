//
//  utils.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/5.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "utils.hpp"
#include <string>
#include<vector>
#include <regex>
#include <fstream>
#include <dirent.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

vector<string> get_all_files(string path, string suffix)
{
    vector<string> files;
    regex reg_obj(suffix, regex::icase);
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(path.c_str())) == NULL)
    {
        cerr << "can not open this file." << endl;
    }
    else{
        while((dirp = readdir(dp)) != NULL)
        {
            //cout<<dirp->d_name<<endl;
            if(dirp->d_type == 8 && regex_match(dirp->d_name, reg_obj))
            {
                //cout << dirp->d_name << endl;
                string file_absolute_path = path.c_str();
                file_absolute_path = file_absolute_path.append("/");
                file_absolute_path = file_absolute_path.append(dirp->d_name);
                files.push_back(file_absolute_path);
            }
        }
    }
    closedir(dp);
    return files;
}

vector<string> my_split(string my_str,string seperate)
{
    vector<string> result;
    size_t split_index = my_str.find(seperate);
    size_t start = 0;
    
    while(string::npos!=split_index)
    {
        result.push_back(my_str.substr(start,split_index-start));
        //cout<<my_str.substr(start,split_index-start)<<endl;
        start = split_index+seperate.size();
        split_index = my_str.find(seperate,start);
    }
    result.push_back(my_str.substr(start,split_index-start));
    return result;
}

bool searchkey(vector<int> a, int value)
{
    for(int i=0;i<a.size();i++)
    {
        if(a[i]==value)
            return true;
    }
    return false;
}


void getFromText(String nameStr, Mat &myMat)
{
    ifstream myFaceFile;
    myFaceFile.open(nameStr);
    vector<string> result(3);
    string tmp;
    int i=0,line = 0;
    while(getline(myFaceFile, tmp))
    {
        result = my_split(tmp, ",");
        vector<string>::iterator iter;
        i = 0;
        for(iter=result.begin();iter!=result.end();++iter,i++)
        {
            istringstream iss(*iter);
            float num;
            iss >> num;
            myMat.at<float>(line,i)=num;
        }
        ++line;
        
    }
    myFaceFile.close();
}


