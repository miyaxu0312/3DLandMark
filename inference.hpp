//
// inference.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/11.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef inference_hpp
#define inference_hpp
#include <cassert>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cuda_runtime_api.h>
#include <string>
#include <numeric>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <utility>
#include "common.h"
#include "NvUffParser.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "pre_process.hpp"
using namespace nvuffparser;
using namespace nvinfer1;
void* safeCudaMalloc(size_t memSize);
void* createCudaBuffer(int64_t eltCount, nvinfer1::DataType dtype, int run, bool isinput=false);
std::vector<std::pair<int64_t,nvinfer1::DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize);
ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchsize, IUffParser* parser, IHostMemory*& trtModelStream);
void doInference(IExecutionContext& context, float* inputData, float* outputData, int batchSize, int N);
void readImage(const std::string& filename, uint8_t* buffer);
int inference(std::string image_path, std::string save_path, vector<Affine_Matrix> &affine_matrix);
#endif /* inference_hpp */
