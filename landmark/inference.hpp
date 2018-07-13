//
//  inference.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/11.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef inference_hpp
#define inference_hpp

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include <string>
#include <numeric>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include "common.h"
void readPPMFile(const std::string& filename, samples_common::PPM<C, H, W>& ppm);
std::string locateFile(const std::string& input);
void* safeCudaMalloc(size_t memSize);
void* createCudaBuffer(int64_t eltCount, DataType dtype, int run, bool isinput=false)
void calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize);
ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser, IHostMemory*& trtModelStream);
void doInference(IExecutionContext& context, float* inputData, float* outputData, int batchSize);
void readImage(const std::string& filename, uint8_t* buffer);
void inference(std::string image_path, std::string save_path);
#endif /* inference_hpp */
