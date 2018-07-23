//
//  inference.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/11.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//


#include "inference.hpp"
#include "utils.hpp"
#include "pre_process.hpp"
#include <algorithm>
#include <cublas_v2.h>
#include <cudnn.h>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <dirent.h>
#include <io.h>
#include <unistd.h>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <sstream>
#include <vector>
#include <map>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "NvUtils.h"
#include "common.h"
using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;

static Logger gLogger;
static samples_common::Args args;
#define MAX_WORKSPACE (1<<30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_landmark: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

static const int BatchSize = 1;
static const int INPUT_H = 256;
static const int INPUT_W = 256;
static const int INPUT_CHANNELS = 3;
static const char*  OUTPUT_BLOB_NAME = "resfcn256/Conv2d_transpose_16/Sigmoid";
static const char*  INPUT_BLOB_NAME = "Placeholder";

static const int OUTPUT_SIZE = BatchSize * INPUT_H * INPUT_W * INPUT_CHANNELS;

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory..." << std::endl;
        exit(1);
    }
    return deviceMem;
}

std::string locateFile(const std::string& input)
{
	std::vector<std::string> dirs{
		"data/landmark/"};
	return locateFile(input, dirs);
}


std::vector<std::pair<int64_t, nvinfer1::DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, nvinfer1::DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        nvinfer1::DataType dtype = engine.getBindingDataType(i);
        
        int64_t eltCount = samples_common::volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }
    
    return sizes;
}


void* createCudaBuffer(int64_t eltCount, nvinfer1::DataType dtype, int run)
{
    /* in that specific case, eltCount == INPUT_H * INPUT_W */
    assert(eltCount == BatchSize * INPUT_H * INPUT_W * INPUT_CHANNELS);
    assert(samples_common::getElementSize(dtype) == sizeof(float));
    
    size_t memSize = eltCount * samples_common::getElementSize(dtype);
    float* inputs = new float[eltCount];
    
    /* read jpg file */
    uint8_t fileData[INPUT_W * INPUT_H * INPUT_CHANNELS];
    // readPGMFile(std::to_string(run) + ".pgm", fileData);
   
    
    /* initialize the inputs buffer */
    for (int i = 0; i < eltCount; i++)
        inputs[i] = float(fileData[i]) / 255.;
    
    void* deviceMem = safeCudaMalloc(memSize);
   
    
    CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));
    delete[] inputs;
    
    
    return deviceMem;
}



ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser, IHostMemory*& trtModelStream)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    
    std::cout << "Begin parsing model..." << std::endl;
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    std::cout << "End parsing model..." << std::endl;
    
    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);
    
    std::cout << "Begin building engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
    std::cout << "End building engine..." << std::endl;
    
    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();
    trtModelStream = engine->serialize();
    return engine;
}

void doInference(IExecutionContext& context, float* inputData, float* outputData, int batchSize, int run_num, int iteration)
{
    const ICudaEngine& engine = context.getEngine();
    int nbBindings = engine.getNbBindings();
    /*point to the input and output node*/
    size_t memSize =256*256*3*sizeof(float);
    std::vector<void*> buffers(nbBindings);
    std::vector<std::pair<int64_t, nvinfer1::DataType>> buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);
    int bindingIdxInput = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (engine.bindingIsInput(i))
            bindingIdxInput = i;
        else
        {
            auto bufferSizesOutput = buffersSizes[i];
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first * samples_common::getElementSize(bufferSizesOutput.second));
    }
    }
    
    auto bufferSizesInput = buffersSizes[bindingIdxInput];
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    float *tmpdata;
    for (int i = 0; i < iteration; i++)
    {
        float total = 0, ms;
        for (int run = 0; run < run_num; run++)
        {
            /*create space for input and set the input data*/
	    tmpdata = &inputData[0] + run * INPUT_W *INPUT_H *INPUT_CHANNELS;
            buffers[bindingIdxInput] = safeCudaMalloc(bufferSizesInput.first * samples_common::getElementSize(bufferSizesInput.second));
            CHECK(cudaMemcpyAsync(buffers[inputIndex],tmpdata, batchSize * INPUT_CHANNELS * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice));
            auto t_start = std::chrono::high_resolution_clock::now();
            context.execute(batchSize, &buffers[0]);
            auto t_end = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            total += ms;
            for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
            {
                if (engine.bindingIsInput(bindingIdx))
                    continue;
                auto bufferSizesOutput = buffersSizes[bindingIdx];
            }
            total /= run_num;
            std::cout << "Average over " << run_num << " runs is " << total << " ms." << std::endl;
	    tmpdata = &outputData[0] + run*INPUT_W*INPUT_H*INPUT_CHANNELS;
            CHECK(cudaMemcpyAsync(tmpdata, buffers[outputIndex], memSize, cudaMemcpyDeviceToHost));
       }
    }
    /*get the output data*/
    //CHECK(cudaMemcpyAsync(outputData, buffers[outputIndex], memSize, cudaMemcpyDeviceToHost));
    /*free space*/ 
 
  CHECK(cudaFree(buffers[outputIndex]));
}



int inference(std::string image_path, std::string save_path, vector<Affine_Matrix> &affine_matrix)
{
    vector<string> files;
    vector<string> split_result;
    string  suffix = ".*.jpg";
    std::map<int,string> img_name;
    Mat img,similar_img;
    string tmpname = " ";
    
    auto fileName = locateFile("face.pb.uff");
    std::cout << fileName << std::endl;
    
    auto parser = createUffParser();
  
    /* Register tensorflow input */
    parser->registerInput(INPUT_BLOB_NAME, Dims3(INPUT_CHANNELS, INPUT_W, INPUT_H), UffInputOrder::kNCHW);
    parser->registerOutput(OUTPUT_BLOB_NAME);
    
    IHostMemory* trtModelStream{nullptr};
    ICudaEngine* tmpengine = loadModelAndCreateEngine(fileName.c_str(), BatchSize, parser,trtModelStream);
    
    assert(trtModelStream != nullptr);
    if (!tmpengine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed...");
    tmpengine->destroy();
    
    files = get_all_files(image_path, suffix);
    int N = files.size();
    int run_num = N;
    int iteration = 1;
    /*read image from the folder*/
    vector<float> networkOut(N * INPUT_CHANNELS * INPUT_H * INPUT_W);
    vector<float> data;
    int num = 0;
	std::cout<<"prepare data..."<<endl;
    
    for(int i = 0; i < N; ++i)
    {
	bool isfind = false;
        split_result = my_split(files[i],"/");
        tmpname = split_result[split_result.size()-1];
        img_name.insert(pair<int, string>(i, tmpname));
	vector<Affine_Matrix>::iterator iter;
	for(iter = affine_matrix.begin(); iter!= affine_matrix.end(); ++iter)
	{

	    if((*iter).name == tmpname)
	    {
		img = (*iter).crop_img;
		isfind = true;
		continue;
	    }
	}
	img.convertTo(img, CV_32FC3);
	if( !isfind )
	    continue;
        for(int c=0; c<INPUT_CHANNELS; ++c)
        {
            for(int row=0; row<INPUT_W; row++)
            {
                for(int col=0; col<INPUT_H; col++, ++num)
                {
		    data.push_back(img.at<Vec3f>(row,col)[c]);
                }
            }
        }
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(),nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    
    /*data should be flattened*/
    doInference(*context, &data[0], &networkOut[0], BatchSize, run_num, iteration);
    std::cout<<"Inference uploaded..."<<endl;
    float* outdata=nullptr;
    
    for(int i = 0; i < N; ++i)
    {
        Mat position_map(INPUT_W, INPUT_H, CV_32FC3);
	outdata = &networkOut[0] + i * INPUT_W * INPUT_H * INPUT_CHANNELS;
        vector<float> mydata;
        for (int j=0; j<INPUT_W * INPUT_H * INPUT_CHANNELS; ++j)
        {
            mydata.push_back(outdata[j] * INPUT_W * 1.1);
        }
        
        int n=0;
        for(int row=0; row<INPUT_W; row++)
        {
            for(int col=0; col<INPUT_H; col++)
            {
                position_map.at<Vec3f>(row,col)[2] = mydata[n];
                ++n;
                position_map.at<Vec3f>(row,col)[1] = mydata[n];
                ++n;
                position_map.at<Vec3f>(row,col)[0] = mydata[n];
                ++n;
            }
         }
        tmpname = img_name[i];
	if (access(save_path.c_str(),6) == -1)
   	{
    	 	mkdir(save_path.c_str(), S_IRWXU);
    	}
        cv::imwrite(save_path + "/"+ tmpname, position_map);
    }
    
    /* we need to keep the memory created by the parser */
    parser->destroy();
    engine->destroy();
    return 0;
}
