//
//  inference.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/11.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//


#include "inference.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cublas_v2.h>
#include <cudnn.h>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <sstream>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;

using namespace std;
#include "common.h"
static Logger gLogger;
static samples_common::Args args;
#define MAX_WORKSPACE (1<<30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_landmark: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

/*
void* createCudaBuffer(int64_t eltCount, nvinfer1::DataType dtype, int run);


inline int64_t volume(const Dims& d)
{
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; i++)
        v *= d.d[i];
    return v;
}



inline unsigned int elementSize(DataType t)
{
    switch (t)
    {
        case DataType::kINT32:
            // Fallthrough, same as kFLOAT
        case DataType::kFLOAT: return 4;
        case DataType::kHALF: return 2;
        case DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}
*/
static const int BatchSize = 1;
static const int INPUT_H = 256;
static const int INPUT_W = 256;
static const int INPUT_CHANNELS = 3;
static const int iteration = 1;
static const int run_num = 1;
static const char*  OUTPUT_BLOB_NAME = "resfcn256/Conv2d_transpose_16/Sigmoid";
static const char*  INPUT_BLOB_NAME = "Placeholder";

static const int OUTPUT_SIZE = BatchSize * INPUT_H * INPUT_W * INPUT_CHANNELS;

template <int C, int H, int W>
void readPPMFile(const std::string& filename, samples_common::PPM<INPUT_CHANNELS, INPUT_H, INPUT_W>& ppm)
{
    ppm.fileName = filename;
    std::ifstream infile(locateFile(filename), std::ifstream::binary);
    infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/landmark/", "data/samples/landmark/"};
    return locateFile(input,dirs);
}


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

void doInference(IExecutionContext& context, float* inputData, float* outputData, int batchSize)
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
            //buffers[i] = createCudaBuffer(bufferSizesOutput.first, bufferSizesOutput.second, 1, false);
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first * samples_common::getElementSize(bufferSizesOutput.second));
            //memSize = bufferSizesOutput.first * samples_common::getElementSize(bufferSizesOutput.second);
	}
	
    }
    
    auto bufferSizesInput = buffersSizes[bindingIdxInput];
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    
    for (int i = 0; i < iteration; i++)
    {
        float total = 0, ms;
        for (int run = 0; run < run_num; run++)
        {
            /*create space for input and set the input data*/
            //buffers[bindingIdxInput] = createMnistCudaBuffer(bufferSizesInput.first,bufferSizesInput.second, run, true);
            buffers[bindingIdxInput] = safeCudaMalloc(bufferSizesInput.first * samples_common::getElementSize(bufferSizesInput.second));
            CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_CHANNELS * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice));
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
                //printOutput(bufferSizesOutput.first, bufferSizesOutput.second, buffers[bindingIdx]);
            }
           
            total /= run_num;
            std::cout << "Average over " << run_num << " runs is " << total << " ms." << std::endl;
       }
    }
    /*get the output data*/
   
    //size_t memSize = bufferSizesOutput.first * samples_common::getElementSize(bufferSizesOutput.second);
    CHECK(cudaMemcpyAsync(outputData, buffers[outputIndex], memSize, cudaMemcpyDeviceToHost));
    /*free space*/
    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
}



int inference(std::string image_path, std::string save_path, cv::Mat img, std::string imgname)
{
    string  suffix = ".*.jpg";
    
    //vector<string> files = get_all_files(image_path, suffix);
    
    auto fileName = locateFile("face.pb.uff");
    std::cout << fileName << std::endl;
    
    auto parser = createUffParser();
  
    /* Register tensorflow input */
    parser->registerInput(INPUT_BLOB_NAME, Dims3(3, 256, 256), UffInputOrder::kNCHW);
    parser->registerOutput(OUTPUT_BLOB_NAME);
    
    IHostMemory* trtModelStream{nullptr};
    ICudaEngine* tmpengine = loadModelAndCreateEngine(fileName.c_str(), BatchSize, parser,trtModelStream);
    
    assert(trtModelStream != nullptr);
    if (!tmpengine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed...");
    tmpengine->destroy();
    /*N is the number of the image when batchsize=1*/
    //files = get_all_files(image_path, suffix);
    int N = 1;//files.size();
    
    
    float* inputs = new float[BatchSize * INPUT_W * INPUT_H * INPUT_CHANNELS];
    //std::vector<samples_common::PPM<INPUT_CHANNELS, INPUT_W, INPUT_H>> networkOut(N);
    /*read image from the folder*/
    vector<float> networkOut(N * INPUT_CHANNELS * INPUT_H * INPUT_W);
    vector<float> data;
    int num=0;
    //Mat img(INPUT_W,INPUT_H,CV_8UC3);
	std::cout<<"prepare data..."<<endl;
    for(int i = 0;i<1; ++i)
    {
	
    img.convertTo(img,CV_32FC3);
	for(int c=0;c<=2;++c)
        {
            for(int row=0;row<INPUT_W;row++)
            {
			for(int col=0;col<INPUT_H;col++,++num)
			{
				data.push_back(img.at<Vec3f>(row,col)[c]);
				//data.push_back(img.at<Vec3f>(row,col)[1]);
			    //data.push_back(img.at<Vec3f>(row,col)[0]);
				//cout<<data[num]<<",";
			    //cout<<img.at<Vec3f>(row,col)[0]<<","<<img.at<Vec3f>(row,col)[1]<<","<<img.at<Vec3f>(row,col)[2]<<endl;
			}
		}
 
        }
    }
	

    std::cout << " Data Size  " << data.size() << std::endl;
    std::cout<<"Data prepared..."<<endl;
    
    std::cout << "Model deserializing" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(),nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    
    /*data should be flattened*/
    doInference(*context, &data[0], &networkOut[0], BatchSize);
    std::cout<<"Inference uploaded..."<<endl;
    
    for(int i = 0; i < N; ++i)
    {
        float* outdata=nullptr;
		outdata =&networkOut[0];//+ i * INPUT_W * INPUT_H * INPUT_CHANNELS;
        vector<float> mydata;
	//std::vector<samples_common::PPM<INPUT_CHANNELS, INPUT_W, INPUT_H>> pm(N);
	    //pm = &networkOut[0];
		for (int i=0;i<256*256*3;++i)
		{

			mydata.push_back(outdata[i]*256*1.1);
		}
	Mat position_map(INPUT_W, INPUT_H, CV_32FC3);
	int n=0;
        	for(int row=0;row<256;row++)
        	{
            	
				for(int col=0;col<256;col++)
           	 	{
					position_map.at<Vec3f>(row,col)[2]= mydata[n];++n;
					position_map.at<Vec3f>(row,col)[1] = mydata[n];++n;
					position_map.at<Vec3f>(row,col)[0] = mydata[n];++n;
          	  	}
       		 }

	
        cv::imwrite(save_path + "/"+imgname, position_map);
    
    }
    
    /* we need to keep the memory created by the parser */
    parser->destroy();
    
    //execute(*engine);
    engine->destroy();
    //shutdownProtobufLibrary();
    return 0;
}
