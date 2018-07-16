//
//  inference.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/11.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "inference.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <cudnn.h>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include "utils.hpp"
//#include "BatchStreamPPM.h"
//#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
using namespace nvuffparser;
using namespace nvinfer1;
//using namespace cv;
#include "common.h"
#include "utils.hpp"

static Logger gLogger;
static samples_common::Args args;
#define MAX_WORKSPACE (1_GB)

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
static const string OUTPUT_BLOB_NAME = "resfcn256/Conv2d_transpose_16/Sigmoid";
static const string INPUT_BLOB_NAME = "Placeholder";

static const int OUTPUT_SIZE = BatchSize * INPUT_H * INPUT_W * INPUT_CHANNELS;

template <int C, int H, int W>
void readPPMFile(const std::string& filename, samples_common::PPM<INPUT_CHANNEL,INPUT_H, INPUT_W>& ppm)
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

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);
        
        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }
    
    return sizes;
}


void* createCudaBuffer(int64_t eltCount, DataType dtype, int run, bool isinput=false)
{
    /* in that specific case, eltCount == INPUT_H * INPUT_W */
    assert(eltCount == BatchSize * INPUT_H * INPUT_W * INPUT_CHANNELS);
    assert(elementSize(dtype) == sizeof(float));
    
    size_t memSize = eltCount * samples_common::getElementSize(dtype);
    float* inputs = new float[eltCount];
    
    /* read jpg file */
    uint8_t fileData[INPUT_W * INPUT_H * INPUT_CHANNELS];
    // readPGMFile(std::to_string(run) + ".pgm", fileData);
   
    
    /* initialize the inputs buffer */
    for (int i = 0; i < eltCount; i++)
        inputs[i] = float(fileData[i]) / 255.0;
    
    void* deviceMem = safeCudaMalloc(memSize);
    if (isinput)
    {
        CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));
        delete[] inputs;
    }
    
    return deviceMem;
}

/*
void printOutput(int64_t eltCount, DataType dtype, void* buffer)
{
    std::cout << eltCount << " eltCount" << std::endl;
    assert(elementSize(dtype) == sizeof(float));
    std::cout << "--- OUTPUT ---" << std::endl;
    
    size_t memSize = eltCount * elementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));
    
    int maxIdx = 0;
    for (int i = 0; i < eltCount; ++i)
        if (outputs[i] > outputs[maxIdx])
            maxIdx = i;
    eltCount = 10;
    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        std::cout << eltIdx << " => " << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
            std::cout << "***";
        std::cout << "\n";
    }
    
    std::cout << std::endl;
    delete[] outputs;
}
*/

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
    std::vector<void*> buffers(nbBindings);
    std::vector<std::pair<int64_t, DataType>> buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);
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
            buffers[bindingIdxInput] = safeCudaMalloc(bufferSizesInput.first * elementSize(bufferSizesInput.second));
            CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_CHANNELS * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice));
            auto t_start = std::chrono::high_resolution_clock::now();
            context->execute(batchSize, &buffers[0]);
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
   
    size_t memSize = bufferSizesOutput.first * samples_common::getElementSize(bufferSizesOutput.second)
    CHECK(cudaMemcpyAsync(outputData, buffers[outputIndex], memSize, cudaMemcpyDeviceToHost));
    /*free space*/
    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
}



void inference(std::string image_path, std::string save_path)
{
    string  suffix = ".*.ppm";
    Mat img,similar_img;
    files = get_all_files(filePath, suffix);
    
    auto fileName = locateFile("face.pb.uff");
    std::cout << fileName << std::endl;
    
    auto parser = createUffParser();
    
    /* Register tensorflow input */
    parser->registerInput(INPUT_BLOB_NAME, Dims3(3, 256, 256), UffInputOrder::kNCHW);
    parser->registerOutput(OUTPUT_BLOB_NAME);
    
    IHostMemory* trtModelStream{nullptr};
    ICudaEngine* engine = loadModelAndCreateEngine(fileName.c_str(), BatchSize, parser,trtModelStream);
    
    assert(trtModelStream != nullptr)
    if (!engine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed...");
    
    /*N is the number of the image when batchsize=1*/
    files = get_all_files(image_path, suffix);
    int N = files.size();
    
    std::vector<samples_common::PPM<INPUT_CHANNELS, INPUT_W, INPUT_H>> ppms(N);
    samples_common::PPM<INPUT_CHANNELS, INPUT_W, INPUT_H> ppm;
    float* inputs = new float[BatchSize * INPUT_W * INPUT_H * INPUT_CHANNELS];
    std::vector<samples_common::networkOut<INPUT_CHANNELS, INPUT_W, INPUT_H>> networkOut(N);
    /*read image from the folder*/
    
    for(int i = 0;i < files.size(); ++i)
    {
        /*read image into buffer*/
        //readPPMFile(files[i], ppms[i]);
    }
    vector<float> data(N * INPUT_CHANNELS * INPUT_H * INPUT_W);
    
    for (int i = 0, volImg = INPUT_CHANNELS * INPUT_H * INPUT_W; i < N; ++i)
    {
        for (int c = 0; c < INPUT_CHANNELS; ++c)
        {
            for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j) {
                data[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(ppms[i].buffer[j * INPUT_CHANNELS + c])-1.0;
            }
        }
    }
    std::cout << " Data Size  " << data.size() << std::endl;
    std::cout<<"Data prepared..."<<endl;
    
    std::cout << "Model deserializing" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size());
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    
    /*data should be flattened*/
    doInference(*context, &data[0], &networkOut[0], N);
    std::cout<<"Inference uploaded..."<<endl;
    
    for(int i = 0; i < N; ++i)
    {
        float* outdata = &networkOut[0] + i * INPUT_W * INPUT_H * INPUT_CHANNELS;
        Mat position_map(INPUT_W, INPUT_H, CV_64FC3, outdata,CV_AUTOSTEP);
        //cv::imwrite(save_path + ppm[i].fileName, position_map);
        /*
        ppm = ppms[i];
        string filename = ppm.fileName;
        std::ofstream outfile(save_path + filename, std::ofstream::binary);
        assert(!outfile.fail());
        outfile << "Output save..." << "\n" << ppm.w << " " << ppm.h << "\n" << ppm.max << "\n";
        outfile.write(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
        */
    }
    
    /* we need to keep the memory created by the parser */
    parser->destroy();
    
    execute(*engine);
    engine->destroy();
    shutdownProtobufLibrary();
}
