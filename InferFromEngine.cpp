#include <iostream>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>
#include <NvInferPlugin.h>
#include <NvInfer.h>

#include <opencv2/opencv.hpp>

using namespace std;

class Logger : public nvinfer1::ILogger 
{
public:
	void log(nvinfer1::ILogger::Severity severity, const char *msg) override
	{
		switch(severity)
		{
			case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
				cout<<"INTERNAL:"<<msg<<endl;
				break;
			case nvinfer1::ILogger::Severity::kERROR:
				cout<<"ERROR:"<<msg<<endl;
				break;				
			case nvinfer1::ILogger::Severity::kWARNING:
				cout<<"WARNING:"<<msg<<endl;
				break;					
			case nvinfer1::ILogger::Severity::kINFO:
				cout<<"INFO:"<<msg<<endl;
				break;			
			case nvinfer1::ILogger::Severity::kVERBOSE:
				cout<<"VERBOSE:"<<msg<<endl;
				break;				
		}		
	}
};

static Logger logger;

int main()
{
  const char * serialized = "serialized.bin";
  const char * imageFileName = "image.jpg";
  int maxBatchSize = 1;

  nvinfer1::IExecutionContext *context;
  nvinfer1::ICudaEngine*engine;	
  initLibNvInferPlugins(&logger, "");
  cout<<"Deserialize"<<endl;
  std::ifstream in(serialized, std::ios::binary | std::ios::ate);
  if(in)
  {
    std::streamsize size = in.tellg();
    in.seekg(0,std::ios::beg);
    std::vector<char> buffer(size);
    if(in.read(buffer.data(),size))
    {
      nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
      engine = runtime->deserializeCudaEngine((const void *)buffer.data(), (std::size_t)size, nullptr);		   
      context = engine->createExecutionContext();				
    }    
  }

  cout<<"Deserialize Done"<<endl;
 
  std::vector<void*> buffers;
  std::vector<int> binding_element_size;
  {
    int numberOfBuffers = engine->getNbBindings();
    binding_element_size = std::vector<int>(numberOfBuffers);
    buffers = std::vector<void*>(numberOfBuffers);
    int element_size;
    for(int i = 0; i < numberOfBuffers; ++i)
    {
      auto dims = engine->getBindingDimensions(i);
      auto type = engine->getBindingDataType(i);
      switch(type)
      {
        case nvinfer1::DataType::kFLOAT:
          element_size = 4;
          break;
        case nvinfer1::DataType::kINT8:
          element_size = 1;
          break;
      }
      int binding_elements = 1;
      for(int j = 0; j < dims.nbDims; ++j)
      {
        binding_elements *= dims.d[j];
      }
      binding_element_size[i] = binding_elements * element_size;
      cudaMalloc(&buffers[i], binding_element_size[i] * maxBatchSize);
    }
  }

  const char * inputName = "data";
  const char * outputName = "detection_out";
  constexpr int WIDTH    = 300;
  constexpr int HEIGHT   = 300;
  constexpr int CHANNELS =   3;	

  constexpr int MAX_OBJECTS = 200;
  constexpr int OBJECT_ITEMS = 7;

  int input_index = engine->getBindingIndex("data");
  int output_index = engine->getBindingIndex("detection_out");

  if(binding_element_size[input_index] == WIDTH * HEIGHT * CHANNELS * sizeof(float)) 
  {
    cout<<"Input correct size"<<endl;
  }
  else
  {
    cout<<"Wrong input size!"<<endl;
    return -1;
  }

  if(binding_element_size[output_index] == MAX_OBJECTS * OBJECT_ITEMS * sizeof(float)) 
  {
    cout<<"Output correct size"<<endl;
  }
  else
  {
    cout<<"Wrong output size!"<<endl;
    return -1;
  }

  if(binding_element_size[input_index] == WIDTH * HEIGHT * CHANNELS * sizeof(float)) 
  {
    cout<<"Input correct size"<<endl;
  }
  else
  {
    cout<<"Wrong input size!"<<endl;
    return -1;
  }

  cv::Mat image = cv::imread(imageFileName);
  cv::Size windowSize(WIDTH, HEIGHT);
  cv::Mat input8u;
  cv::resize(image, input8u, windowSize);
  cv::Mat input32f;
  input8u.convertTo(input32f,CV_32F);
  
  cv::Mat channels[3];
  split(input32f,channels);

  cv::Mat completeInput;
  completeInput.push_back(channels[0]);
  completeInput.push_back(channels[1]);
  completeInput.push_back(channels[2]);

  if(cudaMemcpy(buffers[input_index],completeInput.data,binding_element_size[input_index], cudaMemcpyHostToDevice) == cudaSuccess)
  {
    cout<<"Input successfully copied to CUDA memory"<<endl;
  }
  else
  {
    cout<<"Could not copy input to CUDA memory!"<<endl;
    return -1;
  }

  if(context->execute(1,&buffers[0])) 
  {
    cout<<"Inference success"<<endl;
  }
  else
  {
    cout<<"Inference failed!"<<endl;
    return -1;
  }

  cv::Mat detectionMat(MAX_OBJECTS, OBJECT_ITEMS, CV_32F);		
  if(cudaMemcpy(detectionMat.data, buffers[output_index],binding_element_size[output_index], cudaMemcpyDeviceToHost) == cudaSuccess)
  {
    cout<<"Output successfully copied from CUDA memory"<<endl;
  }
  else
  {
    cout<<"Could not copy output from CUDA memory!"<<endl;
    return -1;
  }

  int frameHeight = image.rows;
  int frameWidth = image.cols;
  float threshold = 0.3;
  cv::Scalar color(0, 0, 255);
  int thickness = frameWidth/250;
  for(int i = 0; i < detectionMat.rows; i++)
  {
    float confidence = detectionMat.at<float>(i, 2);
    if(confidence > threshold)
    {
      int x1 = detectionMat.at<float>(i, 3) * frameWidth;
      int y1 = detectionMat.at<float>(i, 4) * frameHeight;
      int x2 = detectionMat.at<float>(i, 5) * frameWidth;
      int y2 = detectionMat.at<float>(i, 6) * frameHeight;
      if(x1 < 0)
        x1 = 0;
      if(y1 < 0)
        y1 = 0;			
      if(x2 > frameWidth)
        x2 = frameWidth;
      if(y2 > frameHeight)
        y2 = frameHeight;	
      
      cv::Point tl(x1, y1);				
      cv::Point br(x2, y2);				
      cv::rectangle(image, tl, br, color, thickness);
    }		
  }
  cv::imwrite("results.jpg", image);
}
