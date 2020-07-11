#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <NvInferPlugin.h>
#include <NvInfer.h>
#include <NvCaffeParser.h>

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
  const char * proto = "deploy.trt.prototxt";
  const char * caffemodel = "res10_300x300_ssd_iter_140000_fp16.caffemodel";
  const char * serialized = "serialized.bin";
  const char * outputs[]{"keep_count","detection_out"};
  auto force_use_fp16 = true;
  int maxBatchSize = 8;
  int workSpaceSize = 1073741824;

  cout<<"Serialize"<<endl;
  initLibNvInferPlugins(&logger, "");
  nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
  nvinfer1::INetworkDefinition *network = builder->createNetwork();
  nvcaffeparser1::ICaffeParser *parser = nvcaffeparser1::createCaffeParser();
  nvinfer1::DataType modelDataType = force_use_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
  const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor = 
		parser->parse(proto, caffemodel, *network, modelDataType);

  for (auto &s : outputs)
    network->markOutput(*blobNameToTensor->find(s));

  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(workSpaceSize);

  auto engine = builder->buildCudaEngine(*network);

  {
    auto hostMemory = engine->serialize();
    std::ofstream out(serialized,std::ios::binary);
    out.write((const char *)hostMemory->data(), hostMemory->size());    
    hostMemory->destroy();
  }
  network->destroy();
  parser->destroy();
  builder->destroy();	
  cout<<"Serialize Done"<<endl;
}
