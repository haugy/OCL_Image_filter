/*===========================================
*    
*    ocl convolution for image 
*    
*    Created by Haugy on 25/07/2018
*    Copyright (c) 2018 Haugy. All right reserved.
*    
*===========================================*/


#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace cv;

/*选择平台并创建上下文
 * 输出：cl_context
 * 输入：无
 */
cl_context Createdcontext() {
  
    cl_int status;
    cl_uint numPlatforms = 0;
    cl_platform_id platforms = NULL;
    //cl_uint numDevices = 0;
    //cl_device_id *devices = NULL;
    cl_context context = NULL;

    //选择可用的平台中的一个
    status = clGetPlatformIDs(1, &platforms, &numPlatforms);
    //给每一个platform申请内存空间--delete
    //platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
    //Fill in the platform
    //status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (status != CL_SUCCESS || numPlatforms <= 0) {
	std::cerr << "Failed to find any OpenCL platforms." << std::endl;
	return NULL;
    }
    
    /************delete
    //选择devices
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    //申请空间
    devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
    //Fill in the devices
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    if (status != CL_SUCCESS || numDevices <= 0) {
	std::cerr << "Failed to find any OpenCL devices." << std::endl;
	return NULL;
    }
    *************/
    
    //创建一个OpenCL上下文环境
    cl_context_properties contextProperties[] = {
	CL_CONTEXT_PLATFORM,
	(cl_context_properties)platforms,
	0
    };
    
    
    //基于platform和device创建context
    //context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
	std::cerr << "Failed to Create OpenCL context." << std::endl;
	return NULL;
    }
    return context;
}

/*创建设备并创建命令队列
 * 输出：cl_command_queue
 * @param:cl_context
 * @param:cl_device_id
 */
cl_command_queue CreateCmdQueue(cl_context context, cl_device_id *device) {
    
    cl_int status;
    cl_device_id *devices;
    cl_command_queue cmdQueue = NULL;
    size_t deviceBufferSize = -1;
 
    // 获取设备缓冲区大小
    status= clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
 
    if (deviceBufferSize <= 0) {
	std::cerr << "No devices available.";
	return NULL;
    }
    
    // 为设备分配缓存空间
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    status= clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    
    //选取可用设备中的第一个
    cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
    if (status != CL_SUCCESS) {
	std::cerr << "Failed to Create OpenCL commandqueue." << std::endl;
	return NULL;
    }
    
    //获取devices[0]地址
    *device = devices[0];
    delete[] devices;
    
    return cmdQueue;

}

/*创建构建程序对象
 * Output:cl_program
 * @param:cl_context
 * @param:cl_device_id
 * @param:filename of kernel function
 */
cl_program CreateProgram(cl_context context, cl_device_id device, const char* filename) {

    cl_int status;
    cl_program program;
    
    std::ifstream kernelfile(filename, std::ios::in);
    if (!kernelfile.is_open()) {
	std::cerr << "Failed to open file for reading: " << filename << std::endl;
	return NULL;
    }
    
    std::ostringstream oss;
    oss << kernelfile.rdbuf();
    
    std::string srcStdStr = oss.str();
    const char* programSrc = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1, (const char**)&programSrc, NULL, &status);
    
    //build (compile) the program for device
    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
	std::cerr << "Failed to OpenCL CreateProgram." << std::endl;
	char errbuf[0x10000];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000,errbuf,NULL);
	std::cout<<errbuf<<std::endl;
	return NULL;
    }
    
    return program;
    
}

//
int main(int argc, char** argv) {
    
    //opencl
    cl_int status;
    cl_context context;
    cl_device_id device = 0;
    cl_command_queue cmdQueue = 0;
    cl_program program = 0;
    cl_kernel kernel = 0;
 
    
    //opemcv
    Mat img;
    Mat gray;
    const int filterSize = 3;
    size_t buffiltersize = filterSize * filterSize * sizeof(int);
    
    //opencl初始化
    context = Createdcontext();
    cmdQueue = CreateCmdQueue(context, &device);
    program = CreateProgram(context, device, "../convkernel.cl");
    
    
    //存放图片数据
    //uchar* bufInput = NULL;
    uchar* bufOutput = NULL;
    int filterdata[9] = {-1, 0, 1, -2, 0 , 2, -1, 0, 1};
    
    //获取图片
    img = imread(argv[1]);
    imshow("src_img", img);
    cvtColor(img, gray, CV_BGR2GRAY);
    imshow("gray_image", gray);
    //waitKey(0);
    
    //获取cl文件名
    //const char* clfilename = argv[1];
    
    //图片数据大小
    const int width = gray.cols;
    const int height = gray.rows;
    
    //dst image
    Mat dst = Mat::zeros(height, width, CV_8U);
    
    //申请内存空间
    //bufInput = (uchar*)malloc(width * height * sizeof(uchar));
    //bufOutput = (uchar*)malloc(width * height * sizeof(uchar));
        
    //拷贝图片数据到input buffer,设置output buffer值
    //memcpy(bufInput, gray.data, width * height * sizeof(uchar));
    //memset(bufOutput, 0x0, width * height * sizeof(uchar));
    
    /*create image buffer*/
    //descriptor initializes a 2D image with no pitch
    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width;
    desc.image_height = height;
    desc.image_depth = 0;
    desc.image_array_size = 0;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = NULL;
    
    //image format 描述了每个像素点的属性
    cl_image_format format;
    format.image_channel_order = CL_R;//单通道
    format.image_channel_data_type = CL_UNSIGNED_INT8;//6位无符号int
    
    cl_mem bufferInputImage = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, NULL, &status);
    cl_mem bufferOutputImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &status);
    cl_mem bufferFilter = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, buffiltersize, filterdata, &status);
    if (status != CL_SUCCESS) {
	std::cerr << "Failed to OpenCL CreateBuffer." << std::endl;
	return NULL;
    }  
    
    size_t origin[3] = {0, 0, 0};//offset  
    size_t region[3] = {width, height, 1};//size of image

    status = clEnqueueWriteImage(cmdQueue, bufferInputImage, CL_FALSE, origin, region, 0, 0, gray.data, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
	std::cerr << "Failed to OpenCL EnqueueWriteImage." << std::endl;
	return NULL;
    }
    //采样器描述如何访问image的对象
    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &status);
    
    kernel = clCreateKernel(program, "convolution", &status);
    if (status != CL_SUCCESS) {
	std::cerr << "Failed to OpenCL CreateKernel." << std::endl;
	return -1;
    }
    /*set args
     * @param src image
     * @param dst image
     * @param width
     * @param height
     * @param filter kernel
     * @param filter kernel size
     * @param sampler
     */
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferInputImage);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferOutputImage);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferFilter);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &filterSize);
    status |= clSetKernelArg(kernel, 6, sizeof(cl_sampler), &sampler);
    
    if (status != CL_SUCCESS) {
	std::cerr << "Failed to OpenCL SetKernelArg." << std::endl;
	return -1;
    }
    
    size_t local_ws[2] = {1, 1};
    size_t global_ws[2] = {width, height};
    
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, global_ws, local_ws, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
	std::cerr << "Failed to OpenCL EnqueueNDRangeKernel." << std::endl;
	return -1;
    }    
    //内存从gpu拷贝到主机
    status = clEnqueueReadImage(cmdQueue, bufferOutputImage, CL_TRUE, origin, region, 0, 0, dst.data, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
	std::cerr << "Failed to OpenCL EnqueueReadImage." << std::endl;
	return -1;
    }
    //memcpy(dst.data, bufOutput, width * height * sizeof(uchar));
    //std::cout << dst << std::endl;
    
    //release
    clReleaseContext(context);
    clReleaseCommandQueue(cmdQueue);
    clReleaseKernel(kernel);
    clReleaseSampler(sampler);
    //clReleaseEvent(status);
    clReleaseMemObject(bufferFilter);
    clReleaseMemObject(bufferInputImage);
    clReleaseMemObject(bufferOutputImage);
    
    
    imshow("sobel", dst);
    waitKey(0);
    
    //Mat imggrad;
    //Sobel(gray, imggrad, 0, 1, 0, 3);
    //Mat imgsobel;
    //convertScaleAbs(imggrad, imgsobel);
    //imshow("cvsobel", imgsobel);
    //waitKey();
    
    return 0;
}


