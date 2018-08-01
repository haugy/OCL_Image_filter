# OCL_Image_filter
本代码只是完成使用gpu构成简单的图像滤波处理
环境要求：opencv3.opencl
主要文件包含：
README.md、CMakeLists.txt、ocl_filter.cpp、convkernel.cl
系统环境：Ubuntu16.04
* * *
##opencl简介
OpenCL程序是分成两部分的：一部分是在设备上执行的（对于我们，是GPU），另一部分是在主机上运行的（对于我们，是CPU）。为了能在设备上执行代码，程序员需要写一个特殊的函数（kernel函数）。这个函数需要使用OpenCL语言编写。OpenCL语言采用了C语言的一部分加上一些约束、关键字和数据类型。
* * *
### 设备（Device）
Kernel：你可以把它想像成一个可以在设备上执行的函数。当然也会有其他可以在设备上执行的函数，但是他们之间是有一些区别的。Kernel是设备程序执行的入口点。换言之，Kernel是唯一可以从主机上调用执行的函数。
现在的问题是：我们如何来编写一个Kernel？在Kernel中如何表达并行性？它的执行模型是怎样的？解决这些问题，我们需要引入下面的概念：

SIMT：单指令多线程（SINGLE INSTRUCTION MULTI THREAD）的简写。就像这名字一样，相同的代码在不同线程中并行执行，每个线程使用不同的数据来执行同一段代码。

Work-item（工作项）：Work-item与CUDA Threads是一样的，是最小的执行单元。每次一个Kernel开始执行，很多（程序员定义数量）的Work-item就开始运行，每个都执行同样的代码。每个work-item有一个ID，这个ID在kernel中是可以访问的，每个运行在work-item上的kernel通过这个ID来找出work-item需要处理的数据。

Work-group（工作组）：work-group的存在是为了允许work-item之间的通信和协作。它反映出work-item的组织形式（work-group是以N维网格形式组织的，N=1，2或3）。

Work-group等价于CUDA thread blocks。像work-items一样，work-groups也有一个kernel可以读取的唯一的ID。

ND-Range:ND-Range是下一个组织级别，定义了work-group的组织形式（ND-Rang以N维网格形式组织的，N=1，2或3）。
* * *
__kernel函数示例：
```cpp
__kernel void vector_add_gpu (__global const float* src_a,
                     __global const float* src_b,
                     __global float* res,
           const int num)
{
   /* get_global_id(0) 返回正在执行的这个线程的ID。 
   许多线程会在同一时间开始执行同一个kernel，
   每个线程都会收到一个不同的ID，所以必然会执行一个不同的计算。*/
   const int idx = get_global_id(0);

   /* 每个work-item都会检查自己的id是否在向量数组的区间内。
   如果在，work-item就会执行相应的计算。*/
   if (idx < num)
      res[idx] = src_a[idx] + src_b[idx];
}
```
有一些需要注意的地方：
1. Kernel关键字定义了一个函数是kernel函数。Kernel函数必须返回void。
2. Global关键字位于参数前面。它定义了参数内存的存放位置。
* * *
API函数clGetPlatformIDs()用来获取指定系统上可用的计算平台：
```cpp
cl_int clGetPlatformIDs (cl_uint num_entries,
               cl_platform_id *platforms,
               cl uint *num_platforms)
```
clGetPlatformIDs()通常由应用程序调用两次。首次调用时，将unsigned int指针和NULL分别传递给num_platforms和platforms参数，编程人员可以分配空间来容纳该平台的信息。第二次调用时，将cl_platform_ id指针传递给已为num- entries平台分配足够空间的具体实现。平台被发现之后，调用clGetPlatformInfo()来确定具体使用哪一个已定义的实现（厂商）平台。
* * *
函数clGetDeviceIDs()的调用过程类似于clGetPlatformIDs()。它也需要3步，区别在于额外还有平台和设备类型参数。device_type参数将设备限制为仅有GPU(CL_DEVICE_TYPE_GPU)、仅CPU（CL—DEVICE—TYPE—CPU）、所有设备(CL DEVICE TYPE_ALL)和其他选项。
```cpp
cl_int clGetDeviceIDs (cl_platform_id platform,
             cl_device_type device_type,
             cl_uint num entries,
             cl_device id *devices,
             cl_uint *num devices)
```
* * *
上下文(context)是一个抽象容器并存在于主机端。它能协调主机．设备之间的交互机制，管理设备上可用的内存对象，跟踪针对每个设备新建的kemel和程序。
新建上下文的API函数是clCreateContext()。参数properties用来限定上下文的范围。它可提供指定的具体硬件平台，开启OpenGL/OpenCL的互操作性或者开启其他着眼于未来的参数。将上下文局限于某个特定的平台使编程人员为多个平台提供上下文，并且充分利用由多个厂商资源组成的系统。随后，编程人员必须提供希望与上下文关联的设备数量和设备ID。OpenCL允许在新建上下文时提供用户回调函数，这些函数用于报告可能在其整个生命周期中产生的额外错误信息。

* * *
通过提交命令到命令队列(command queue)来开始与设备进行通信。命令队列是主机端用于向设备端发送请求的行为机制。一旦主机端指定运行kernel的设备并且上下文已新建，那么每个设备必须新建一个命令队列（即每个命令队列只关联一个设备）。

* * *
