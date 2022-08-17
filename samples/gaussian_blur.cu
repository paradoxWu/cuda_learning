#include<iostream>
#include<string>
#include<cmath>
#include <chrono>
#include <thread>
#include <vector>

#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#ifndef WITHOUT_CV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
using namespace cv;
using namespace std;

static const double pi = 3.1415926;

__global__ 
void gaussian_kernel(uchar *d_img_in, uchar *d_img_out, double *d_arr,
                                const int img_cols, const int img_rows, const int size)
{
    const int col_id = blockIdx.x * blockDim.x + threadIdx.x;    //col
    const int row_id = blockIdx.y * blockDim.y + threadIdx.y;  //row
    
    int board = size/2;

    double sum =0.0;
    int index = 0;
    if (col_id > board && col_id < img_cols - board && row_id>board &&row_id < img_rows - board)
    {            
        for (int n = row_id - board; n < row_id + board +1; n++)
        {
            for (int m = col_id - board; m < col_id + board +1; m++)
            {
                sum += d_arr[index++] * d_img_in[n*img_cols+m];
            }
        }
    }
    if(sum>255.0){
        d_img_out[col_id + row_id * img_cols] = static_cast<uchar>(255);
    }
    else if(sum<0.0){
        d_img_out[col_id + row_id * img_cols] = static_cast<uchar>(0);
    }
    else{
        d_img_out[col_id + row_id * img_cols] = static_cast<uchar>(sum);
    }

}

void gaussian_cuda(const Mat &img_in, Mat &img_out, const int &size, const double &sigma)
{
    const int img_sizeof = img_in.cols*img_in.rows * sizeof(uchar);
    const int arr_sizeof = size * size * sizeof(double);
    img_out = Mat::zeros(img_in.size(), CV_8UC1);

    double *arr = (double*)malloc(size*size * sizeof(double));
    auto getGassianArray = [&]()
    {
        double sum = 0.0;
        auto sigma_2 = sigma * sigma;
        int center = size / 2; 

        for (int i = 0; i < size; ++i)
        {
            auto dx_2 = pow(i - center, 2);
            for (int j = 0; j < size; ++j)
            {
                auto dy_2 = pow(j - center, 2);
                double g = exp(-(dx_2 + dy_2) / (2 * sigma_2));
                g /= 2 * pi * sigma;
                arr[i * size + j] = g;
                sum += g;
            }
        }
        //归一化，卷积核，窗内和必须为1，保证原图的总值强度不变
        for (size_t i = 0; i < size; ++i)
        {
            for (size_t j = 0; j < size; ++j)
            {
                arr[i * size + j] /= sum;
            }
        }
    };
    getGassianArray();

    double *d_arr;		//之后做成共享内存
    uchar *d_img_in;
    uchar *d_img_out;
    cudaMalloc(&d_arr, arr_sizeof);
    cudaMalloc(&d_img_in,img_sizeof);
    cudaMalloc(&d_img_out,img_sizeof);

    cudaMemcpy(d_arr, arr, arr_sizeof, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_in, img_in.data, img_sizeof, cudaMemcpyHostToDevice);

    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,&blockSize,gaussian_kernel,0,100000);
    int block_size = sqrt(blockSize);
    std::cout<<"suitable block size:"<<block_size<<std::endl;

    int BLOCKDIM_X=block_size,BLOCKDIM_Y=block_size;
    dim3 Block_G (BLOCKDIM_X, BLOCKDIM_Y);
    dim3 Grid_G ((uint)ceil((double)img_in.cols / BLOCKDIM_X),(uint)ceil((double)img_in.rows/BLOCKDIM_Y));

    std::chrono::time_point<std::chrono::system_clock> start_time(std::chrono::system_clock::now());
    gaussian_kernel <<< Grid_G, Block_G >>>(d_img_in, d_img_out, d_arr, img_in.cols, img_in.rows, size);
    auto duration =std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_time);
    std::cout<<"cost by GPU:"<<duration.count()<<"ms"<<std::endl;

    cudaMemcpy(img_out.data, d_img_out, img_sizeof, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    free(arr);
}

// @param[in]: src_img, size,simga:(gaussian window size and sigma value)
// @param[out]: dst_img
void gaussian_thread(const cv::Mat &src_img, cv::Mat &dst_img,const size_t &size,const double &sigma)
{
    dst_img = cv::Mat::zeros(src_img.size(),CV_8UC1);
    double arr[size*size];
    const auto size_2 = size>>1;
    const auto max_thread = std::thread::hardware_concurrency();
    std::vector<std::thread> thread_bar;
    const auto t_rows = src_img.rows / (max_thread);
    //initialize the gaussian window
    auto getGassianArray = [&]()
    {
        double sum = 0.0;
        auto sigma_2 = sigma * sigma;
        int center = size / 2; 

        for (int i = 0; i < size; ++i)
        {
            auto dx_2 = pow(i - center, 2);
            for (int j = 0; j < size; ++j)
            {
                auto dy_2 = pow(j - center, 2);
                double g = exp(-(dx_2 + dy_2) / (2 * sigma_2));
                g /= 2 * pi * sigma;
                arr[i * size + j] = g;
                sum += g;
            }
        }
        //归一化，卷积核，窗内和必须为1，保证原图的总值强度不变
        for (size_t i = 0; i < size; ++i)
        {
            for (size_t j = 0; j < size; ++j)
            {
                arr[i * size + j] /= sum;
            }
        }
    };
    getGassianArray();
#ifdef DEBUG
    for(size_t i = 0; i < size; i++){
        for(size_t j = 0; j < size; j++){
            cout<<arr[i*size+j]<<" ";
        }
        cout<<endl;
    }
#endif

    auto compGassion_thread = [&](const int thread_id){
        for(auto i{ t_rows * (thread_id - 1)}; i < t_rows *thread_id; ++i)
        {
            auto out_p = &dst_img.data[i * src_img.cols];
            for(auto j{size_2}; j < src_img.cols - size_2; ++j)
            {
                double sum = 0.0;
                for(size_t y = 0; y < size; ++y)
                {
                    auto in_p = &src_img.data[(i+y) * src_img.cols + j];
                    for(size_t x = 0; x < size; ++x)
                    {
                        sum += *(in_p + x) * arr[x * size + y];
                    }
                }
                *(out_p + j)=static_cast<char>(sum);
            }
        }
    };

    std::chrono::time_point<std::chrono::system_clock> start_time(std::chrono::system_clock::now());
    for(int thread_id = 1; thread_id <= max_thread; ++thread_id){
        thread_bar.emplace_back(compGassion_thread, thread_id);
    }
    for(auto &i : thread_bar){
        i.join();
    }
    auto duration =std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_time);
    std::cout<<"cost by CPU_Concurrency:"<<duration.count()<<"ms"<<std::endl;
}


int main(int argc, char *argv[])
{
    double sigma = atof(argv[1]);
    int window_size = atoi(argv[2]);
    auto img = imread("images/Lenna.jpeg");
    Mat img_gray;
    cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    Mat gaussian;
    gaussian_cuda(img_gray, gaussian, window_size, sigma);
    gaussian_thread(img_gray, gaussian,window_size,sigma);
    string save_name = "images/gaussian_cuda"+to_string(window_size)+to_string(sigma)+".jpg";
    imwrite(save_name, gaussian);
}
