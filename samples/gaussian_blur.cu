
#include<iostream>
#include<string>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cmath>

#ifndef WITHOUT_CV
#include<opencv2/opencv.hpp>
#endif
using namespace cv;
using namespace std;

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

void gaussian_cuda(const Mat &img_in, Mat &img_out, const int &size, const double &sigma, int block_size = 16)
{
    bool ifdebug = true;

    const int img_sizeof = img_in.cols*img_in.rows * sizeof(uchar);
    const int arr_sizeof = size * size * sizeof(double);
    img_out = Mat::zeros(img_in.size(), CV_8UC1);

    double *arr = (double*)malloc(size*size * sizeof(double));
    auto getGuassionArray = [&]()
    {
        double sum = 0.0;
        auto sigma_2 = sigma * sigma;
        for (int i{}; i < size; ++i)
        {
            auto dx = i - size;
            for (int j{}; j < size; ++j)
            {
                auto dy = j - size;
                arr[i * size + j] = exp(-(dx*dx + dy * dy) / (sigma_2 * 2));
                sum += arr[i * size + j];
            }
        }
        //归一化，卷积核，窗内和必须为1，保证原图的总值强度不变
        for (size_t i{}; i < size; ++i)
        {
            for (size_t j{}; j < size; ++j)
            {
                arr[i * size + j] /= sum;
            }
        }
    };
    getGuassionArray();

    if(ifdebug){
        double sum = 0.0;
        for (int i{}; i < size; ++i)
        {
            for (int j{}; j < size; ++j){
                cout << arr[j + i * size] << " ";
                sum += arr[j + i * size];
            }
            cout << endl;
        }
        cout<<"gaussian_template_sum:"<<sum<<endl;
    }

    double *d_arr;		//之后做成共享内存
    uchar *d_img_in;
    uchar *d_img_out;
    cudaMalloc(&d_arr, arr_sizeof);
    cudaMalloc(&d_img_in,img_sizeof);
    cudaMalloc(&d_img_out,img_sizeof);

    cudaMemcpy(d_arr, arr, arr_sizeof, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_in, img_in.data, img_sizeof, cudaMemcpyHostToDevice);

    // if(ifdebug){
    //     cout<<"img_in "<<img_in<<endl;
    // }


    int BLOCKDIM_X=32,BLOCKDIM_Y=32;
    dim3 Block_G (BLOCKDIM_X, BLOCKDIM_Y);
    dim3 Grid_G ((uint)ceil((double)img_in.cols / BLOCKDIM_X),(uint)ceil((double)img_in.rows/BLOCKDIM_Y));
    // cout<<Grid_G.x<<Grid_G.y<<endl;

    gaussian_kernel <<< Grid_G, Block_G >>>(d_img_in, d_img_out, d_arr, img_in.cols, img_in.rows, size);

    cudaMemcpy(img_out.data, d_img_out, img_sizeof, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_img_in);
    cudaFree(d_img_out);

    // if(ifdebug){
    //     cout<<"img_out "<<img_out<<endl;
    // }

    free(arr);
}

int main(int argc, char *argv[])
{
    int sigma = atoi(argv[1]);
    int window_size = atoi(argv[2]);
    auto img = imread("images/sample.jpg");
    Mat img_gray;
    cvtColor(img, img_gray, CV_BGR2GRAY);
    imwrite("images/sample_gray.jpg", img_gray);
    // auto img2 {Mat::zeros(33,33, CV_8UC1)};
    Mat gaussian;
    gaussian_cuda(img_gray, gaussian, window_size, sigma);
    string save_name = "images/gaussian_cuda"+to_string(window_size)+to_string(sigma)+".jpg";
    imwrite(save_name, gaussian);

}