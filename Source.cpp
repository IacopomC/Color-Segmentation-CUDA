#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void colorTransfCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int dimX, int dimY, int x, int y);

struct MouseParams
{
    cv::cuda::GpuMat d_result;
    cv::cuda::GpuMat d_img;
};

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        MouseParams* mp = (MouseParams*)userdata;

        colorTransfCUDA(mp->d_img, mp->d_result, 32, 32, x, y);

        cv::imshow("Processed Image", mp->d_result);
    }
}

int main(int argc, char** argv)
{
    cv::namedWindow("Equirectangular Image", cv::WINDOW_OPENGL);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL);

    cv::Mat_<cv::Vec3b> h_img = cv::imread(argv[1]);
    cv::cuda::GpuMat d_result;
    cv::cuda::GpuMat d_img;

    cv::Vec3b pixel = h_img(2131, 4990);
    cout << "Pixel " << pixel << endl;


    cv::Size s = h_img.size();
    float height = s.height;
    float width = s.width;

    h_img = h_img(cv::Range(0, height / 2), cv::Range(0, width)).clone();

    d_img.upload(h_img);
    d_result.upload(h_img);

    MouseParams mp;

    mp.d_result = d_result;
    mp.d_img = d_img;

    //set the callback function for any mouse event
    cv::setMouseCallback("Equirectangular Image", CallBackFunc, (void*)&mp);

    cv::imshow("Equirectangular Image", h_img);
    cv::imshow("Processed Image", d_result);

    cv::waitKey();
    return 0;
}