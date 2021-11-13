#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
 
using namespace cv;
using namespace std;


///                         Procesamiento de imágenes en c++
///
///                            Jorge Robles – 614192010
///                        Fundación Universitaria Konrad Lorenz
///
///     En el siguiente trabajo se presenta un grupo de algoritmos aplicados en el procesamiento
///     de imágenes, los cuales se basan en múltiples aplicaciones de métodos de Gauus y la librería OpenCv, tales como:
///
///     •    Filtro  -  Gray
///     •    Filtro  - Blur
///     •    Filtro  -  Canny
///     •    Filtro – Dilation

vector<double> get_guassian_kernel(const int kernel_size, double sigma);
void blur_gaussian(const cv::Mat &input, cv::Mat &output, const int kernel_size, const double sigma);
void initImg(string path, Mat *img);
void getInfoImg(Mat img, string path);
void filterGrayImg(Mat img, Mat *imgGray);
void filterBlurImg(Mat img, Mat *imgBlur);
void filterCannyImg(Mat img, Mat *imgCanny, double umbral1, double umbral2);
void filterDilationImg(Mat img, Mat *imgDil);

int main() {
    
    Mat img, imgGray, imgBlur, imgCanny, imgDil, imgInv;
    initImg("Resources/prueba.png", &img);
    
    filterGrayImg(img, &imgGray);
    filterBlurImg(imgGray, &imgBlur);
    filterCannyImg(imgGray, &imgCanny, 25,75);
    filterDilationImg(imgCanny, &imgDil);

    imshow("Image Dilation", imgDil);
    imshow("Filtro Canny", imgCanny);
    imshow("Filtro Blur", imgBlur);
    imshow("Filtro Gris", imgGray);
    imshow("Original", img);
    waitKey(0);

}

vector<double> get_guassian_kernel(const int kernel_size, double sigma) {

    std::vector<double> kernel(kernel_size*kernel_size,0);
    
    if (sigma <=0 ){
        sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8;
    }

    // sum is for normalization
    double sum = 0.0;

    // generating nxn kernel
    int i,j;
    double mean = kernel_size/2;
    for (i=0 ; i<kernel_size ; i++) {
        for (j=0 ; j<kernel_size ; j++) {
            kernel[(i*kernel_size)+j] =exp( -0.5 * (pow((i-mean)/sigma, 2.0) + pow((j-mean)/sigma,2.0)) )
            / (2 * M_PI * sigma * sigma);
            sum += kernel[(i*kernel_size)+j];
        }
    }

    // normalising the Kernel
    for (int i = 0; i < kernel.size(); ++i){
        kernel[i] /= sum;
    }

    return kernel;
}

void initImg(string path, Mat *img) {
    *img = imread(path);
    getInfoImg(*img, path);
}

void getInfoImg(Mat img, string path) {
    cout << path << endl;
    cout << img.size << endl;
}

void filterGrayImg(Mat img, Mat *imgGray) {
    Mat _img;
    img.copyTo(_img);
    int i = 0;
    for (int j = 0; j < img.cols; j++) {
        if(_img.at<int>(i,j) < 123) {
            _img.at<int>(i,j) = _img.at<int>(i,j)/2;
        } else {
            _img.at<int>(i,j) += (255 - _img.at<int>(i,j)) /2;
        }
    }
    cvtColor(img, *imgGray, COLOR_BGR2GRAY);
}


void filterBlurImg(Mat img, Mat *imgBlur) {
    int red = 0;
    int green = 0;
    int blue = 0;
    int counter = 0;
    int i = 0;
    Mat _img;
    img.copyTo(_img);
    for (int j = 0; j < _img.cols; j++) {
        if (i + 1 && j - 1) {
            red += _img.at<int>(i,j) + _img.at<int>(i + 1, j - 1);
            counter++;
        }
        if (j + 1) {
            green += _img.at<int>(i,j) + _img.at<int>(i, j + 1);
            counter++;
        }
        if (i + 1 && j + 1) {
            blue += _img.at<int>(i,j) + _img.at<int>(i + 1, j + 1);
            counter++;
        }
        if (i + 1) {
            red += _img.at<int>(i,j) + _img.at<int>(i + 1, j);
            counter++;
        }
        if (j - 1) {
            green = _img.at<int>(i,j) + _img.at<int>(i, j - 1);
            counter++;
        }
        if (i - 1) {
            blue = _img.at<int>(i,j) + _img.at<int>(i - 1, j);
            counter++;
        }
        _img.at<int>(i,j) = (red + green + blue) /counter;
    }
    GaussianBlur(img, *imgBlur, Size(0, 0), 9, 0);
}



void filterCannyImg(Mat img, Mat *imgCanny, double umbral1, double umbral2) {
    Mat _img;
    img.copyTo(_img);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if((255 - _img.at<int>(0,i)) < umbral1) {
                _img.at<int>(0,i) = _img.at<int>(0,i) * 1/umbral2;
            }if((255 - _img.at<int>(0,i)) < umbral2) {
                _img.at<int>(0,i) = _img.at<int>(0,i) * 1/umbral2;
            }
        }
    }
    Canny(img, *imgCanny, umbral1,umbral2);
}

void filterDilationImg(Mat img, Mat *imgDil) {
    Mat _img;
    img.copyTo(_img);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if(_img.at<int>(0,i) > 250) {
                _img.at<int>(0,i) = 255;
                if(i + 1) {
                    _img.at<int>(0,i + 1) = 255;
                }
            }
        }
    }
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(img, *imgDil, kernel);
}

void blur_gaussian(const cv::Mat &input, cv::Mat &output, const int kernel_size, const double sigma) {
    std::vector<double> kernel = get_guassian_kernel(kernel_size,sigma);

    CV_Assert(input.channels() == output.channels());

    unsigned char *data_in = (unsigned char*)(input.data);
    unsigned char *data_out = (unsigned char*)(output.data);

    for (int row = 0; row < input.rows; row++) {
        for (int col = 0;col < (input.cols * input.channels()); col += input.channels()) {
            for (int channel_index = 0; channel_index < input.channels(); channel_index++) {

                if (row <= kernel_size/2 || row >= input.rows-kernel_size/2 ||
                    (input.cols * input.channels()) <= kernel_size/2||
                    col >= (input.cols * input.channels())-kernel_size/2){
                    data_out[output.step * row + col + channel_index] = data_in[output.step * row + col + channel_index];
                    continue;
                }

                int k_ind = 0;
                double sum = 0 ;
                for (int k_row = -kernel_size / 2; k_row <= kernel_size / 2; ++k_row) {
                    for (int k_col = -kernel_size / 2; k_col <= kernel_size / 2; ++k_col) {
                        sum += kernel[k_ind] * (data_in[input.step * (row + k_row) + col + (k_col*input.channels()) + channel_index]/255.0);
                        k_ind++;
                    }
                }
                data_out[output.step * row + col + channel_index] = (unsigned int) (std::max(std::min(sum, 1.0), 0.0) * 255.0);
            }
        }
    }
}
