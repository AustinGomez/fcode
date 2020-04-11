#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <math.h>
#include <queue>
#include <chrono>

using namespace cv;
using namespace std;

struct Block
{
    int startX;
    int startY;
    int size;
    Scalar a;
    Scalar r;

    Block(int startX, int startY, int size) : startX{startX}, startY{startY}, size{size} {}
};

float findParamsAndError(const Mat &rangeBlock, const Mat &domainBlock, const int errorThreshold, Scalar &a, Scalar &r)
{
    int minError = INT16_MAX;
    int error = minError;
    int bestA = -5;
    r = mean(rangeBlock);
    Scalar d_mean = mean(domainBlock);
    Mat D = domainBlock - d_mean;
    Mat R = rangeBlock - r;
    a = 0.5;
    //Mat D21 = domainBlock(Rect(0, 0, domainBlock.size().width, domainBlock.size().height / 2)) - d_mean;
    //Mat R21 = rangeBlock(Rect(0, 0, rangeBlock.size().width, rangeBlock.size().width / 2)) - r;
    //Mat D22 = domainBlock(Rect(0, rangeBlock.size().width / 2, domainBlock.size().width, domainBlock.size().height / 2)) - d_mean;
    //Mat R22 = rangeBlock(Rect(0, rangeBlock.size().width / 2, rangeBlock.size().width, rangeBlock.size().width / 2)) - r;

    for (int trialA = -8; trialA <= 8; ++trialA)
    {
        float scaledA = trialA * 0.125;
        //double norm1 = norm(scaledA * D21, R21);
        //double error1 = pow(norm1, 2);
        //if (error1 > errorThreshold) {
        //    continue;
        //}
        //error = pow(norm1 + norm(scaledA * D22, R22), 2);
        error = pow(norm(scaledA * D, R), 2);
        if (error < minError)
        {
            minError = error;
            a = scaledA;
        }
    }
    return minError;
}

Mat findReducedDomainBlock(const Mat &image, const Block &rangeBlock)
{
    int domainBlockSize = rangeBlock.size * 2;
    int domainStartXCoordinate = max(rangeBlock.startX - rangeBlock.size / 2, 0);
    int domainStartYCoordinate = max(rangeBlock.startY - rangeBlock.size / 2, 0);
    int domainEndXCoordinate = domainStartXCoordinate + domainBlockSize;
    int domainEndYCoordinate = domainStartYCoordinate + domainBlockSize;
    if (domainEndXCoordinate > image.size().width)
    {
        domainStartXCoordinate = image.size().width - domainBlockSize - 1;
    }
    if (domainEndYCoordinate > image.size().height)
    {
        domainStartYCoordinate = image.size().height - domainBlockSize - 1;
    }

    Mat domainBlock(cv::Size(rangeBlock.size, rangeBlock.size), CV_8U);
    Mat roi = image(Rect(domainStartXCoordinate, domainStartYCoordinate, domainBlockSize, domainBlockSize));
    resize(roi, roi, Size(), 0.5, 0.5, INTER_NEAREST);
    // Get every other pixel.
    return roi;
}

Mat preprocess(const Mat &image)
{
    Mat result(image.size(), CV_8U);
    for (int i = 0; i < image.size().height; ++i)
    {
        for (int j = 0; j < image.size().width; ++j)
        {
            int sum = 0;
            int count = 0;
            if (i > 0)
            {
                sum += image.at<uchar>(i - 1, j);
                ++count;
            }
            if (j > 0)
            {
                sum += image.at<uchar>(i, j - 1);
                ++count;
            }
            if (i < image.size().height - 1)
            {
                sum += image.at<uchar>(i + 1, j);
                ++count;
            }
            if (j < image.size().width - 1)
            {
                sum += image.at<uchar>(i, j + 1);
                ++count;
            }
            result.at<uchar>(i, j) = sum / count;
        }
    }
    return result;
}

vector<Block> compress(const Mat &image, const int startRangeSize, const int minRangeSize, const int errorThreshold)
{
    vector<Block> transformations;
    int width = image.size().width;
    int height = image.size().height;

    queue<Block> uncoveredRangeBlocks;
    for (int i = 0; i < height; i += startRangeSize)
    {
        for (int j = 0; j < width; j += startRangeSize)
        {
            uncoveredRangeBlocks.push(Block(i, j, startRangeSize));
        }
    }
    Mat processedImage = preprocess(image);
    while (uncoveredRangeBlocks.size() != 0)
    {
        Block rangeBlock = uncoveredRangeBlocks.front();
        Mat rangeBlockImage = image(Rect(rangeBlock.startX, rangeBlock.startY, rangeBlock.size, rangeBlock.size));
        Mat domainBlockImage = findReducedDomainBlock(processedImage, rangeBlock);
        int level = (int)log2(startRangeSize / rangeBlock.size);
        int newErrorThreshold = errorThreshold;//(pow(2, level)) * errorThreshold + (pow(2, level)) - 1;
        Scalar a;
        Scalar r;
        float error = findParamsAndError(rangeBlockImage, domainBlockImage, newErrorThreshold, a, r);
        rangeBlock.a = a;
        rangeBlock.r = r;
        uncoveredRangeBlocks.pop();
        if (error < newErrorThreshold || rangeBlock.size == minRangeSize)
        {
            transformations.push_back(rangeBlock);
        }
        else
        {
            int newRangeSize = rangeBlock.size / 2;
            uncoveredRangeBlocks.push(Block(rangeBlock.startX, rangeBlock.startY, newRangeSize));
            uncoveredRangeBlocks.push(Block(rangeBlock.startX + newRangeSize, rangeBlock.startY, newRangeSize));
            uncoveredRangeBlocks.push(Block(rangeBlock.startX, rangeBlock.startY + newRangeSize, newRangeSize));
            uncoveredRangeBlocks.push(Block(rangeBlock.startX + newRangeSize, rangeBlock.startY + newRangeSize, newRangeSize));
        }
    }
    return transformations;
}

Mat decompress(const vector<Block> &transformations, const int outputSize, const int numberIterations = 8)
{

    vector<Mat> iterations;
    Mat currentImage(cv::Size(outputSize, outputSize), CV_8U);
    Mat firstImage(cv::Size(outputSize, outputSize), CV_8U);
    iterations.push_back(firstImage);
    cout << transformations.size() << endl;
    for (int i = 0; i < numberIterations; ++i)
    {
        cout << i << endl;
        Mat processedImage = preprocess(iterations.back());
        for (auto &block : transformations)
        {
            Mat domainBlock = findReducedDomainBlock(processedImage, block);
            Scalar average = mean(domainBlock);
            domainBlock = (domainBlock - average) * block.a.val[0] + block.r.val[0];
            domainBlock.copyTo(currentImage(Rect(block.startX, block.startY, block.size, block.size)));
        }
        iterations.push_back(currentImage);
        currentImage = Mat(cv::Size(outputSize, outputSize), CV_8U);
    }
    return iterations.back();
}

double getPSNR(const Mat &I1, const Mat &I2)
{
    Mat s1;
    absdiff(I1, I2, s1);      // |I1 - I2|
    s1.convertTo(s1, CV_32F); // cannot make a square on 8 bits
    s1 = s1.mul(s1);          // |I1 - I2|^2

    Scalar s = sum(s1); // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

int main()
{
    Mat image = imread("ss2.png", IMREAD_GRAYSCALE);
    image = image(Rect(300, 300, 512, 512));

    if (!image.data)
    {
        cout << "Couldn't open the image." << std::endl;
        return -1;
    }
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    vector<Block> transformations = compress(image, 16, 2, 20);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Time difference = " << (float)chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000000 << "s" << endl;
    Mat decompressed = decompress(transformations, 512, 15);

    imwrite("lena512cpp.bmp", decompressed);
    namedWindow("Display window", WINDOW_AUTOSIZE);
    putText(decompressed,                                 //target image
            std::to_string(getPSNR(image, decompressed)), //text
            cv::Point(10, image.rows / 2),                //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(118, 185, 0), //font color
            2);

    imshow("Display window", decompressed);
    waitKey(0);
    return 0;

    /*VideoCapture cap("C:\\Users\\austi\\Downloads\\bird.avi");*/

    //if (!cap.isOpened()) {
    //    cout << "Error opening video stream or file" << endl;
    //    return -1;
    //}

    //while (1) {

    //    Mat frame;

    //    cap >> frame;

    //    if (frame.empty()) {
    //        break;
    //    }

    //    vector<Mat> domain_blocks = generate_domain_blocks(frame, 16);
    //    imshow("Frame", frame);

    //    // Stop the video with escape.
    //    char c = (char)waitKey(25);
    //    if (c == 27) {
    //        break;
    //    }
    //}

    //cap.release();
    //destroyAllWindows();

    return 0;
}
