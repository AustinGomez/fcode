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
    int startFrame;
    int startX;
    int startY;
    int size;
    Scalar a;
    Scalar r;

    Block(int startFrame, int startX, int startY, int size) : startFrame{startFrame}, startX{startX}, startY{startY}, size{size} {}
};

float findParamsAndError(const vector<Mat> &rangeBlock, const vector<Mat> &domainBlock, const int errorThreshold, Scalar &a, Scalar &r)
{
    int minError = INT16_MAX;
    int error = minError;
    int bestA = -5;

    int frameCount = 0;
    Scalar sum;
    for (auto &frame : rangeBlock)
    {
        frameCount += 1;
        sum += mean(frame);
    }
    r = sum / (Scalar)frameCount;

    sum = 0;
    for (auto &frame : domainBlock)
    {
        sum += mean(frame);
    }
    Scalar d_mean = sum / (Scalar)frameCount;

    a = 0.5;
    for (float trialA = -1; trialA <= 1; trialA += 0.05)
    {
        float error = 0;
        for (int z = 0; z < domainBlock.size(); ++z)
        {
            for (int i = 0; i < domainBlock[z].size().height; ++i)
            {
                for (int j = 0; j < domainBlock[z].size().height; ++j)
                {
                    error += pow(((domainBlock[z].at<uchar>(i, j) - d_mean.val[0]) * trialA) - (rangeBlock[z].at<uchar>(i, j) - r.val[0]), 2);
                    if (error > errorThreshold || error > minError)
                        break;
                }
                if (error > errorThreshold || error > minError)
                    break;
            }
            if (error > errorThreshold || error > minError)
                break;
        }

        if (error > errorThreshold)
            continue;
        if (error < minError)
        {
            minError = error;
            a = trialA;
        }
    }
    // if (minError > 200 && errorThreshold == INT_MAX) cout << "Failed " << endl;
    //cout << "Min " << minError << endl;
    return minError;
}

// In this case, image is already reduced.
vector<Mat> findReducedDomainBlock(const vector<Mat> &video, const Block &rangeBlock)
{
    vector<Mat> domainBlock;
    int domainBlockSize = rangeBlock.size * 2;
    int domainStartFrame = max(rangeBlock.startFrame - rangeBlock.size / 2, 0);
    int domainStartXCoordinate = max(rangeBlock.startX - rangeBlock.size / 2, 0);
    int domainStartYCoordinate = max(rangeBlock.startY - rangeBlock.size / 2, 0);

    int domainEndFrame = domainStartFrame + domainBlockSize;
    int domainEndXCoordinate = domainStartXCoordinate + domainBlockSize;
    int domainEndYCoordinate = domainStartYCoordinate + domainBlockSize;

    // This should be the number of frames.
    if (domainEndFrame > video.size())
    {
        domainStartFrame = video.size() - domainBlockSize;
        domainEndFrame = domainStartFrame + domainBlockSize;
    }
    if (domainEndXCoordinate > video[0].size().width)
    {
        domainStartXCoordinate = video[0].size().width - domainBlockSize;
        domainEndXCoordinate = domainStartXCoordinate + domainBlockSize;
    }
    if (domainEndYCoordinate > video[0].size().height)
    {
        domainStartYCoordinate = video[0].size().height - domainBlockSize;
        domainEndYCoordinate = domainStartYCoordinate + domainBlockSize;
    }

    for (int i = domainStartFrame; i < domainStartFrame + domainBlockSize; i += 2)
    {
        Mat domainBlockFrame = video[i](Rect(domainStartXCoordinate, domainStartYCoordinate, domainBlockSize, domainBlockSize));
        Mat roi(rangeBlock.size, rangeBlock.size, CV_8U);
        resize(domainBlockFrame, roi, Size(), 0.5, 0.5);
        domainBlock.push_back(roi);
    }
    return domainBlock;
    //vector<Mat> domainBlock;
    //for (int i = domainStartFrame; i < domainStartFrame + domainBlockSize; i += 2) {
    //    Mat domainBlockFrame = video[i](Rect(domainStartXCoordinate, domainStartYCoordinate, domainBlockSize, domainBlockSize));
    //    Mat roi(rangeBlock.size, rangeBlock.size, CV_8U);
    //    for (int j = 0; j < domainBlockFrame.size().height; j += 2) {
    //        for (int k = 0; k < domainBlockFrame.size().width; k += 2) {
    //            roi.at<uchar>(k/2, j/2) = domainBlockFrame.at<uchar>(k, j);
    //        }
    //    }
    //    domainBlock.push_back(roi);
    //}

    //vector<Mat> domainBlock;
    //for (int i = domainStartFrame; i < domainEndFrame; ++i) {
    //    if (i % 2 == 0) continue;
    //    Mat domainBlockFrame = video[i];
    //    Mat roi = domainBlockFrame(Rect(domainStartXCoordinate, domainStartYCoordinate, domainBlockSize, domainBlockSize));
    //    resize(roi, roi, Size(), 0.5, 0.5, INTER_NEAREST);
    //    domainBlock.push_back(roi);
    //}
    return domainBlock;
}

vector<Mat> preprocess(const vector<Mat> &video)
{
    vector<Mat> result;

    int numFrames = video.size();
    for (int z = 0; z < numFrames; ++z)
    {
        Mat frame = Mat(video[0].size(), CV_8U);
        result.push_back(frame);
        for (int i = 0; i < video[0].size().height; ++i)
        {
            for (int j = 0; j < video[0].size().width; ++j)
            {
                int sum = 0;
                int count = 0;
                if (z > 0)
                {
                    sum += video[z - 1].at<uchar>(i, j);
                    ++count;
                }
                if (i > 0)
                {
                    sum += video[z].at<uchar>(i - 1, j);
                    ++count;
                }
                if (j > 0)
                {
                    sum += video[z].at<uchar>(i, j - 1);
                    ++count;
                }
                if (z < numFrames - 1)
                {
                    sum += video[z + 1].at<uchar>(i, j);
                    ++count;
                }
                if (i < video[0].size().height - 1)
                {
                    sum += video[z].at<uchar>(i + 1, j);
                    ++count;
                }
                if (j < video[0].size().width - 1)
                {
                    sum += video[z].at<uchar>(i, j + 1);
                    ++count;
                }

                result[z].at<uchar>(i, j) = sum / count;
            }
        }
    }
    return result;
}

vector<Block> compress(const vector<Mat> &video, const int startRangeSize, const int minRangeSize, const int errorThreshold)
{
    vector<Block> transformations;
    int numFrames = video.size();
    int width = video[0].size().width;
    int height = video[0].size().height;

    queue<Block> uncoveredRangeBlocks;
    for (int z = 0; z < numFrames; z += startRangeSize)
    {
        for (int i = 0; i < height; i += startRangeSize)
        {
            for (int j = 0; j < width; j += startRangeSize)
            {
                uncoveredRangeBlocks.push(Block(z, i, j, startRangeSize));
            }
        }
    }
    vector<Mat> processedVideo = video;
    while (uncoveredRangeBlocks.size() != 0)
    {
        //if (uncoveredRangeBlocks.size() % 10000 == 0) {
        //    cout << uncoveredRangeBlocks.size() << endl;
        //}
        Block rangeBlock = uncoveredRangeBlocks.front();
        //Range ranges[3];
        //ranges[0] = Range(rangeBlock.startFrame, rangeBlock.startFrame + rangeBlock.size);
        //ranges[1] = Range(rangeBlock.startX, rangeBlock.startX+ rangeBlock.size);
        //ranges[2] = Range(rangeBlock.startY, rangeBlock.startY+ rangeBlock.size);
        //Mat rangeBlockVideo = video(ranges);

        vector<Mat> rangeBlockVideo;
        for (int i = rangeBlock.startFrame; i < rangeBlock.startFrame + rangeBlock.size; ++i)
        {
            Mat rangeBlockFrame = video[i];
            Mat roi = rangeBlockFrame(Rect(rangeBlock.startX, rangeBlock.startY, rangeBlock.size, rangeBlock.size));
            rangeBlockVideo.push_back(roi);
        }
        vector<Mat> domainBlockVideo = findReducedDomainBlock(processedVideo, rangeBlock);
        int level = (int)log2(startRangeSize / rangeBlock.size);
        int newErrorThreshold;
        if (rangeBlock.size == minRangeSize)
            newErrorThreshold = INT_MAX;
        else
            newErrorThreshold = (pow(2, level)) * errorThreshold + (pow(2, level)) - 1;
        Scalar a;
        Scalar r;

        float error = findParamsAndError(rangeBlockVideo, domainBlockVideo, newErrorThreshold, a, r);
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
            uncoveredRangeBlocks.push(Block(rangeBlock.startFrame, rangeBlock.startX, rangeBlock.startY, newRangeSize));
            uncoveredRangeBlocks.push(Block(rangeBlock.startFrame, rangeBlock.startX + newRangeSize, rangeBlock.startY, newRangeSize));
            uncoveredRangeBlocks.push(Block(rangeBlock.startFrame, rangeBlock.startX, rangeBlock.startY + newRangeSize, newRangeSize));
            uncoveredRangeBlocks.push(Block(rangeBlock.startFrame, rangeBlock.startX + newRangeSize, rangeBlock.startY + newRangeSize, newRangeSize));

            if (rangeBlock.startFrame + newRangeSize < numFrames)
            {
                uncoveredRangeBlocks.push(Block(rangeBlock.startFrame + newRangeSize, rangeBlock.startX, rangeBlock.startY, newRangeSize));
                uncoveredRangeBlocks.push(Block(rangeBlock.startFrame + newRangeSize, rangeBlock.startX + newRangeSize, rangeBlock.startY, newRangeSize));
                uncoveredRangeBlocks.push(Block(rangeBlock.startFrame + newRangeSize, rangeBlock.startX, rangeBlock.startY + newRangeSize, newRangeSize));
                uncoveredRangeBlocks.push(Block(rangeBlock.startFrame + newRangeSize, rangeBlock.startX + newRangeSize, rangeBlock.startY + newRangeSize, newRangeSize));
            }
        }
    }
    return transformations;
}

vector<Mat> decompress(const vector<Block> &transformations, const int numFrames, const int outputSize, const int numberIterations = 8)
{
    vector<vector<Mat>> iterations;
    vector<Mat> currentVideo;
    for (int i = 0; i < numFrames; ++i)
    {
        currentVideo.push_back(Mat(cv::Size(outputSize, outputSize), CV_8U));
    }
    vector<Mat> firstVideo = vector<Mat>(currentVideo);
    iterations.push_back(firstVideo);
    cout << transformations.size() << endl;
    for (int i = 0; i < numberIterations; ++i)
    {
        cout << i << endl;
        vector<Mat> processedVideo = iterations.back();
        for (auto &rangeBlock : transformations)
        {
            vector<Mat> domainBlock = findReducedDomainBlock(processedVideo, rangeBlock);
            Scalar sum = 0;
            vector<Scalar> means;
            for (auto &frame : domainBlock)
            {
                sum += mean(frame);
            }
            Scalar average = sum / (Scalar)rangeBlock.size;
            for (int z = 0; z < rangeBlock.size; z++)
            {
                domainBlock[z] = (domainBlock[z] - average.val[0]) * rangeBlock.a.val[0] + rangeBlock.r.val[0];
                //if (z < domainBlock.size() - 1) domainBlock[z] = domainBlock[z] - mean(domainBlock[z + 1]);
                domainBlock[z].copyTo(currentVideo[z + rangeBlock.startFrame](Rect(rangeBlock.startX, rangeBlock.startY, rangeBlock.size, rangeBlock.size)));
                //for (int k = 0; k < rangeBlock.size; k++) {
                //    for (int l = 0; l < rangeBlock.size; l++) {
                //        currentVideo[z + rangeBlock.startFrame].at<uchar>(l + rangeBlock.startY, k + rangeBlock.startX) = domainBlock[z].at<uchar>(l, k);
                //    }
                //}
            }
            //vector<Mat> D;
            //for (auto& frame : domainBlock) {
            //    D.push_back((frame - average) * rangeBlock.a.val[0] + rangeBlock.r.val[0]);
            //}
            //for (int z = 0; z < D.size(); ++z) {
            //    if (z < D.size() - 1) D[z] = D[z] - mean(D[z] - D[z + 1]);
            //}
            //for (int j = 0; j < rangeBlock.size; ++j) {
            //    D[j].copyTo(currentVideo[j + rangeBlock.startFrame](Rect(rangeBlock.startX, rangeBlock.startY, rangeBlock.size, rangeBlock.size)));
            //}
        }
        iterations.push_back(vector<Mat>(currentVideo));
        currentVideo.empty();
    }
    return iterations.back();
}

int main()
{
    string fileName = "sintel";
    string fileExtension = ".y4m";
    VideoCapture cap("/Users/Austin/dev/fcode/" + fileName + fileExtension);

    if (!cap.isOpened())
    {
        std::cout << "Error opening video stream or file" << endl;
        return -1;
    }

    vector<Mat> frames;
    int frameCount = 0;
    int skipFrames = 600;
    int skippedFrames = 0;
    int maxFrames = 32;
    int outputSize = 720;
    while (frameCount < maxFrames)
    {
        Mat frame, greyFrame;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        if (skippedFrames < skipFrames)
        {
            skippedFrames++;
            continue;
        }
        ++frameCount;
        cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
        frames.push_back(greyFrame(Rect(256, 0, outputSize, outputSize)));
    }
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    vector<Block> transformations = compress(frames, 16, 2, 5);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Done in " << (float)chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000000 << "seconds" << endl;
    begin = chrono::steady_clock::now();
    vector<Mat> decompressed = decompress(transformations, frameCount, outputSize, 10);
    end = chrono::steady_clock::now();
    cout << "Done in " << (float)chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000000 << "seconds" << endl;
    VideoWriter video(fileName + "c.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 24, Size(outputSize, outputSize), false);

    for (Mat &frame : decompressed)
    {
        video.write(frame);
    }
    cap.release();
    destroyAllWindows();

    return 0;
}
