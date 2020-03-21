#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <queue>
#include <chrono>

using namespace cv;
using namespace std;

struct Block {
    int startFrame;
    int startX;
    int startY;
    int size;
    int numFrames;
    int error;
    Scalar a;
    Scalar r;

    Block(int startFrame, int startX, int startY, int size, int numFrames = 0) : startFrame{startFrame}, startX{startX},
                                                                                 startY{startY}, size{size}, numFrames{
                    numFrames == 0 ? size : numFrames} {}
};

vector<Mat> findReducedDomainBlockNOPREPROCESSING(const vector<Mat> &video, const Block &rangeBlock) {
    int domainBlockSize = rangeBlock.size * 2;
    int domainStartFrame = rangeBlock.startFrame;
    int domainStartXCoordinate = max(rangeBlock.startX - rangeBlock.size / 2, 0);
    int domainStartYCoordinate = max(rangeBlock.startY - rangeBlock.size / 2, 0);

    int domainEndFrame = domainStartFrame + rangeBlock.numFrames;
    int domainEndXCoordinate = domainStartXCoordinate + domainBlockSize;
    int domainEndYCoordinate = domainStartYCoordinate + domainBlockSize;

    if (domainEndXCoordinate > video[0].size().width) {
        domainStartXCoordinate = video[0].size().width - domainBlockSize;
        domainEndXCoordinate = domainStartXCoordinate + domainBlockSize;
    }
    if (domainEndYCoordinate > video[0].size().height) {
        domainStartYCoordinate = video[0].size().height - domainBlockSize;
        domainEndYCoordinate = domainStartYCoordinate + domainBlockSize;
    }

    vector<Mat> domainBlock;
    for (int i = domainStartFrame; i < domainEndFrame; ++i) {
        //if (i % 2 != 0) continue;
        Mat domainBlockFrame = video[i];
        Mat roi = domainBlockFrame(
                Rect(domainStartXCoordinate, domainStartYCoordinate, domainBlockSize, domainBlockSize));
        resize(roi, roi, Size(), 0.5, 0.5);
        domainBlock.push_back(roi);
    }
    return domainBlock;
}

// In this case, image is already reduced.
vector<Mat> findReducedDomainBlock(const vector<Mat> &video, const Block &rangeBlock) {
    int domainBlockSize = rangeBlock.size * 2;
    int domainStartFrame = rangeBlock.startFrame;
    int domainStartXCoordinate = max(rangeBlock.startX - rangeBlock.size / 2, 0);
    int domainStartYCoordinate = max(rangeBlock.startY - rangeBlock.size / 2, 0);

    int domainEndFrame = domainStartFrame + rangeBlock.numFrames;
    int domainEndXCoordinate = domainStartXCoordinate + domainBlockSize;
    int domainEndYCoordinate = domainStartYCoordinate + domainBlockSize;

    if (domainEndXCoordinate > video[0].size().width) {
        domainStartXCoordinate = video[0].size().width - domainBlockSize;
        domainEndXCoordinate = domainStartXCoordinate + domainBlockSize;
    }
    if (domainEndYCoordinate > video[0].size().height) {
        domainStartYCoordinate = video[0].size().height - domainBlockSize;
        domainEndYCoordinate = domainStartYCoordinate + domainBlockSize;
    }

    vector<Mat> domainBlock;
    for (int i = rangeBlock.startFrame; i < rangeBlock.startFrame + rangeBlock.numFrames; ++i) {
        Mat domainBlockFrame = video[i](
                Rect(domainStartXCoordinate, domainStartYCoordinate, domainBlockSize, domainBlockSize));
        Mat roi(rangeBlock.size, rangeBlock.size, CV_8U);
        for (int j = 0; j < domainBlockFrame.size().height; j += 2) {
            for (int k = 0; k < domainBlockFrame.size().width; k += 2) {
                roi.at<uchar>(k / 2, j / 2) = domainBlockFrame.at<uchar>(k, j);
            }
        }
        domainBlock.push_back(roi);
    }

    //vector<Mat> domainBlock;
    //for (int i = domainStartFrame; i < domainEndFrame; ++i) {
    //    //if (i % 2 != 0) continue;
    //    Mat domainBlockFrame = video[i];
    //    Mat roi = domainBlockFrame(
    //            Rect(domainStartXCoordinate, domainStartYCoordinate, domainBlockSize, domainBlockSize));
    //    resize(roi, roi, Size(), 0.5, 0.5, INTER_NEAREST);
    //    domainBlock.push_back(roi);
    //}
    return domainBlock;
}

vector<Mat> preprocess(const vector<Mat> &video) {
    vector<Mat> result;

    int numFrames = video.size();
    for (int z = 0; z < numFrames; ++z) {
        Mat frame = Mat(video[0].size(), CV_8U);
        result.push_back(frame);
        for (int i = 0; i < video[0].size().height; ++i) {
            for (int j = 0; j < video[0].size().width; ++j) {
                int sum = 0;
                int count = 0;
                if (i > 0) {
                    sum += video[z].at<uchar>(i - 1, j);
                    ++count;
                }
                if (j > 0) {
                    sum += video[z].at<uchar>(i, j - 1);
                    ++count;
                }
                if (i < video[0].size().height - 1) {
                    sum += video[z].at<uchar>(i + 1, j);
                    ++count;
                }
                if (j < video[0].size().width - 1) {
                    sum += video[z].at<uchar>(i, j + 1);
                    ++count;
                }

                result[z].at<uchar>(i, j) = sum / count;
            }
        }
    }
    return result;
}

Scalar getVideoAverage(const vector<Mat> &video) {
    int numPixels = video[0].size().height * video[0].size().width;
    int frameCount = 0;
    Scalar frameSum;
    for (auto &frame : video) {
        ++frameCount;
        frameSum += sum(frame);
    }
    return frameSum[0] / (Scalar) (numPixels * frameCount);
}

// Check all frames to see if they meet the error threshold.
float
findParamsAndError(const vector<Mat> &rangeBlock, const vector<Mat> &domainBlock, const float errorThreshold, Scalar &a,
                   Scalar &r) {
    float minError = INT_MAX;
    float error = minError;
    int bestA = -5;
    r = getVideoAverage(rangeBlock);
    Scalar d_mean = getVideoAverage(domainBlock);
    //cout << d_mean << " " << r << endl;
    a = 0.5;
    float maxFrameError = INT_MIN;
    for (float trialA = -0.5; trialA <= 0.5; trialA += 0.125) {
        error = 0;
        maxFrameError = 0;
        float count = 0;
        for (int z = 0; z < domainBlock.size(); ++z) {
//            float frameError = pow(norm((domainBlock[z] - d_mean) * trialA + r, rangeBlock[z]), 2);
            float frameError = 0;
            for (int i = 0; i < domainBlock[z].size().height; ++i) {
                for (int j = 0; j < domainBlock[z].size().height; ++j) {
                    ++count;
                    frameError += pow(((domainBlock[z].at<uchar>(i, j) - d_mean.val[0]) * trialA) -
                                      (rangeBlock[z].at<uchar>(i, j) - r.val[0]),
                                      2);
                }
            }
            error += frameError;
            if (frameError > maxFrameError) maxFrameError = frameError;
            //if (frameError > minError) break;
        }
        //error = error / count;
        //error = sqrt(error);
        //cout << error << endl;
        if (maxFrameError < minError) {
            minError = maxFrameError;
            a = trialA;
        }
    }
    //cout << "Threshold: " << errorThreshold << " maxFrame " << maxFrameError << " a: " << a << endl;
    return maxFrameError;
}

vector<Block>
compress(const vector<Mat> &video, const int startRangeSize, const int minRangeSize, const int errorThreshold) {
    vector<Block> transformations;
    int numFrames = video.size();
    int width = video[0].size().width;
    int height = video[0].size().height;

    queue<Block> uncoveredRangeBlocks;
    for (int z = 0; z < numFrames; z += startRangeSize) {
        for (int i = 0; i < height; i += startRangeSize) {
            for (int j = 0; j < width; j += startRangeSize) {
                uncoveredRangeBlocks.push(Block(z, i, j, startRangeSize));
            }
        }
    }
    vector<Mat> processedVideo = preprocess(video);
    while (!uncoveredRangeBlocks.empty()) {
        Block rangeBlock = uncoveredRangeBlocks.front();
        uncoveredRangeBlocks.pop();
        vector<Mat> rangeBlockVideo;
        for (int i = rangeBlock.startFrame; i < rangeBlock.startFrame + rangeBlock.numFrames; ++i) {
            Mat rangeBlockFrame = video[i];
            Mat roi = rangeBlockFrame(Rect(rangeBlock.startX, rangeBlock.startY, rangeBlock.size, rangeBlock.size));
            rangeBlockVideo.push_back(roi);
        }
        vector<Mat> domainBlockVideo = findReducedDomainBlock(processedVideo, rangeBlock);
        int level = (int) log2(startRangeSize / rangeBlock.size);
        float newErrorThreshold;
        if (rangeBlock.numFrames == 1)
            newErrorThreshold = INT_MAX;
        else
            newErrorThreshold = (pow(2, level)) * errorThreshold + (pow(2, level)) - 1;
        //cout << newErrorThreshold << endl;
        Scalar a;
        Scalar r;

        float error = findParamsAndError(rangeBlockVideo, domainBlockVideo, newErrorThreshold, a, r);
        rangeBlock.a = a;
        rangeBlock.r = r;
        if (error < newErrorThreshold || rangeBlock.numFrames == 1) {
            transformations.push_back(rangeBlock);
        } else {
            if (rangeBlock.size != minRangeSize) {
                int newRangeSize = rangeBlock.size / 2;
                int newNumFrames = newRangeSize;
                uncoveredRangeBlocks.push(
                        Block(rangeBlock.startFrame, rangeBlock.startX, rangeBlock.startY, newRangeSize, newNumFrames));
                uncoveredRangeBlocks.push(
                        Block(rangeBlock.startFrame, rangeBlock.startX + newRangeSize, rangeBlock.startY, newRangeSize,
                              newNumFrames));
                uncoveredRangeBlocks.push(
                        Block(rangeBlock.startFrame, rangeBlock.startX, rangeBlock.startY + newRangeSize, newRangeSize,
                              newNumFrames));
                uncoveredRangeBlocks.push(
                        Block(rangeBlock.startFrame, rangeBlock.startX + newRangeSize, rangeBlock.startY + newRangeSize,
                              newRangeSize, newNumFrames));

                uncoveredRangeBlocks.push(
                        Block(rangeBlock.startFrame + newRangeSize, rangeBlock.startX, rangeBlock.startY, newRangeSize,
                              newNumFrames));
                uncoveredRangeBlocks.push(
                        Block(rangeBlock.startFrame + newRangeSize, rangeBlock.startX + newRangeSize, rangeBlock.startY,
                              newRangeSize, newNumFrames));
                uncoveredRangeBlocks.push(
                        Block(rangeBlock.startFrame + newRangeSize, rangeBlock.startX, rangeBlock.startY + newRangeSize,
                              newRangeSize, newNumFrames));
                uncoveredRangeBlocks.push(Block(rangeBlock.startFrame + newRangeSize, rangeBlock.startX + newRangeSize,
                                                rangeBlock.startY + newRangeSize, newRangeSize, newNumFrames));
            } else {
                for (int i = 0; i < rangeBlock.numFrames; ++i) {
                    uncoveredRangeBlocks.push(
                            Block(rangeBlock.startFrame + i, rangeBlock.startX, rangeBlock.startY, rangeBlock.size, 1));
                }
            }
        }
    }
    return transformations;
}

vector<Mat> decompress(const vector<Block> &transformations, const int numFrames, const int outputSize,
                       const int numberIterations = 8, bool showOutlines = false) {
    vector<vector<Mat>> iterations;
    vector<Mat> currentVideo;
    for (int i = 0; i < numFrames; ++i) {
        currentVideo.emplace_back(Mat(cv::Size(outputSize, outputSize), CV_8U));
    }
    vector<Mat> firstVideo = vector<Mat>(currentVideo);
    iterations.push_back(firstVideo);
    for (int i = 0; i < numberIterations; ++i) {
        //cout << i << endl;
        vector<Mat> processedVideo = preprocess(iterations.back());
        for (auto &rangeBlock : transformations) {
            vector<Mat> domainBlock = findReducedDomainBlock(processedVideo, rangeBlock);
            Scalar average = getVideoAverage(domainBlock);
            for (int z = 0; z < rangeBlock.numFrames; z++) {
                domainBlock[z] = (domainBlock[z] - average) * rangeBlock.a.val[0] + rangeBlock.r.val[0];

                if (showOutlines && i == numberIterations - 1) {
                    for (int k = 0; k < domainBlock[z].size().height; ++k) {
                        for (int j = 0; j < domainBlock[z].size().height; ++j) {
                            if (k == 0 || k == domainBlock[z].size().height - 1 || j == 0 ||
                                j == domainBlock[z].size().height - 1) {
                                domainBlock[z].at<uchar>(k, j) = 0;
                            }
                        }
                    }
                }
                domainBlock[z].copyTo(currentVideo[z + rangeBlock.startFrame](
                        Rect(rangeBlock.startX, rangeBlock.startY, rangeBlock.size, rangeBlock.size)));
            }
        }
        iterations.emplace_back(vector<Mat>(currentVideo));
        currentVideo = vector<Mat>(iterations.back());
    }
    return iterations.back();
}

int main() {

    // PARAMS
    int startBlockSize = 16;
    int minBlockSize = 2;
    int errorThreshold = 20;
    int numberIterations = 9;
    bool showOutlines = 0;

    string fileName = "sintel";
    string fileExtension = ".y4m";
    VideoCapture cap("/Users/Austin/dev/fcode/" + fileName + fileExtension);

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << endl;
        return -1;
    }

    vector<Mat> frames;
    int frameCount = 0;
    int skipFrames = 128;
    int skippedFrames = 0;
    long transformationCount = 0;
    int maxFrames = 32;
    int outputSize = 704;
    float totalTime = 0;
    int blockSize = 32;
    VideoWriter video(fileName + "c.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 24, Size(outputSize, outputSize),
                      false);
    while (true) {
        Mat frame, greyFrame;
        while (skippedFrames < skipFrames) {
            cap.grab();
            ++skippedFrames;
        }
        if (frameCount > maxFrames && maxFrames != 0)
            break;

        cap >> frame;
        if (frame.empty()) {
            break;
        }
        ++frameCount;
        cvtColor(frame, greyFrame, COLOR_BGR2GRAY);
        frames.push_back(greyFrame(Rect(0, 0, outputSize, outputSize)));
        if (frames.size() % 32 == 0) {
            cout << "Compressing block..." << endl;
            chrono::steady_clock::time_point begin = chrono::steady_clock::now();
            vector<Block> transformations = compress(frames, startBlockSize, minBlockSize, errorThreshold);
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            float seconds = (float) chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000000;
            totalTime += seconds;
            cout << "Done compressing" << endl;
            cout << "Transformation count: " << transformations.size() << endl;
            cout << "Compression ratio: "
                 << (outputSize * outputSize * blockSize * 8) / (float) (transformations.size() * 11) << endl
                 << endl;
            transformationCount += transformations.size();
            cout << "Decompressing block..." << endl;
            vector<Mat> decompressed = decompress(transformations,
                                                  blockSize,
                                                  outputSize,
                                                  numberIterations,
                                                  showOutlines);
            cout << "Done decompressing" << endl
                 << endl;
            for (Mat &frame : decompressed) {
                video.write(frame);
            }
            frames.clear();
        }
    }
    cout << "Done." << endl;
    cout << "Total encoding time: " << totalTime << endl;
    cout << "Frames encoded per second: " << frameCount / totalTime << endl;
    cout << "Transformation count: " << transformationCount << endl;
    cout << "Compression ratio: " << (outputSize * outputSize * frameCount * 8) / (double) (transformationCount * 11)
         << endl;

    cap.release();
    destroyAllWindows();

    return 0;
}
