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
    int numFrames; int error;
    int level=0;
    Scalar a;
    Scalar r;
    vector<Block *> children;
    bool hasChildren = false;

    Block(int startFrame, int startX, int startY, int size, int numFrames = 0, int level=0):
            startFrame{startFrame},
            startX{startX},
            startY{startY},
            size{size},
            numFrames{ numFrames == 0 ? size : numFrames},
            level{level} {};
};

Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}

double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);        // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

// In this case, image is already reduced.
vector<Mat> findReducedDomainBlock(const vector<Mat> &video, const Block *rangeBlock) {
    int domainBlockSize = rangeBlock->size * 2;
    int domainStartFrame = rangeBlock->startFrame;
    int domainStartXCoordinate = max(rangeBlock->startX - rangeBlock->size / 2, 0);
    int domainStartYCoordinate = max(rangeBlock->startY - rangeBlock->size / 2, 0);

    int domainEndFrame = domainStartFrame + rangeBlock->numFrames;
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

    //vector<Mat> domainBlock;
    //for (int i = rangeBlock->startFrame; i < rangeBlock->startFrame + rangeBlock->numFrames; ++i) {
    //    Mat domainBlockFrame = video[i](
    //            Rect(domainStartXCoordinate, domainStartYCoordinate, domainBlockSize, domainBlockSize));
    //    Mat roi(rangeBlock->size, rangeBlock->size, CV_8U);
    //    for (int j = 0; j < domainBlockFrame.size().height; j += 2) {
    //        for (int k = 0; k < domainBlockFrame.size().width; k += 2) {
    //            roi.at<uchar>(k / 2, j / 2) = domainBlockFrame.at<uchar>(k, j);
    //        }
    //    }
    //    domainBlock.push_back(roi);
    //}

    vector<Mat> domainBlock;
    for (int i = domainStartFrame; i < domainEndFrame; ++i) {
        //if (i % 2 != 0) continue;
        Mat domainBlockFrame = video[i];
        Mat roi = domainBlockFrame(
                Rect(domainStartXCoordinate, domainStartYCoordinate, domainBlockSize, domainBlockSize));
        resize(roi, roi, Size(), 0.5, 0.5, INTER_NEAREST);
        domainBlock.push_back(roi);
    }
    return domainBlock;
}

vector<Mat> preprocess(const vector<Mat> &video) {
    vector<Mat> result;

    int numFrames = video.size();
    for (int z = 0; z < numFrames; ++z) {
        Mat frame = Mat(video[0].size(), CV_8U);
        for (int i = 0; i < video[0].size().height; ++i) {
            for (int j = 0; j < video[0].size().width; ++j) {
                int sum = 0;
                int count = 0;
                if (z > 0) {
                    sum += video[z-1].at<uchar>(i, j);
                    ++count;
                }
                if (z < numFrames - 1) {
                    sum += video[z+1].at<uchar>(i, j);
                    ++count;
                }
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

                frame.at<uchar>(i, j) = sum / count;
            }
        }
        result.push_back(frame);
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

void octTreeDown(Block *block, queue<Block *> &blockQueue, const int maxDepth) {
    block->hasChildren = true;
    int newLevel = block->level + 1;
    if (block->level < maxDepth) {
        int newRangeSize = block->size / 2;
        int newNumFrames = newRangeSize;
        Block *q1 = new Block(block->startFrame, block->startX, block->startY, newRangeSize, newNumFrames, newLevel);
        Block *q2 = new Block(block->startFrame, block->startX + newRangeSize, block->startY, newRangeSize, newNumFrames, newLevel);
        Block *q3 = new Block(block->startFrame, block->startX, block->startY + newRangeSize, newRangeSize, newNumFrames, newLevel);
        Block *q4 = new Block(block->startFrame, block->startX + newRangeSize, block->startY + newRangeSize, newRangeSize, newNumFrames, newLevel);

        Block *q5 = new Block(block->startFrame + newRangeSize, block->startX, block->startY, newRangeSize, newNumFrames, newLevel);
        Block *q6 = new Block(block->startFrame + newRangeSize, block->startX + newRangeSize, block->startY, newRangeSize, newNumFrames, newLevel);
        Block *q7 = new Block(block->startFrame + newRangeSize, block->startX, block->startY + newRangeSize, newRangeSize, newNumFrames, newLevel);
        Block *q8 = new Block(block->startFrame + newRangeSize, block->startX + newRangeSize, block->startY + newRangeSize, newRangeSize, newNumFrames, newLevel);

        blockQueue.push(q1);
        blockQueue.push(q2);
        blockQueue.push(q3);
        blockQueue.push(q4);
        blockQueue.push(q5);
        blockQueue.push(q6);
        blockQueue.push(q7);
        blockQueue.push(q8);

        block->children.push_back(q1);
        block->children.push_back(q2);
        block->children.push_back(q3);
        block->children.push_back(q4);
        block->children.push_back(q5);
        block->children.push_back(q6);
        block->children.push_back(q7);
        block->children.push_back(q8);
    } else {
        for (int i = 0; i < block->numFrames; ++i) {
            Block * newChild = new Block(block->startFrame + i, block->startX, block->startY, block->size, 1, newLevel);
            blockQueue.push(newChild);
            block->children.push_back(newChild);
        }
    }
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
    for (float trialA = -1; trialA <= 1; trialA += 0.125) {
        error = 0;
        maxFrameError = 0;
        for (int z = 0; z < domainBlock.size(); ++z) {
            //float frameError = pow(norm((domainBlock[z] - d_mean) * trialA + r, rangeBlock[z]), 2);
            float frameError = 0;
            for (int i = 0; i < domainBlock[z].size().height; ++i) {
               for (int j = 0; j < domainBlock[z].size().height; ++j) {
                    frameError += pow(((domainBlock[z].at<uchar>(i, j) - d_mean.val[0]) * trialA) -
                                         (rangeBlock[z].at<uchar>(i, j) - r.val[0]),
                                         2);
                }
            }
            error += frameError;
            if (frameError > maxFrameError) maxFrameError = frameError;
        }
        //error = error / rangeBlock.size();
        if (error < minError) {
            minError = error;
            a = trialA;
        }
    }
    //cout << "Threshold: " << errorThreshold << " maxFrame " << maxFrameError << " a: " << a << " size " << rangeBlock.size() << endl;
    return minError;
}

vector<Block *>
compress(const vector<Mat> &video, const int startRangeSize, const int minRangeSize, const int errorThreshold) {
    vector<Block> transformations;
    int numFrames = video.size();
    int width = video[0].size().width;
    int height = video[0].size().height;

    queue<Block *> uncoveredRangeBlocks;
    vector<Block *> topRangeBlocks;
    for (int z = 0; z < numFrames; z += startRangeSize) {
        for (int i = 0; i < height; i += startRangeSize) {
            for (int j = 0; j < width; j += startRangeSize) {
                Block *rangeBlock = new Block(z, j, i, startRangeSize, startRangeSize, 0);
                uncoveredRangeBlocks.push(rangeBlock);
                topRangeBlocks.push_back(rangeBlock);
            }
        }
    }
    vector<Mat> processedVideo = preprocess(video);
    while (!uncoveredRangeBlocks.empty()) {
        Block *rangeBlock = uncoveredRangeBlocks.front();
        uncoveredRangeBlocks.pop();

        vector<Mat> rangeBlockVideo;
        for (int i = rangeBlock->startFrame; i < rangeBlock->startFrame + rangeBlock->numFrames; ++i) {
            Mat rangeBlockFrame = video[i];
            Mat roi = rangeBlockFrame(Rect(rangeBlock->startX, rangeBlock->startY, rangeBlock->size, rangeBlock->size));
            rangeBlockVideo.push_back(roi);
        }

        vector<Mat> domainBlockVideo = findReducedDomainBlock(processedVideo, rangeBlock);

        float newErrorThreshold = errorThreshold;
        if (rangeBlock->numFrames == 1) newErrorThreshold = INT_MAX;

        Scalar a;
        Scalar r;
        float error = findParamsAndError(rangeBlockVideo, domainBlockVideo, newErrorThreshold, a, r);
        rangeBlock->a = a;
        rangeBlock->r = r;

        if (error >= newErrorThreshold && rangeBlock->numFrames != 1) {
            int maxDepth = (int) log2(startRangeSize / minRangeSize);
            octTreeDown(rangeBlock, uncoveredRangeBlocks, maxDepth);
        }
    }
    return topRangeBlocks;
}


void drawPartitions(Mat &block) {
    for (int k = 0; k < block.size().height; ++k) {
        for (int j = 0; j < block.size().height; ++j) {
            if (k == 0 || k == block.size().height - 1 || j == 0 ||
                j == block.size().height - 1) {
                block.at<uchar>(k, j) = 0;
            }
        }
    }
}

vector<Mat> decompress(const vector<Block *> &topRangeBlocks, const int numFrames, const int outputSize,
                       const int numberIterations = 8, bool showOutlines = false, const int maxDepth=4) {
    vector<vector<Mat>> iterations;
    vector<Mat> currentVideo;
    for (int i = 0; i < numFrames; ++i) {
        currentVideo.push_back(Mat(cv::Size(outputSize, outputSize), CV_8U, Scalar(0)));
    }
    vector<Mat> firstVideo = vector<Mat>(currentVideo);
    iterations.push_back(firstVideo);

    queue<Block *> uncheckedRangeBlocks;
    vector<Block *> rangeBlocks;
    int startRangeSize = outputSize / (int) sqrt(topRangeBlocks.size());
    for (int k = 0; k < topRangeBlocks.size(); ++k) {
        Block *block = topRangeBlocks[k];
        block->numFrames = numFrames;
        block->size = startRangeSize;
        block->level = 0;
        block->startFrame = 0;
        block->startX = ((startRangeSize * k) % outputSize);
        block->startY = startRangeSize * ((startRangeSize * k) / outputSize);
        block->hasChildren = topRangeBlocks[k]->hasChildren;
        uncheckedRangeBlocks.push(block);
    }

    while (!uncheckedRangeBlocks.empty()) {
        Block *rangeBlock = uncheckedRangeBlocks.front();
        uncheckedRangeBlocks.pop();
        if (rangeBlock->hasChildren) {
            for (int i = 0; i < rangeBlock->children.size(); ++i) {
                Block *child = rangeBlock->children[i];
                child->level = rangeBlock->level + 1;
                if (rangeBlock->level < maxDepth) {
                    child->size = rangeBlock->size / 2;
                    child->numFrames = numFrames / (pow(2, child->level));
                    child->startX = (i % 2 == 0) ? rangeBlock->startX : rangeBlock->startX + child->size;
                    child->startY = (i == 2 || i == 3 || i == 6 || i == 7) ? rangeBlock->startY + child->size : rangeBlock->startY;
                    child->startFrame = (i < 4) ? rangeBlock->startFrame : rangeBlock->startFrame + child->numFrames;
                    uncheckedRangeBlocks.push(child);
                } else {
                    child->size = rangeBlock->size;
                    child->numFrames = 1;
                    child->startX = rangeBlock->startX;
                    child->startY = rangeBlock->startY;
                    child->startFrame = rangeBlock->startFrame + i;
                    rangeBlocks.push_back(child);
                }
            }
        }
        else {
            rangeBlocks.push_back(rangeBlock);
        }
    }
    cout << rangeBlocks.size() << endl;
    for (int i = 0; i < numberIterations; ++i) {
        vector<Mat> processedVideo = preprocess(iterations.back());
        int counter = 0;
        for (auto &rangeBlock : rangeBlocks) {
            counter++;
            vector<Mat> domainBlock = findReducedDomainBlock(processedVideo, rangeBlock);
            Scalar average = getVideoAverage(domainBlock);
            for (int z = 0; z < rangeBlock->numFrames; z++) {
                //domainBlock[z].convertTo(domainBlock[z], CV_32F);
                domainBlock[z] = (domainBlock[z] - average) * rangeBlock->a.val[0] + rangeBlock->r.val[0];
                // Show outlines
                //showOutlines = rangeBlocks.size() > 800000;
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
                //if (showOutlines && i == numberIterations - 1) drawPartitions(domainBlock[z]);
                domainBlock[z].copyTo(currentVideo[z + rangeBlock->startFrame](
                        Rect(rangeBlock->startX, rangeBlock->startY, rangeBlock->size, rangeBlock->size)));
            }
        }

        iterations.emplace_back(vector<Mat>(currentVideo));
        currentVideo = vector<Mat>(iterations.back());
    }
    return iterations.back();
}

vector<Mat> generateVideo(const vector<Mat> &frames, const int &writeSize, const int &errorThreshold=0) {
    // PARAMS
    int startBlockSize = 32;
    int minBlockSize = 2;
    int numberIterations = 6;
    bool showOutlines = false;
    int blockSize = 32;
    cout << "Compressing block..." << endl;
    vector<Block *> transformations = compress(frames, startBlockSize, minBlockSize, errorThreshold);
    cout << "Done compressing" << endl;
    cout << "Decompressing block..." << endl;
    vector<Mat> decompressed = decompress(transformations,
                                          blockSize,
                                          writeSize,
                                          numberIterations,
                                          showOutlines);
    return decompressed;
}

int main() {
    int outputSize = 512;
    int writeSize = outputSize;
    int skipFrames = 360;
    int maxFrames = 32;

    string fileName = "bunny";
    string fileExtension = ".y4m";
    VideoCapture cap(fileName + fileExtension);

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << endl;
        return -1;
    }

    vector<Mat> frames;
    vector<Mat> channel0;
    vector<Mat> channel1;
    vector<Mat> channel2;
    VideoWriter video(fileName + "c.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 24, Size(writeSize, writeSize),
                      true);

    int frameCount = 0;
    int skippedFrames = 0;
    long transformationCount = 0;
    float totalTime = 0;
    double totalSSIM = 0;
    double totalPSNR = 0;
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
        Rect roi(500, 0, outputSize, outputSize);
        frames.push_back(frame(roi));
        Mat channels[3];
        Mat ycrcb;
        cvtColor(frame(roi), ycrcb, COLOR_BGR2YCrCb);
        split(ycrcb, channels);

        // Reduce the size of the non-luminance channels by 4, rounded to the nearest multiple of blockSize;
        //resize(channels[0], channels[0], Size(), 0.5, 0.5);
        resize(channels[1], channels[1], Size(), 0.5, 0.5);
        resize(channels[2], channels[2], Size(), 0.5, 0.5);

        channel0.push_back(channels[0]);
        channel1.push_back(channels[1]);
        channel2.push_back(channels[2]);


        if (frameCount % 32 == 0) {
            vector<Mat> decompressedChannel0;
            vector<Mat> decompressedChannel1;
            vector<Mat> decompressedChannel2;
            decompressedChannel0 = generateVideo(channel0, writeSize, 30);
            decompressedChannel1 = generateVideo(channel1, writeSize, 30);
            decompressedChannel2 = generateVideo(channel2, writeSize, 30);
            vector<Mat> newVideo;
            for (int i = 0; i < decompressedChannel0.size(); ++i) {
                vector<Mat> newChannels;
                cout << decompressedChannel0[i].type() <<endl;
                newChannels.push_back(decompressedChannel0[i]);
                newChannels.push_back(decompressedChannel1[i]);
                newChannels.push_back(decompressedChannel2[i]);
                Mat newFrame;
                merge(newChannels, newFrame);
                cvtColor(newFrame, newFrame, COLOR_YCrCb2BGR);
                newVideo.push_back(newFrame);
            }
            for (int i = 0; i < newVideo.size(); ++i) {
                //Mat filtered;
                //bilateralFilter(newVideo[i], filtered, 3, 20, 20);
                video.write(newVideo[i]);
                Mat compressedG;
                Mat videoG;
                cvtColor(newVideo[i], compressedG, COLOR_BGR2GRAY);
                cvtColor(frames[i], videoG, COLOR_BGR2GRAY);
                totalPSNR += getPSNR(compressedG, videoG);
                totalSSIM += getMSSIM(compressedG, videoG)[0];
            }
            frames.clear();
            newVideo.clear();
        }
    }
    cout << "Done." << endl;
    cout << "Total encoding time: " << totalTime << endl;
    cout << "Frames encoded per second: " << frameCount / totalTime << endl;
    cout << "Transformation count: " << transformationCount << endl;
    cout << "Compression ratio: " << (outputSize * outputSize * maxFrames * 8) / (double) (transformationCount * 11)
         << endl;
    cout << "SSIM: " <<  totalSSIM/(double) frameCount << endl;
    cout << "PSNR: " <<  totalPSNR/(double) frameCount << endl;

    cap.release();
    destroyAllWindows();

    return 0;
}

