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
    Block *q1;
    Block *q2;
    Block *q3;
    Block *q4;

    Block(int startFrame, int startX, int startY, int size, int numFrames = 0) : startFrame{startFrame}, startX{startX},
                                                                                 startY{startY}, size{size}, numFrames{
                    numFrames == 0 ? size : numFrames} {}
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
        Mat roi(rangeBlock.size, rangeBlock.size, CV_8U);
        for (int j = 0; j < domainBlockSize; j += 2) {
            for (int k = 0; k < domainBlockSize; k += 2) {
                roi.at<uchar>(k / 2, j / 2) = video[i].at<uchar>(k + domainStartYCoordinate, j + domainStartXCoordinate);
            }
        }
        domainBlock.push_back(roi);
    }
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
    a = 0.5;
    float maxFrameError = INT_MIN;
    for (float trialA = -0.5; trialA <= 0.5; trialA += 0.125) {
        error = 0;
        maxFrameError = 0;
        float count = 0;
        for (int z = 0; z < domainBlock.size(); ++z) {
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
        }
        if (maxFrameError < minError) {
            minError = maxFrameError;
            a = trialA;
        }
    }
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
            newErrorThreshold = errorThreshold;
        Scalar a;
        Scalar r;

        float error = findParamsAndError(rangeBlockVideo, domainBlockVideo, newErrorThreshold, a, r);
        rangeBlock.a = a;
        rangeBlock.r = r;
        if (error < newErrorThreshold || rangeBlock.numFrames == 1) {
            transformations.push_back(rangeBlock);
        } else {
            if (rangeBlock.size != minRangeSize) {
                int newRangeSize = rangeBlock.size / 2; int newNumFrames = newRangeSize;
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

vector<Mat> generateVideo(const vector<Mat> &frames, const int &outputSize) {
    // PARAMS
    int startBlockSize = 32;
    int minBlockSize = 2;
    int errorThreshold = 10;
    int numberIterations = 6;
    bool showOutlines = 0;
    int blockSize = 32;
    cout << "Compressing block..." << endl;
    vector<Block> transformations = compress(frames, startBlockSize, minBlockSize, errorThreshold);
    cout << "Done compressing" << endl;
    cout << "Decompressing block..." << endl;
    vector<Mat> decompressed = decompress(transformations,
                                          blockSize,
                                          outputSize,
                                          numberIterations,
                                          showOutlines);
    cout << "Done decompressing" << endl;
    return decompressed;
}

int main() {
    int outputSize = 1024;
    int skipFrames = 0;
    int maxFrames = 32;

    string fileName = "boat";
    string fileExtension = ".mp4";
    VideoCapture cap("/Users/Austin/dev/fcode/" + fileName + fileExtension);

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << endl;
        return -1;
    }

    vector<Mat> frames;
    vector<Mat> channel0;
    vector<Mat> channel1;
    vector<Mat> channel2;
    VideoWriter video(fileName + "c.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 24, Size(outputSize, outputSize),
                      true);

    int frameCount = 0;
    int skippedFrames = 0;
    long transformationCount = 0;
    int blockSize = 32;
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
        frames.push_back(frame(Rect(0, 0, outputSize, outputSize)));
        Mat channels[3];
        Mat ycrcb;
        //cvtColor(frame(Rect(0, 0, outputSize, outputSize)), frame);
        //cvtColor(frame(Rect(0, 0, outputSize, outputSize)), frame, COLOR_BGR2YCrCb);
        cvtColor(frame(Rect(0, 0, outputSize, outputSize)), ycrcb, COLOR_RGB2YCrCb);
        split(frame(Rect(0, 0, outputSize, outputSize)), channels);

        // Reduce the size of the non luminance channels.
        //resize(channels[1], channels[1], Size(), 1, 0.25);
        //resize(channels[2], channels[2], Size(), 0.25, 0.25);

        channel0.push_back(channels[0]);
        channel1.push_back(channels[1]);
        channel2.push_back(channels[2]);


        vector<Mat> decompressedChannel0;
        vector<Mat> decompressedChannel1;
        vector<Mat> decompressedChannel2;
        vector<Mat> videoFrames;
        if (frameCount % 32 == 0) {
            decompressedChannel0 = generateVideo(channel0, outputSize);
            decompressedChannel1 = generateVideo(channel1, outputSize);
            decompressedChannel2 = generateVideo(channel2, outputSize);
            vector<Mat> newVideo;
            for (int i = 0; i < decompressedChannel0.size(); ++i) {
                vector<Mat> newChannels;
                newChannels.push_back(decompressedChannel0[i]);
                newChannels.push_back(decompressedChannel1[i]);
                newChannels.push_back(decompressedChannel2[i]);
                Mat newFrame;
                merge(newChannels, newFrame);
                //cvtColor(newFrame, newFrame, COLOR_YCrCb2BGR);
                //cvtColor(newFrame, newFrame, COLOR_BGR2GRAY);
                newVideo.push_back(newFrame);
            }
            for (int i = 0; i < newVideo.size(); ++i) {
                video.write(newVideo[i]);
                totalPSNR += getPSNR(frames[i], newVideo[i]);
                totalSSIM += getMSSIM(frames[i], newVideo[i])[0];
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



//int main() {
//
//    // PARAMS
//    int startBlockSize = 32;
//    int minBlockSize = 2;
//    int errorThreshold = 70;
//    int numberIterations = 6;
//    bool showOutlines = 0;
//    int outputSize = 704;
//    int skipFrames = 0;
//    int maxFrames = 1248;
//
//    string fileName = "sintel";
//    string fileExtension = ".y4m";
//    VideoCapture cap("/Users/Austin/dev/fcode/" + fileName + fileExtension);
//
//    if (!cap.isOpened()) {
//        std::cout << "Error opening video stream or file" << endl;
//        return -1;
//    }
//
//    vector<Mat> frames;
//    VideoWriter video(fileName + "c.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 24, Size(outputSize, outputSize),
//                      false);
//
//    int frameCount = 0;
//    int skippedFrames = 0;
//    long transformationCount = 0;
//    int blockSize = 32;
//    float totalTime = 0;
//    double totalSSIM = 0;
//    double totalPSNR = 0;
//    vector<Mat[3]> splitFrames;
//    while (true) {
//        Mat frame, greyFrame;
//        while (skippedFrames < skipFrames) {
//            cap.grab();
//            ++skippedFrames;
//        }
//        if (frameCount > maxFrames && maxFrames != 0)
//            break;
//
//        cap >> frame;
//        if (frame.empty()) {
//            break;
//        }
//        ++frameCount;
//        Mat channels[3];
//        Mat ycrcb;
//        cvtColor(frame, ycrcb, COLOR_BGR2YCrCb);
//        split(ycrcb, channels);
//        frames.push_back(frame);
//        //frames.push_back(greyFrame(Rect(0, 0, outputSize, outputSize)));
//        if (frames.size() % 32 == 0) {
//            cout << "Compressing block..." << endl;
//            chrono::steady_clock::time_point begin = chrono::steady_clock::now();
//            vector<Block> transformations = compress(frames, startBlockSize, minBlockSize, errorThreshold);
//            chrono::steady_clock::time_point end = chrono::steady_clock::now();
//            float seconds = (float) chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000000;
//            totalTime += seconds;
//            cout << "Done compressing" << endl;
//            cout << "Transformation count: " << transformations.size() << endl;
//            cout << "Compression ratio: "
//                 << (outputSize * outputSize * blockSize * 8) / (float) (transformations.size() * 11) << endl
//                 << endl;
//            transformationCount += transformations.size();
//            cout << "Decompressing block..." << endl;
//            vector<Mat> decompressed = decompress(transformations,
//                                                  blockSize,
//                                                  outputSize,
//                                                  numberIterations,
//                                                  showOutlines);
//            cout << "Done decompressing" << endl
//                 << endl;
//            for (int i = 0; i < decompressed.size(); ++i) {
//                video.write(decompressed[i]);
//                totalPSNR += getPSNR(frames[i], decompressed[i]);
//                totalSSIM += getMSSIM(frames[i], decompressed[i])[0];
//            }
//            frames.clear();
//        }
//    }
//    cout << "Done." << endl;
//    cout << "Total encoding time: " << totalTime << endl;
//    cout << "Frames encoded per second: " << frameCount / totalTime << endl;
//    cout << "Transformation count: " << transformationCount << endl;
//    cout << "Compression ratio: " << (outputSize * outputSize * maxFrames * 8) / (double) (transformationCount * 11)
//         << endl;
//    cout << "SSIM: " <<  totalSSIM/(double) frameCount << endl;
//    cout << "PSNR: " <<  totalPSNR/(double) frameCount << endl;
//
//    cap.release();
//    destroyAllWindows();
//
//    return 0;
//}
