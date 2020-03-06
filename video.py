import skvideo.datasets
import skvideo.io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import blockcompression
import numpy as np
import pickle
import compresssion

video = cv2.VideoCapture("chungus128.mp4")
frame_count = 0
ret, frame = video.read()
grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
transformations = []

max_frames = 30
frame_block_length = 10
while video.isOpened():
    frame_block = []
    for i in range(frame_block_length):
        ret, frame = video.read()
        frame_count += 1
        if np.shape(frame) == () or frame_count >= max_frames:
            print(frame_count)
            video.release()
            break
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_block.append(grayFrame[:128, :128])


    transformations.append(blockcompression.compress_block(frame_block, 8))

pickle.dump(transformations, open("transformations2.pkl", "wb"))

output_resolution = 128
if not transformations:
    transformations = pickle.load(open("transformations2.pkl", "rb"))
decompressedVideo = blockcompression.decompress(transformations, output_resolution, output_resolution, 66, 10, 8)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('chungus128c.mp4', fourcc, 24, (output_resolution, output_resolution), isColor=False)
print(np.shape(decompressedVideo))
for frame_block in decompressedVideo:
    for frame in frame_block:
        out.write(frame.astype(np.uint8))


cv2.destroyAllWindows()

# img = compresssion.reduce(mpimg.imread('lena512.bmp'), 4)
# transformations.append(blockcompression.compress_block([img], 8))