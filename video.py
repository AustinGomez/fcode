import skvideo.datasets
import skvideo.io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import nosearchblock
import numpy as np
import pickle
import compresssion
from timeit import default_timer as timer
from datetime import timedelta
video = cv2.VideoCapture("chungustrimmed.mp4")
frame_count = 0
ret, frame = video.read()
grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
transformations = []

max_frames = 33
frame_block_length = 10
# frames = []
# while video.isOpened():
#     for i in range(max_frames):
#         ret, frame = video.read()
#         frame_count += 1
#         if np.shape(frame) == () or frame_count >= max_frames:
#             print(frame_count)
#             video.release()
#             break
#         grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frames.append(grayFrame[:512, :512])
#
# start = timer()
# transformations = nosearchblock.octtree_compress(frames, 16, error_threshold=4, min_range_size=2)
# end = timer()
# print(timedelta(seconds=end-start))
#
#
# pickle.dump(transformations, open("transformations2.pkl", "wb"))

output_resolution = 512
if not transformations:
    transformations = pickle.load(open("transformations2.pkl", "rb"))

start = timer()
decompressed_video = nosearchblock.octtree_decompress(transformations, output_resolution, num_frames=33, number_iterations=10, factor=1)
end = timer()
print(timedelta(seconds=end-start))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('chungus512c.mp4', fourcc, 24, (output_resolution, output_resolution), isColor=False)
print(np.shape(decompressed_video))
for frame in decompressed_video:
        out.write(frame.astype(np.uint8))


video.release()
cv2.destroyAllWindows()

# img = compresssion.reduce(mpimg.imread('lena512.bmp'), 4)
# transformations.append(blockcompression.compress_block([img], 8))