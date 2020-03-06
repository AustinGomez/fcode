import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage, signal
import numpy as np
import math
from collections import deque

class RangeBlock:
    def __init__(self, size, start_frame, start_x, start_y):
        self.start_x = start_x
        self.start_y = start_y
        self.start_frame = start_frame
        self.size = size


def reduce(frames, factor=2):
    return ndimage.zoom(frames, 1/factor, order=0)


def distance(range_block, domain_block, error_threshold=0):
    domain_block_average = np.average(domain_block)
    range_block_average = np.average(range_block)
    min_error = float('inf')
    best_a = 0
    D = domain_block - domain_block_average
    R = range_block - range_block_average
    for a in range(24):
        error = np.linalg.norm(a * D - R) ** 2
        if error < min_error:
            min_error = error
            best_a = a
        if min_error < error_threshold:
            break
    print(error, error_threshold)
    return min_error, best_a, range_block_average


# Maybe add scene detection here.
def find_domain_start(range_block):
    domain_size = range_block.size * 2
    domain_start_frames = max(range_block.start_frame - domain_size // 2, 0)
    domain_startx = max(range_block.start_x - domain_size // 2, 0)
    domain_starty = max(range_block.start_y - domain_size // 2, 0)

    domain_end_frames = domain_start_frames + domain_size
    domain_endx = domain_startx + domain_size
    domain_endy = domain_starty + domain_size

    return domain_startx, domain_starty, domain_endx, domain_endy, domain_start_frames, domain_end_frames


#instead of storing size, store level.
def octtree_compress(img, range_size, error_threshold=0, min_range_size=2):
    transformations = []
    img = np.array(img)
    width = height = img.shape[1]
    num_frames = img.shape[0]
    range_blocks = [RangeBlock(range_size, start_frame, i, j)
                    for start_frame in range(0, num_frames, range_size)
                    for i in range(0, height, range_size)
                    for j in range(0, width, range_size)]
    uncovered_range_blocks = deque(range_blocks)
    while len(uncovered_range_blocks):
        if len(uncovered_range_blocks) % 300 == 0:
            print(len(uncovered_range_blocks))
        range_block = uncovered_range_blocks.popleft()
        range_block_frames = img[
                             range_block.start_frame:range_block.start_frame + range_block.size,
                             range_block.start_x:range_block.start_x + range_block.size,
                             range_block.start_y:range_block.start_y + range_block.size
                             ]

        domain_startx, domain_starty, domain_endx, domain_endy, domain_start_frames, domain_end_frames = find_domain_start(
            range_block)

        domain_block_frames = img[
                                  domain_start_frames:domain_end_frames,
                                  domain_starty:domain_endy,
                                  domain_startx:domain_endx
                              ]
        reduced_domain_block_frames = reduce(domain_block_frames)

        new_error_threshold = error_threshold  ** (range_size/range_block.size) + 1
        error, a, r = distance(range_block_frames, reduced_domain_block_frames, new_error_threshold)
        if error < new_error_threshold or range_block.size == min_range_size:
            transformations.append(
                (range_block.start_frame, range_block.start_x, range_block.start_y, range_block.size, a, r))
        else:
            start_frame = range_block.start_frame
            start_x = range_block.start_x
            start_y = range_block.start_y

            # split into octants.
            new_range_size = range_block.size // 2
            oct_1 = RangeBlock(new_range_size, start_frame, start_x, start_y)
            oct_2 = RangeBlock(new_range_size, start_frame, start_x + new_range_size, start_y)
            oct_3 = RangeBlock(new_range_size, start_frame, start_x, start_y + new_range_size)
            oct_4 = RangeBlock(new_range_size, start_frame, start_x + new_range_size, start_y + new_range_size)

            uncovered_range_blocks += [oct_1, oct_2, oct_3, oct_4]
            if start_frame + new_range_size < img.shape[0]:
                oct_5 = RangeBlock(new_range_size, start_frame + new_range_size, start_x, start_y)
                oct_6 = RangeBlock(new_range_size, start_frame + new_range_size, start_x + new_range_size, start_y)
                oct_7 = RangeBlock(new_range_size, start_frame + new_range_size, start_x, start_y + new_range_size)
                oct_8 = RangeBlock(new_range_size, start_frame + new_range_size, start_x + new_range_size,
                                   start_y + new_range_size)

                uncovered_range_blocks += [oct_5, oct_6, oct_7, oct_8]

    return transformations


def octtree_decompress(transformations, output_size, num_frames, number_iterations=8, factor=1):
    iterations = [np.random.randint(0, output_size, (num_frames, output_size, output_size))]
    cur_video = np.zeros((num_frames, output_size, output_size))
    print(len(transformations))
    for iteration in range(number_iterations):
        print(iteration)
        for start_frame, start_x, start_y, range_block_size, a, r in transformations:
            range_block = RangeBlock(range_block_size * factor, start_frame * factor, start_x * factor,
                                     start_y * factor)
            domain_startx, domain_starty, domain_endx, domain_endy, domain_start_frame, domain_end_frame = find_domain_start(
                range_block)
            S = reduce(iterations[-1][domain_start_frame:domain_end_frame, domain_starty:domain_endy,
                       domain_startx:domain_endx])
            D = S * a + r
            cur_video[
                start_frame:start_frame + range_block_size * factor,
                start_x:start_x + range_block_size * factor,
                start_y:start_y + range_block_size * factor
            ] = D
        iterations.append(cur_video)
        cur_video = np.zeros((num_frames, output_size, output_size))
    return iterations[-1]


# Tests
def test_greyscale():
    img = reduce(mpimg.imread('lena512.bmp'))
    transformations = []
    plt.figure()
    # plt.imshow(img, cmap='gray', interpolation='none')
    transformations = octtree_compress(img, 16, 2, 4)
    pickle.dump(transformations, open("transformationsNoSearch.pkl", "wb"))
    if not transformations:
        transformations = pickle.load(open("transformationsNoSearch.pkl", "rb"))
    iterations = octtree_decompress(transformations, 4, 2048, 12, factor=4)
    # mpimg.imsave('lena1028.bmp', iterations[-1], cmap='gray')
    # iterations512 = decompress(transformations, 16, 8, 16, new_size=512)
    # mpimg.imsave('lena256compressed.bmp', iterations[-1],  vmin=0, vmax=255, cmap='gray')
    plt.imshow(iterations[-1], vmin=0, vmax=255, cmap='gray')
    # plot_iterations(iterations, img)
    plt.show()


if __name__ == '__main__':
    test_greyscale()
    # test_rgb()
