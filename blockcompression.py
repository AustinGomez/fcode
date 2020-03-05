from collections import defaultdict

import numpy as np
from scipy import ndimage
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import compresssion

np.seterr('raise')


def find_brightness_fixed_contrast(range_block, domain_block):
    contrast = 0.75
    brightness = np.sum(range_block - contrast * domain_block) / range_block.size
    return contrast, brightness


def fit_contrast_and_brightness(range_block, domain_block):
    # Fit the contrast and the brightness
    A = np.concatenate((np.ones((domain_block.size, 1)), np.reshape(domain_block, (domain_block.size, 1))), axis=1)
    b = np.reshape(range_block, (range_block.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    # x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
    return x[1], x[0]


# In this case, reflection means flipping each individual frame. NOT reversing the frames in any way. <<<< NOT TRUE
def reflect(frames, direction):
    if direction != 1:
        new_frames = []
        for frame in frames:
            new_frames.append(frame[::direction, :])
        return np.array(new_frames)[::direction, :]
    else:
        return frames


def rotate(frames, angle):
    new_frames = []
    for frame in frames:
        new_frames.append(ndimage.rotate(frame, angle, reshape=False))
    return np.array(new_frames)


def decimate_frames(frames, factor=2):
    return ndimage.zoom(frames, (1, 0.5, 0.5))

    # print(frames.shape)
    new_frames = []
    for frame in frames:
        result = np.zeros((frame.shape[0] // factor, frame.shape[1] // factor))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.mean(frame[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor])
        new_frames.append(result)

    return np.array(new_frames)


def apply_transformation(frames, direction, angle, contrast=1.0, brightness=0.0):
    transformed_frames = rotate(reflect(frames, direction), angle)
    new_frames = []
    for frame in transformed_frames:
        new_frames.append(contrast * frame + brightness)
    return np.array(new_frames)


def distance(block_1, block_2):
    n = block_1.shape[1]  # width / height.
    M = block_1.shape[0]  # Number of Frames

    difference = np.subtract(block_1, block_2)
    error = 0
    for i in range(difference.shape[0]):
        for j in range(difference.shape[1]):
            for k in range(difference.shape[2]):
                error += difference[i][j][k] ** 2

    return ((1 / (n * M ** 0.5)) * error) ** 0.5


# Each frame must be square.
def generate_domain_blocks(frame_block, domain_block_size, range_to_domain_ratio=2):
    transformed_blocks = []
    number_of_frames, height, width = np.shape(frame_block)
    step = domain_block_size
    for k in range((height - domain_block_size) // domain_block_size + 1):
        for l in range((width - domain_block_size) // domain_block_size + 1):
            domain_block = decimate_frames(np.array(frame_block)[:, k * step:(k + 1) * step, l * step:(l + 1) * step])
            for direction, angle in all_possible_transformations:
                transformed_blocks.append(
                    (apply_transformation(domain_block, direction, angle), k, l, direction, angle))

    return transformed_blocks


def compress_block(frame_block, domain_block_size, range_to_domain_ratio=2):
    domain_blocks = generate_domain_blocks(frame_block, domain_block_size, range_to_domain_ratio)
    transformations = []
    number_of_frames, height, width = np.shape(frame_block)
    counter = 0
    range_block_size = domain_block_size // range_to_domain_ratio
    for i in range(height // range_block_size):
        transformations.append([])
        for j in range(width // range_block_size):
            print("{}/{} ; {}/{}".format(i, height // range_block_size, j, width // range_block_size))
            transformations[i].append([])
            min_distance = float('inf')
            range_block = np.array(frame_block)[:, i * range_block_size:(i + 1) * range_block_size,
                          j * range_block_size:(j + 1) * range_block_size]
            for domain_block, k, l, direction, angle in domain_blocks:
                contrast, brightness = fit_contrast_and_brightness(range_block, domain_block)
                counter += 1
                # if counter % 2 == 0:
                #     continue

                d2 = distance(range_block, domain_block)
                # d=np.sum(np.square(domain_block-range_block))
                # print(d, d2)
                if d2 < min_distance:
                    min_distance = d2
                    transformations[i][j] = (k, l, direction, angle, contrast, brightness)
                    # if d < 4:
                    #     break

    return transformations


# To the i,jth  domain block, repeatedly apply decimation and then transformation[i][j].
# set the i,jth range block to be the decimated, transformed, transform[i][j]th domain block.
def decompress(transformations, output_width, output_height, num_frames, frame_block_length=10, number_iterations=20):
    # return compresssion.decompress(transformations[0], 8, 4, 8, 30)
    video = [np.random.randint(0, output_width, (num_frames, output_height, output_width))]
    domain_blocks = defaultdict(int)
    frames_so_far = 0
    while frames_so_far < num_frames:
        current_video = np.zeros((frame_block_length, output_height, output_width))
        iterations = [np.random.randint(0, output_width, (frame_block_length, output_height, output_width))]
        print(frames_so_far)
        for iteration in range(number_iterations):
            print(frames_so_far // frame_block_length)
            current_block_transformations = transformations[frames_so_far // frame_block_length]
            num_blocks_height = len(current_block_transformations[0]) // 2
            domain_block_size = output_height // num_blocks_height
            range_block_size = domain_block_size // 2
            height = len(current_block_transformations[0])
            width = len(current_block_transformations[0])
            # print(height, width)
            for i in range(height):
                for j in range(width):
                    k, l, direction, angle, contrast, brightness = current_block_transformations[i][j]
                    domain_blocks[(k, l)] += 1
                    domain_block = decimate_frames(np.array(iterations[-1])[:,
                                                   k * domain_block_size:(k + 1) * domain_block_size,
                                                   l * domain_block_size:(l + 1) * domain_block_size])
                    transformed_domain_block = apply_transformation(domain_block, direction, angle, contrast,
                                                                    brightness)
                    current_video[:,
                    i * range_block_size:(i + 1) * range_block_size,
                    j * range_block_size:(j + 1) * range_block_size] = transformed_domain_block

            iterations.append(current_video)
            current_video = np.zeros((frame_block_length, output_height, output_width))
        print("append")
        video.append(iterations[-1])
        frames_so_far += frame_block_length

    print(domain_blocks)
    print("video shape", np.shape(video))
    return np.array(video[1:])  # First frame is garbage


# To the i,jth  domain block, repeatedly apply decimation and then transformation[i][j].
# set the i,jth range block to be the decimated, transformed, transform[i][j]th domain block.
# def decompress(transformations, output_width, output_height, num_frames, frame_block_length=10, number_iterations=20):
#     # return compresssion.decompress(transformations[0], 8, 4, 8, 30)
#     video = [np.random.randint(0, 512, (num_frames, output_height, output_width))]
#     iterations = [np.random.randint(0, 512, (frame_block_length, output_height, output_width))]
#     current_video = np.zeros((frame_block_length, output_height, output_width))
#     domain_blocks = defaultdict(int)
#
#     for iteration in range(number_iterations):
#         print(np.shape(iterations))
#         frames_so_far = 0
#         while frames_so_far < num_frames:
#             current_block_transformations = transformations[frames_so_far // frame_block_length]
#             num_blocks_height = len(current_block_transformations[0])//2
#             domain_block_size = output_height // num_blocks_height
#             range_block_size = domain_block_size // 2
#             height = len(current_block_transformations[0])
#             width = len(current_block_transformations[0])
#             print(height, width)
#             for i in range(height):
#                 for j in range(width):
#                     k, l, direction, angle, contrast, brightness = current_block_transformations[i][j]
#                     domain_blocks[(k,l)] += 1
#                     # print(i,j)
#                     domain_block = decimate_frames(np.array(iterations[-1])[frames_so_far:frames_so_far + frame_block_length,
#                                    k * domain_block_size:(k + 1) * domain_block_size,
#                                    l * domain_block_size:(l + 1) * domain_block_size])
#                     transformed_domain_block = apply_transformation(domain_block, direction, angle, contrast, brightness)
#                     current_video[frames_so_far:frames_so_far + frame_block_length,
#                     i * range_block_size:(i + 1) * range_block_size,
#                     j * range_block_size:(j + 1) * range_block_size] = transformed_domain_block
#
#             iterations.append(current_video)
#             current_video = np.zeros((frame_block_length, output_height, output_width))
#             frames_so_far += frame_block_length
#         video.append(iterations[-1])
#     print(domain_blocks)
#     return np.array(video)

directions = [-1, 1]
angles = [0, 90, 180, 270]
# directions = [1]
# angles = [0]
all_possible_transformations = [(direction, angle) for direction in directions for angle in angles]
