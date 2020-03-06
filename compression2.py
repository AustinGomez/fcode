import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy import optimize
import numpy as np
import math
import skvideo.io


# Manipulate channels

def get_greyscale_image(img):
    return img


def extract_rgb(img):
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def assemble_rbg(img_r, img_g, img_b):
    shape = (img_r.shape[0], img_r.shape[1], 1)
    return np.concatenate((np.reshape(img_r, shape), np.reshape(img_g, shape),
                           np.reshape(img_b, shape)), axis=2)


# Transformations

def reduce(img):
    return ndimage.zoom(img, (0.5, 0.5))


def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False)


def flip(img, direction):
    return img[::direction, :]


def apply_transformation(img, direction, angle, contrast=1.0, brightness=0.0):
    return contrast * rotate(flip(img, direction), angle) + brightness


def fit_contrast_and_brightness(D, S):
    # Fit the contrast and the brightness
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    # x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
    return x[1], x[0]


def distance(range_block, domain_block, error_threshold=0):
    domain_block_average = np.average(domain_block)
    range_block_average = np.average(range_block)
    min_error = float('inf')
    best_a = 0
    D = domain_block - domain_block_average
    R = range_block - range_block_average
    for a in range(32):
        error = np.linalg.norm(a * D - R) ** 2
        if error < min_error:
            min_error = error
            best_a = a
        if min_error < error_threshold:
            break

    return min_error, best_a, range_block_average

class RangeBlock:
    def __init__(self, size, start_x, start_y, a=0, r=0, covered=False):
        self.covered = covered
        self.start_x = start_x
        self.start_y = start_y
        self.size = size
        self.a = a
        self.r = r

def find_domain_start(range_block, img_width, img_height):
    domain_size = range_block.size * 2
    domain_startx = max(range_block.start_x - domain_size // 2, 0)
    domain_starty = max(range_block.start_y - domain_size // 2, 0)

    domain_endx = domain_startx + domain_size
    domain_endy = domain_starty + domain_size

    return domain_startx, domain_starty, domain_endx, domain_endy

def quadtree_compress(img, range_size, error_threshold=0, min_range_size=2):
    transformations = []
    img = np.array(img)
    width = height = img.shape[0]
    domain_size = range_size * 2
    range_blocks = [RangeBlock(range_size, i, j)
                    for i in range(0, height, range_size)
                    for j in range(0, width, range_size)]
    uncovered_range_blocks = range_blocks
    while len(uncovered_range_blocks):
        range_block = uncovered_range_blocks[0]
        range_block_img = img[
                          range_block.start_x:range_block.start_x + range_block.size,
                          range_block.start_y:range_block.start_y + range_block.size
                          ]

        domain_startx, domain_starty, domain_endx, domain_endy = find_domain_start(range_block, width, height)

        domain_block = img[domain_starty:domain_endy, domain_startx:domain_endx]
        reduced_domain_block = reduce(domain_block)
        # a, r = fit_contrast_and_brightness(range_block_img, reduced_domain_block)
        error, a, r = distance(range_block_img, reduced_domain_block, error_threshold * 2 ** (range_size/range_block.size))
        if error < error_threshold * (2*(range_size/range_block.size)) or range_block.size == min_range_size:
            del uncovered_range_blocks[0]
            transformations.append((range_block.start_x, range_block.start_y, range_block.size, a, r))
        else:
            del uncovered_range_blocks[0]
            start_x = range_block.start_x
            start_y = range_block.start_y

            new_range_size = range_block.size // 2
            quad_1 = RangeBlock(new_range_size, start_x, start_y)
            quad_2 = RangeBlock(new_range_size, start_x + new_range_size, start_y)
            quad_3 = RangeBlock(new_range_size, start_x, start_y + new_range_size)
            quad_4 = RangeBlock(new_range_size, start_x + new_range_size, start_y + new_range_size)

            uncovered_range_blocks += [quad_1, quad_2, quad_3, quad_4]

    return transformations


def quadtree_decompress(transformations, range_size, output_size, number_iterations=12, factor=1):
    iterations = [np.random.randint(0, output_size, (output_size, output_size))]
    cur_img = np.zeros((output_size, output_size))
    print(len(transformations))
    for iteration in range(number_iterations):
        print(iteration)
        for start_x, start_y, range_block_size, a, r in transformations:
            range_block = RangeBlock(range_block_size*factor, start_x*factor, start_y*factor, a, r)
            domain_startx, domain_starty, domain_endx, domain_endy = find_domain_start(range_block, output_size, output_size)
            S = reduce(iterations[-1][domain_starty:domain_endy, domain_startx:domain_endx])
            D = S * a + r
            cur_img[range_block.start_x:range_block.start_x + range_block_size*factor,
            range_block.start_y:range_block.start_y + range_block_size*factor] = D
        iterations.append(cur_img)
        cur_img = np.zeros((output_size, output_size))
    return iterations


# Tests
def test_greyscale():
    img = reduce(mpimg.imread('lena512.bmp'))
    transformations = []
    img = get_greyscale_image(img)
    plt.figure()
    # plt.imshow(img, cmap='gray', interpolation='none')
    # transformations = quadtree_compress(img, 32, 10, 2)
    # pickle.dump(transformations, open("transformationsNoSearch.pkl", "wb"))
    if not transformations:
        transformations = pickle.load(open("transformationsNoSearch.pkl", "rb"))
    iterations = quadtree_decompress(transformations, 4, 1028, 12, factor=2)
    # mpimg.imsave('lena1028.bmp', iterations[-1], cmap='gray')
    # iterations512 = decompress(transformations, 16, 8, 16, new_size=512)
    mpimg.imsave('lena256compressed.bmp', iterations[-1],  vmin=0, vmax=255, cmap='gray')
    plt.imshow(iterations[-1],  vmin=0, vmax=255, cmap='gray')
    plot_iterations(iterations, img)
    plt.show()

# Instead i should decompress with transformations[i][j]

# def quadtree_compress(img, range_size, error_threshold, depth, max_depth):
#     img = np.array(img)
#     num_blocks_height = num_blocks_width = img.shape[0] // range_size
#     range_blocks = [img[i * range_size:(i + 1) * range_size, j * range_size:(j + 1) * range_size]
#                     for i in range(num_blocks_height) for j in range(num_blocks_width)]
#     range_blocks = []
#     for i in range(num_blocks_height):
#         range_blocks.append([])
#         for j in range(num_blocks_width):
#             range_blocks.append([i * range_size (i + 1) * range_size, j * range_size:(j + 1) * range_size)])
#     range_blocks_covered = [False for block in range_blocks]
#
#     while range_blocks_covered
#
#
# def quadtree_decompress(transformations, range_size, output_size, number_iterations=8):
#     iterations = [np.random.randint(0, output_size, (output_size, output_size))]
#     cur_img = np.zeros((output_size, output_size))
#     print(np.shape(transformations))
#     for iteration in range(number_iterations):
#         for index, domain_startx, domain_starty, domain_endx, domain_endy, brightness, contrast in transformations:
#             # Apply transform
#             new_range_size = (domain_endx - domain_startx) // 2
#             x, y = index
#             # print(domain_endx, domain_startx)
#             # print(new_range_size, x, y)
#
#             S = reduce(iterations[-1][domain_starty:domain_endy, domain_startx:domain_endx])
#             D = S * contrast + brightness
#             cur_img[x:x + new_range_size,
#             y:y + new_range_size] = D
#         iterations.append(cur_img)
#         cur_img = np.zeros((output_size, output_size))
#     return iterations


def decompress(transformations, source_size, destination_size, step, nb_iter=2, new_size=512):
    factor = source_size // destination_size
    height = len(transformations) * destination_size
    width = len(transformations[0]) * destination_size
    iterations = [np.random.randint(0, new_size, (height, width))]
    cur_img = np.zeros((height, width))
    for i_iter in range(nb_iter):
        print(i_iter)
        for i in range(len(transformations)):
            for j in range(len(transformations[i])):
                if len(transformations[i][j]) == 1:
                    print(np.shape(transformations[i][j]))
                else:
                    # Apply transform
                    domain_startx, domain_starty, domain_endx, domain_endy, brightness, contrast = transformations[i][j]
                    S = reduce(iterations[-1][domain_starty:domain_endy, domain_startx:domain_endx], factor)
                    D = S * contrast + brightness
                    cur_img[i * destination_size:(i + 1) * destination_size,
                    j * destination_size:(j + 1) * destination_size] = D
        iterations.append(cur_img)
        cur_img = np.zeros((height, width))

    return iterations


# Compression for color images

def reduce_rgb(img, factor):
    img_r, img_g, img_b = extract_rgb(img)
    img_r = reduce(img_r, factor)
    img_g = reduce(img_g, factor)
    img_b = reduce(img_b, factor)
    return assemble_rbg(img_r, img_g, img_b)


def compress_rgb(img, source_size, destination_size, step):
    img_r, img_g, img_b = extract_rgb(img)
    return [compress(img_r, source_size, destination_size, step), \
            compress(img_g, source_size, destination_size, step), \
            compress(img_b, source_size, destination_size, step)]


def decompress_rgb(transformations, source_size, destination_size, step, nb_iter=8):
    img_r = decompress(transformations[0], source_size, destination_size, step, nb_iter)[-1]
    img_g = decompress(transformations[1], source_size, destination_size, step, nb_iter)[-1]
    img_b = decompress(transformations[2], source_size, destination_size, step, nb_iter)[-1]
    return assemble_rbg(img_r, img_g, img_b)


# Plot

def plot_iterations(iterations, target=None):
    # Configure plot
    plt.figure()
    nb_row = math.ceil(np.sqrt(len(iterations)))
    nb_cols = nb_row
    # Plot
    for i, img in enumerate(iterations):
        plt.subplot(nb_row, nb_cols, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        if target is None:
            plt.title(str(i))
        else:
            # Display the RMSE
            plt.title(str(i) + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(target - img)))) + ')')
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()


# Parameters

directions = [1, -1]
directionsy = [1, -1]
angles = [0, 90, 180, 270]
candidates = [[direction, angle] for direction in directions for angle in angles]





def test_rgb():
    img = mpimg.imread('lena.gif')
    img = reduce_rgb(img, 8)
    transformations = compress_rgb(img, 8, 4, 8)
    retrieved_img = decompress_rgb(transformations, 8, 4, 8)
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.array(img).astype(np.uint8), interpolation='none')
    plt.subplot(122)
    plt.imshow(retrieved_img.astype(np.uint8), interpolation='none')
    plt.show()


if __name__ == '__main__':
    test_greyscale()
    # test_rgb()

# def compress(img, domain_size, range_size, error_threshold=4, depth=0, max_depth=2, start_index=0, step=8):
#     transformations = []
#     img = np.array(img)
#     for i in range(len(img) // range_size):
#         transformations.append([])
#         for j in range(len(img[i]) // range_size):
#             transformations[i].append([])
#             range_block = img[i * range_size:(i + 1) * range_size, j * range_size:(j + 1) * range_size]
#             domain_startx = max(j - domain_size // 2, 0)
#             domain_starty = max(i - domain_size // 2, 0)
#             domain_endx = min(len(img[i]), domain_startx + domain_size)
#             domain_endy = min(len(img), domain_starty + domain_size)
#
#             if domain_endx == len(img[i]):
#                 domain_startx = domain_endx - domain_size
#             if domain_endy == len(img):
#                 domain_starty = domain_endy - domain_size
#
#             domain_block = img[domain_starty:domain_endy, domain_startx:domain_endx]
#             contrast, brightness = fit_contrast_and_brightness(range_block, reduce(domain_block, 2))
#             error = distance(reduce(domain_block, 2) * contrast + brightness, range_block)
#             if error > error_threshold and depth != max_depth:
#                 transformations[i][j].append(
#                     compress(range_block, range_size, range_size // 2, error_threshold, depth + 1,
#                              max_depth))
#             else:
#                 transformations[i][j] = [domain_startx, domain_starty, domain_endx, domain_endy, brightness, contrast]
#
#     return transformations


# def decompress(transformations, source_size, destination_size, step, nb_iter=8, new_size=512):
#     factor = source_size // destination_size
#     height = len(transformations) * destination_size
#     width = len(transformations[0]) * destination_size
#     iterations = [np.random.randint(0, new_size, (height, width))]
#     cur_img = np.zeros((height, width))
#     for i_iter in range(nb_iter):
#         print(i_iter)
#         for i in range(len(transformations)):
#             for j in range(len(transformations[i])):
#                 if len(transformations[i][j]) == 2:  # this is a nested one.
#                     # Upper left quadrant
#                     cur_img[i * destination_size:((i + 1) * destination_size) // 2,
#                     j * destination_size:((j + 1) * destination_size) // 2] = decompress(transformations[i][j],
#                                                                                          destination_size,
#                                                                                          destination_size // 2,
#                                                                                          new_size=destination_size,
#                                                                                          step=destination_size)
#                     # Upper right quadrant
#                     cur_img[((i + 1) * destination_size) // 2:((i + 1) * destination_size),
#                     j * destination_size:((j + 1) * destination_size) // 2] = decompress(transformations[i][j],
#                                                                                          destination_size,
#                                                                                          destination_size // 2,
#                                                                                          new_size=destination_size,
#                                                                                          step=destination_size)
#                     cur_img[i * destination_size:((i + 1) * destination_size) // 2,
#                     ((j + 1) * destination_size) // 2:(j + 1) * destination_size] = decompress(transformations[i][j],
#                                                                                                destination_size,
#                                                                                                destination_size // 2,
#                                                                                                new_size=destination_size,
#                                                                                                step=destination_size)
#                     cur_img[((i + 1) * destination_size) // 2:((i + 1) * destination_size),
#                     ((j + 1) * destination_size) // 2:(j + 1) * destination_size] = decompress(transformations[i][j],
#                                                                                                destination_size,
#                                                                                                destination_size // 2,
#                                                                                                new_size=destination_size,
#                                                                                                step=destination_size)
#                 else:
#                     # Apply transform
#                     domain_startx, domain_starty, domain_endx, domain_endy, brightness, contrast = transformations[i][j]
#                     S = reduce(iterations[-1][domain_starty:domain_endy, domain_startx:domain_endx], factor)
#                     D = S * contrast + brightness
#                     cur_img[i * destination_size:(i + 1) * destination_size,
#                     j * destination_size:(j + 1) * destination_size] = D
#         iterations.append(cur_img)
#         cur_img = np.zeros((height, width))
#
#     return iterations
