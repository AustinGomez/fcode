import pickle
from collections import deque
from skimage import color, io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import math
from cv2 import imread
# Transformations


def reduce(img):
    return ndimage.zoom(img, 0.5)

def find_contrast_and_brightness2(D, S):
    # Fit the contrast and the brightness
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    # x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
    print(x[1], x[0])
    return x[1], x[0]


def distance(range_block, domain_block, error_threshold):
    #a, b = find_contrast_and_brightness2(range_block, domain_block)
    #error = np.linalg.norm(a * D - R) ** 2
    #return error, a, b
    min_error = float('inf')
    domain_block_average = np.average(domain_block)
    range_block_average = np.average(range_block)
    D = domain_block - domain_block_average
    R = range_block - range_block_average
    domain_block_size = np.shape(domain_block)[0]
    range_block_size = domain_block_size
    d21 = domain_block[:, :domain_block_size//2]
    D21 = d21 - np.average(d21)
    r21 = range_block[:, :range_block_size // 2]
    R21 = r21 - np.average(r21)

    for a in range(-5, 6):
        a = a * (1/8)
        #error = np.linalg.norm(a * D21 - R21) ** 2
        # if error > error_threshold and error >= min_error:
        if False:
            continue
        else:
            error = np.linalg.norm(a * D - R) ** 2

        if error < min_error:
            min_error = error
            best_a = a


    return min_error, best_a, range_block_average


class RangeBlock:
    def __init__(self, size, start_x, start_y, a=0, r=0, covered=False):
        self.covered = covered
        self.start_x = start_x
        self.start_y = start_y
        self.size = size
        self.a = a
        self.r = r


def find_domain_start(range_block):
    # return range_block.start_x // 2, range_block.start_y // 2, range_block.start_x // 2 + range_block.size, range_block.start_y // 2 + range_block.size
    domain_size = range_block.size * 2
    domain_startx = max(range_block.start_x - domain_size // 2, 0)
    domain_starty = max(range_block.start_y - domain_size // 2, 0)

    domain_endx = domain_startx + domain_size
    domain_endy = domain_starty + domain_size

    return domain_startx, domain_starty, domain_endx, domain_endy


def find_domain_start_no_block(start_x, start_y, range_block_size, img_width, img_height):
    domain_size = range_block_size * 2
    domain_startx = max(start_x - domain_size // 2, 0)
    domain_starty = max(start_y - domain_size // 2, 0)
    domain_endx = domain_startx + domain_size
    domain_endy = domain_starty + domain_size
    return domain_startx, domain_starty, domain_endx, domain_endy


def preprocess(img):
    result = np.zeros((img.shape[0], img.shape[1]))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.mean(img[i:i + 1, j:(j + 1)])
    return result

def quadtree_compress(img, range_size, error_threshold=0, min_range_size=2):
    transformations = []
    img = np.array(img)
    width = height = img.shape[0]
    domain_size = range_size * 2
    range_blocks = [RangeBlock(range_size, i, j)
                    for i in range(0, height, range_size)
                    for j in range(0, width, range_size)]
    uncovered_range_blocks = deque(range_blocks)
    reduced_img = preprocess(img)
    while len(uncovered_range_blocks):
        range_block = uncovered_range_blocks.popleft()
        range_block_img = img[
                          range_block.start_y:range_block.start_y + range_block.size,
                          range_block.start_x:range_block.start_x + range_block.size
                          ]

        domain_startx, domain_starty, domain_endx, domain_endy = find_domain_start(range_block)
        domain_block = reduced_img[domain_starty:domain_endy:2, domain_startx:domain_endx:2]
        # a, r = fit_contrast_and_brightness(range_block_img, reduced_domain_block)
        level = int(math.log(range_size / range_block.size, 2))
        new_error_threshold = (2 ** level) * error_threshold + (2 ** level) - 1
        error, a, r = distance(range_block_img, domain_block, new_error_threshold)
        if error < new_error_threshold or range_block.size == min_range_size:
            # if level in (1, 2):
            #     print ("FOUND", level, error, a)
            transformations.append((range_block.start_x, range_block.start_y, range_block.size, a, r))
        else:
            start_x = range_block.start_x
            start_y = range_block.start_y

            new_range_size = range_block.size // 2
            quad_1 = RangeBlock(new_range_size, start_x, start_y)
            quad_2 = RangeBlock(new_range_size, start_x + new_range_size, start_y)
            quad_3 = RangeBlock(new_range_size, start_x, start_y + new_range_size)
            quad_4 = RangeBlock(new_range_size, start_x + new_range_size, start_y + new_range_size)

            uncovered_range_blocks += [quad_1, quad_2, quad_3, quad_4]

    return transformations


def quadtree_decompress(transformations, range_size, output_size, number_iterations=9, factor=1):
    iterations = [np.random.randint(0, output_size, (output_size, output_size))]
    cur_img = np.zeros((output_size, output_size))
    print(len(transformations))
    for iteration in range(number_iterations):
        print(iteration)
        reduced_iteration = preprocess(iterations[-1])
        for start_x, start_y, range_block_size, a, r in transformations:
            domain_startx, domain_starty, domain_endx, domain_endy = find_domain_start_no_block(start_x*factor, start_y*factor, range_block_size*factor, output_size, output_size)
            S = reduced_iteration[domain_starty:domain_endy:2, domain_startx:domain_endx:2]

            domain_average = np.average(S)
            D = (S - domain_average) * a + r
            cur_img[
                start_y * factor:(start_y + range_block_size) * factor,
                start_x * factor:(start_x + range_block_size) * factor,
            ] = D
        iterations.append(cur_img)
        cur_img = np.zeros((output_size, output_size))
    return iterations


# Tests
def test_greyscale():
    filename = "lena512.bmp"
    img = imread(filename, 0)
    plt.imshow(img, cmap="gray")
    #img = mpimg.imread('lena512.bmp')[:128,:128]
    transformations = []
    plt.figure()
    # plt.imshow(img, cmap='gray', interpolation='none')
    transformations = quadtree_compress(img, 16, 50, 2)
    pickle.dump(transformations, open("transformationsNoSearch.pkl", "wb"))
    if not transformations:
        transformations = pickle.load(open("transformationsNoSearch.pkl", "rb"))
    iterations = quadtree_decompress(transformations, 4, 512, number_iterations=12, factor=1)
    # mpimg.imsave('lena1028.bmp', iterations[-1], cmap='gray')
    # iterations512 = decompress(transformations, 16, 8, 16, new_size=512)
    mpimg.imsave('lena512compressed.bmp', iterations[-1],  vmin=0, vmax=255, cmap='gray')
    # plt.imshow(iterations[-1],  vmin=0, vmax=255, cmap='gray')
    plot_iterations(iterations, img)
    plt.show()


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


if __name__ == '__main__':
    test_greyscale()
    # test_rgb()
