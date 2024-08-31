# Every function takes an image as input and returns a new transformed image

# Every function is vectorized and uses numpy arrays to speed up the process

# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
import pickle

# load the CIFAR-10 dataset


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_info = unpickle('ASS1/cifar-10-batches-py/batches.meta')

batch1 = unpickle('ASS1/cifar-10-batches-py/data_batch_1')
batch2 = unpickle('ASS1/cifar-10-batches-py/data_batch_2')
batch3 = unpickle('ASS1/cifar-10-batches-py/data_batch_3')
batch4 = unpickle('ASS1/cifar-10-batches-py/data_batch_4')
batch5 = unpickle('ASS1/cifar-10-batches-py/data_batch_5')

test_batch = unpickle('ASS1/cifar-10-batches-py/test_batch')

print("Shape of data in batches", batch1[b'data'].shape)


# reshape the data to be 10000x32x32x3

batch1[b'data'] = batch1[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
batch2[b'data'] = batch2[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
batch3[b'data'] = batch3[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
batch4[b'data'] = batch4[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
batch5[b'data'] = batch5[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

test_batch[b'data'] = test_batch[b'data'].reshape(
    -1, 3, 32, 32).transpose(0, 2, 3, 1)

print("Shape of data in batches after reshape", batch1[b'data'].shape)


# This image will be used for demonstration
test_img = batch1[b'data'][0]
plt.imshow(test_img)
plt.title("Test Image")
plt.show()


# Image Enhancement Function
def enhancement(img):
    img_copy = img.copy()
    min_r = img_copy[:, :, 0].min()
    max_r = img_copy[:, :, 0].max()
    min_g = img_copy[:, :, 1].min()
    max_g = img_copy[:, :, 1].max()
    min_b = img_copy[:, :, 2].min()
    max_b = img_copy[:, :, 2].max()
    img_copy[:, :, 0] = ((img_copy[:, :, 0]-min_r)/max_r-min_r)*255
    img_copy[:, :, 1] = ((img_copy[:, :, 1]-min_g)/max_g-min_g)*255
    img_copy[:, :, 2] = ((img_copy[:, :, 2]-min_b)/max_b-min_b)*255
    return img_copy


plt.imshow(enhancement(test_img))
plt.title("Image Enhancement")
plt.show()


# Image Posterization
min_p = 63
max_p = 190
r = max_p - min_p
divider = 255/r


def posterization(img):
    img_copy = img.copy()
    img_copy = img_copy/divider
    img_copy = img_copy + min_p
    img_copy = img_copy.astype(np.uint8)
    return img_copy


plt.imshow(posterization(test_img))
plt.title("Image Posterization")
plt.show()


# Image Rotation
height = 32
width = 32
channels = 3
X = np.arange(width, dtype=np.uint8)
Y = np.arange(height, dtype=np.uint8)
coordinates = np.array(np.meshgrid(X, Y)).T.reshape(-1, 2)
c1 = np.array([0, 0])
c2 = np.array([height, width])


def randomrot(img):
    angle = np.random.uniform()*360-180
    radians = np.deg2rad(angle)
    rotation_matrix = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    center = np.array([height//2, width//2])
    rotated_coordinates = (
        np.dot(coordinates-center, rotation_matrix).round() + center).astype(np.uint8)
    img_rotated = np.zeros((height, width, channels), dtype=np.uint8)
    feasible_idx = np.all(rotated_coordinates >= c1, axis=1) & np.all(
        rotated_coordinates < c2, axis=1)
    feasible_r = rotated_coordinates[feasible_idx]
    feasible = coordinates[feasible_idx]
    img_rotated[feasible[:, 1], feasible[:, 0],
                :] = img[feasible_r[:, 1], feasible_r[:, 0], :]
    return img_rotated


plt.imshow(randomrot(test_img))
plt.title("Image Rotation")
plt.show()


# Image Contrast and Horizontal Flip
def cont_hor(img):
    alpha = np.random.uniform(0.5, 2.0)
    img_con = np.empty_like(img)
    img_con = np.clip(alpha*(img - 128) + 128, 0, 255).astype(np.uint8)
    if np.random.randint(2):
        img_con = np.fliplr(img_con)
    return img_con


plt.imshow(cont_hor(test_img))
plt.title("Image Contrast and Horizontal Flip")
plt.show()


# Creating Augmented Dataset
func = [enhancement, posterization, randomrot, cont_hor]


def randomfunc(img):
    return func[np.random.randint(4)](img)


randomfunc = np.vectorize(randomfunc, signature='(h,w,c)->(h,w,c)')


augmented_batch1 = np.empty_like(batch1[b'data'])
augmented_batch1 = randomfunc(batch1[b'data'])

augmented_batch2 = np.empty_like(batch2[b'data'])
augmented_batch2 = randomfunc(batch2[b'data'])

augmented_batch3 = np.empty_like(batch3[b'data'])
augmented_batch3 = randomfunc(batch3[b'data'])

augmented_batch4 = np.empty_like(batch4[b'data'])
augmented_batch4 = randomfunc(batch4[b'data'])

augmented_batch5 = np.empty_like(batch5[b'data'])
augmented_batch5 = randomfunc(batch5[b'data'])

plt.imshow(augmented_batch1[-1])
plt.title("Augmented Image from Batch 1")
plt.show()


# Creating Training and Augmented Training set
training_set = np.concatenate(
    (batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']), axis=0)
training_lbl = np.concatenate((batch1[b'labels'], batch2[b'labels'],
                              batch3[b'labels'], batch4[b'labels'], batch5[b'labels']), axis=0)

augmented_training_set = np.concatenate(
    (augmented_batch1, augmented_batch2, augmented_batch3, augmented_batch4, augmented_batch5), axis=0)
augmented_training_lbl = np.concatenate(
    (batch1[b'labels'], batch2[b'labels'], batch3[b'labels'], batch4[b'labels'], batch5[b'labels']), axis=0)

augmented_set = np.concatenate((training_set, augmented_training_set), axis=0)
augmented_lbl = np.concatenate((training_lbl, augmented_training_lbl), axis=0)

test_set = test_batch[b'data']
test_labels = np.array(test_batch[b'labels'])


randon_idx = np.random.randint(100000)
plt.imshow(augmented_set[randon_idx])
str = data_info[b'label_names'][augmented_lbl[randon_idx]].decode()
plt.title("Random Image of from Augmented Set of"+str)
plt.show()


# Saving the Augmented set
with open('augmented_set.pkl', 'wb') as f:
    pickle.dump(augmented_set, f)

with open('augmented_lbl.pkl', 'wb') as f:
    pickle.dump(augmented_lbl, f)

# Saving the Test set
with open('test_set.pkl', 'wb') as f:
    pickle.dump(test_set, f)

with open('test_lbl.pkl', 'wb') as f:
    pickle.dump(test_labels, f)
