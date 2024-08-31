import cv2
import pickle
import numpy as np

# import cv2
# import pickle
# import torch
# import torch.nn as nn
# import numpy as np
# import torchvision
# from torchvision.models import resnet18


# def get_name_to_module(model):
#     name_to_module = {}
#     for m in model.named_modules():
#         name_to_module[m[0]] = m[1]
#     return name_to_module


# def get_activation(all_outputs, name):
#     def hook(model, input, output):
#         all_outputs[name] = output.detach()

#     return hook


# def add_hooks(model, outputs, output_layer_names):
#     """
#     :param model:
#     :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
#     :param output_layer_names:
#     :return:
#     """
#     name_to_module = get_name_to_module(model)
#     for output_layer_name in output_layer_names:
#         name_to_module[output_layer_name].register_forward_hook(
#             get_activation(outputs, output_layer_name))


# class ModelWrapper(nn.Module):
#     def __init__(self, model, output_layer_names, return_single=True):
#         super(ModelWrapper, self).__init__()

#         self.model = model
#         self.output_layer_names = output_layer_names
#         self.outputs = {}
#         self.return_single = return_single
#         add_hooks(self.model, self.outputs, self.output_layer_names)

#     def forward(self, images):
#         self.model(images)
#         output_vals = [self.outputs[output_layer_name]
#                        for output_layer_name in self.output_layer_names]
#         if self.return_single:
#             return output_vals[0]
#         else:
#             return output_vals


# class BBResNet18(object):
#     def __init__(self):
#         self.model = resnet18(pretrained=True)
#         self.device = torch.device(
#             'cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.model.eval()

#         self.model = ModelWrapper(self.model, ['avgpool'], True)

#         self.model.eval()
#         self.model.to(self.device)

#     def feature_extraction(self, x: np.ndarray):
#         '''
#             param:
#                 x: numpy ndarray of shape: [None, 3, 224, 224] and dtype: np.float32

#             return:
#                 numpy ndarray (feature vector) of shape: [None, 512] and dtype: np.float32
#         '''

#         x = torch.from_numpy(x).to(self.device)

#         with torch.no_grad():
#             out = self.model(x).cpu().detach()
#             out = out.view(out.size(0), -1)
#             out = out.numpy()

#         return out


# ---------------------------------------------------------------------------------------------- #
# --------------------------------------- MODIFIED --------------------------------------------- #


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def resize_images(images, size=(224, 224)):
    n, h, w, c = images.shape
    resized_images = np.empty((n, size[0], size[1], c), dtype=np.float32)
    for i in range(n):
        resized_images[i, :, :, :] = cv2.resize(images[i, :, :, :], size)
    return resized_images


if __name__ == '__main__':

    # load augmented and test data from pickled file
    augmented_set = unpickle('augmented_set.pkl')
    print("Shape of augmented set: ", augmented_set.shape,
          "dtype: ", augmented_set.dtype)

    test_set = unpickle('test_set.pkl')
    print("Shape of test set: ", test_set.shape, "dtype: ", test_set.dtype)

    # resize images to 224x224
    resized_images = resize_images(augmented_set[0:1000])
    print("Shape of resized images: ", resized_images.shape,
          "dtype: ", resized_images.dtype)

    test_images = resize_images(test_set)
    print("Shape of resized test images: ",
          test_images.shape, "dtype: ", test_images.dtype)

    # change shape of images to [None, 3, 224, 224]
    transposed_images = np.transpose(resized_images, (0, 3, 1, 2))
    print("Shape of input images: ", resized_images.shape)

    print("Shape of input transposed images: ", transposed_images.shape)

    # compare red channel of original and transposed images
    print("Red channel of both images are same: ",
          (resized_images[0, :, :, 2] == transposed_images[0, 2, :, :]).all())

    # test_images = np.transpose(test_images, (0, 3, 1, 2))
    # print("Shape of input test images: ", test_images.shape)

    # # extract features
    # model = BBResNet18()

    # # extract feature vector in batches of 2500 images
    # features = []
    # for i in range(0, resized_images.shape[0], 2500):
    #     features.append(model.feature_extraction(resized_images[i:i+2500]))
    # features = np.concatenate(features)
    # print("Shape of features: ", features.shape)

    # test_vectors = []
    # for i in range(0, test_images.shape[0], 2500):
    #     test_vectors.append(model.feature_extraction(test_images[i:i+2500]))
    # test_vectors = np.concatenate(test_vectors)
    # print("Shape of test vectors: ", test_vectors.shape)

    # # save features and labels to pickle file
    # with open('features.pkl', 'wb') as f:
    #     pickle.dump(features, f)

    # with open('test_vectors.pkl', 'wb') as f:
    #     pickle.dump(test_vectors, f)
