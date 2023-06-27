# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from jax import scipy as sp
from jax import numpy as jnp
import dataclasses
import functools
from typing import Callable, Optional
from jax.example_libraries import optimizers
import jax
import jax.config
from jax.config import config as jax_config
import neural_tangents as nt
from neural_tangents import stax
from .get_network import FullyConnectedNetwork, FullyConvolutionalNetwork, MyrtleNetwork
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
import scipy.linalg as sp_linalg
# for numerical stability, can disable if not an issue
jax_config.update('jax_enable_x64', True)

# architecture params
# !@param ['FC', 'Conv', 'Myrtle']; choice of neural network architecture yielding the corresponding NTK
# ARCHITECTURE = 'FC'
# DEPTH = 1  # @param {'type': int}; depth of neural network
# !# @param {'type': int}; width of finite width neural network; only used if parameterization is 'standard'
# WIDTH = 1024
# #! @param ['ntk', 'standard']; whether to use standard or NTK parameterization, see https://arxiv.org/abs/2001.07301
# PARAMETERIZATION = 'ntk'

# #! dataset
# DATASET = 'cifar10'  # @param ['cifar10', 'cifar100', 'mnist', 'svhn_cropped']

# #! training params
# LEARNING_RATE = 4e-2  # @param {'type': float};
# SUPPORT_SIZE = 50  # @param {'type': int}; number of images to learn
# #! @param {'type': int}; number of target images to use in KRR for each step
# TARGET_BATCH_SIZE = 5000
# #! @param {'type': bool}; whether to optimize over support labels during training
# LEARN_LABELS = False


class Distiller():
    def __init__(self, itr=300, ARCHITECTURE='FC', DEPTH=1, WIDTH=1024,
                 PARAMETERIZATION='ntk', DATASET='cifar10', LEARNING_RATE=1e-4,
                 SUPPORT_SIZE=50, TARGET_BATCH_SIZE=5000, LEARN_LABELS=False,save_path='results_kip'):
        self.itr = itr
        self.ARCHITECTURE = ARCHITECTURE
        self.DEPTH = DEPTH
        self.WIDTH = WIDTH
        self.PARAMETERIZATION = PARAMETERIZATION
        self.DATASET = DATASET
        self.LEARNING_RATE = LEARNING_RATE
        self.SUPPORT_SIZE = SUPPORT_SIZE
        self.TARGET_BATCH_SIZE = TARGET_BATCH_SIZE
        self.LEARN_LABELS = LEARN_LABELS
        self.save_path = save_path

    def get_tfds_dataset(self, name):
        ds_train, ds_test = tfds.as_numpy(
            tfds.load(
                name,
                split=['train', 'test'],
                batch_size=-1,
                as_dataset_kwargs={'shuffle_files': False}))

        return ds_train['image'], ds_train['label'], ds_test['image'], ds_test['label']

    def one_hot(sef, x,
                num_classes,
                center=True,
                dtype=np.float32):
        assert len(x.shape) == 1
        one_hot_vectors = np.array(x[:, None] == np.arange(num_classes), dtype)
        if center:
            one_hot_vectors = one_hot_vectors - 1. / num_classes
        return one_hot_vectors

    def get_normalization_data(self, arr):
        channel_means = np.mean(arr, axis=(0, 1, 2))
        channel_stds = np.std(arr, axis=(0, 1, 2))
        return channel_means, channel_stds

    def normalize(self, array, mean, std):
        return (array - mean) / std

    def unnormalize(self, array, mean, std):
        return (array * std) + mean

    @dataclasses.dataclass
    class Augmentor:
        """Class for creating augmentation function."""

        # function applied after augmentations (maps uint8 image to float image)
        # if standard preprocessing, this should be function which does channel-wise
        # standardization
        preprocessing_function: Callable[[np.ndarray], np.ndarray]

        # need this to unnormalize images if they are already normalized
        # before applying augmentations
        channel_means: Optional[np.ndarray] = None
        channel_stds: Optional[np.ndarray] = None

        # Specify these to augment at custom rate
        augmentation_frac: float = 1.0
        rotation_range: float = 0.0
        width_shift_range: float = 0.0
        height_shift_range: float = 0.0
        horizontal_flip: bool = False
        channel_shift_range: float = 0.0

        def __post_init__(self):
            self.aug_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=self.rotation_range,
                width_shift_range=self.width_shift_range,
                height_shift_range=self.height_shift_range,
                horizontal_flip=self.horizontal_flip,
                channel_shift_range=self.channel_shift_range,
                preprocessing_function=self.preprocessing_function,
            )

        def __call__(self,
                     x: np.ndarray,
                     normalized: bool = True,
                     seed: Optional[int] = None):
            """Augments a numpy array of images.

            Args:
            x: image array (B,H,W,C)
            normalized: if True, then image is assumed to be standard preprocessed and
                therefore must be unnormalized before augmented
            seed: random seed for augmentations

            Returns:
            augmented images
            """

            if self.augmentation_frac == 0.0:
                return x

            permutation = np.random.permutation(x.shape[0])
            inv_permutation = self.get_inverse_permutation(permutation)
            num_aug_images = int(self.augmentation_frac * x.shape[0])

            x = x[permutation]
            if normalized:
                x_raw = self.unnormalize(
                    x, self.channel_means, self.channel_stds)
            else:
                x_raw = x

            iterator = self.aug_generator.flow(  # pytype: disable=attribute-error
                x_raw[:num_aug_images],
                batch_size=num_aug_images,
                shuffle=False,
                seed=seed)
            x_aug = next(iterator)
            x_aug = np.concatenate([x_aug, x[num_aug_images:]])
            return x_aug[inv_permutation]

    def get_inverse_permutation(self, perm):
        array = np.zeros_like(perm)
        array[perm] = np.arange(len(perm), dtype=int)
        return array

    # define architectures
    def get_kernel_fn(self, architecture, depth, width, parameterization):
        if architecture == 'FC':
            return FullyConnectedNetwork(depth=depth, width=width, parameterization=parameterization)
        elif architecture == 'Conv':
            return FullyConvolutionalNetwork(depth=depth, width=width, parameterization=parameterization)
        elif architecture == 'Myrtle':
            return MyrtleNetwork(depth=depth, width=width, parameterization=parameterization)
        else:
            raise NotImplementedError(
                f'Unrecognized architecture {architecture}')

    def class_balanced_sample(self, sample_size: int,
                              labels: np.ndarray,
                              *arrays: np.ndarray, **kwargs: int):
        """Get random sample_size unique items consistently from equal length arrays.

        The items are class_balanced with respect to labels.

        Args:
        sample_size: Number of elements to get from each array from arrays. Must be
            divisible by the number of unique classes
        labels: 1D array enumerating class label of items
        *arrays: arrays to sample from; all have same length as labels
        **kwargs: pass in a seed to set random seed

        Returns:
        A tuple of indices sampled and the corresponding sliced labels and arrays
        """
        if labels.ndim != 1:
            raise ValueError(
                f'Labels should be one-dimensional, got shape {labels.shape}')
        n = len(labels)
        if not all([n == len(arr) for arr in arrays[1:]]):
            raise ValueError(
                f'All arrays to be subsampled should have the same length. Got lengths {[len(arr) for arr in arrays]}')
        classes = np.unique(labels)
        n_classes = len(classes)
        n_per_class, remainder = divmod(sample_size, n_classes)

        # if remainder != 0:
        #     raise ValueError(
        #         f'Number of classes {n_classes} in labels must divide sample size {sample_size}.'
        #     )
        if kwargs.get('seed') is not None:
            np.random.seed(kwargs['seed'])
        min_value_in_class = 999
        class_have_min_value = None
        for c in classes:
            size_of_this_class = np.where(labels == c)[0].size
            if size_of_this_class < min_value_in_class:
                min_value_in_class = size_of_this_class
                class_have_min_value = c
        if min_value_in_class < 20:
            new_classes = [c for c in classes if c != class_have_min_value]
            classes = new_classes
        
        min_value_in_class = 999
        for c in classes:
            size_of_this_class = np.where(labels == c)[0].size
            if size_of_this_class < min_value_in_class:
                min_value_in_class = size_of_this_class
                class_have_min_value = c        
        
        # Gán giá trị cho X và làm tròn
        n_per_class = round((2/3) * min_value_in_class)
        if kwargs.get('save_img_mode') is not None:
            # n_per_class = classes.size
            # print(f'n_per_class: {n_per_class}')
            n_per_class = 1
        inds = np.concatenate([
            np.random.choice(np.where(labels == c)[
                             0], n_per_class, replace=False)
            for c in classes
        ])
        # return (1, 1) + tuple(
        #     [arr.copy() for arr in arrays])
        return (inds, classes,labels[inds].copy()) + tuple(
            [arr[inds].copy() for arr in arrays])

    def save_image_synthetic(self, sample_raw: np.ndarray, sample_init: np.ndarray, sample_final:np.ndarray, num_classes: int):
        save_path = f'{self.save_path}/synthetic.png'
        fig = plt.figure(figsize=(33, 10))
        fig.suptitle('Image comparison.\n\nRow 1: Original uint8.  Row2: Original normalized.  Row 3: KIP learned images.', fontsize=16, y=1.02)

        for i, img in enumerate(sample_raw):
            plt.subplot(3, num_classes, i+1)
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(np.squeeze(img))

        for i, img in enumerate(sample_init, 1):
            plt.subplot(3, num_classes, num_classes + i)
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(np.squeeze(img))

        for i, img in enumerate(sample_final, 1):
            plt.subplot(3, num_classes, num_classes * 2 + i)
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(np.squeeze(img))

        plt.savefig(save_path)
        print(f'Done save image to {save_path}')


    # def make_loss_acc_fn(self, kernel_fn):

    #     @jax.jit
    #     def loss_acc_fn(x_support, y_support, x_target, y_target, reg=1e-6):
    #         y_support = jax.lax.cond(
    #             self.LEARN_LABELS, lambda y: y, jax.lax.stop_gradient, y_support)
    #         k_ss = kernel_fn(x_support, x_support)
    #         k_ts = kernel_fn(x_target, x_support)
    #         k_ss_reg = (k_ss + jnp.abs(reg) * jnp.trace(k_ss) *
    #                     jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
    #         pred = jnp.dot(k_ts, sp.linalg.solve(
    #             k_ss_reg, y_support, sym_pos=True))
    #         mse_loss = 0.5*jnp.mean((pred - y_target) ** 2)
    #         acc = jnp.mean(jnp.argmax(pred, axis=1) ==
    #                        jnp.argmax(y_target, axis=1))
    #         return mse_loss, acc

    #     return loss_acc_fn

    def make_loss_acc_fn(self, kernel_fn):

        @jax.jit
        def loss_acc_fn(x_support, y_support, x_target, y_target, reg=1e-6, weight_decay=1e-4):
            y_support = jax.lax.cond(
                self.LEARN_LABELS, lambda y: y, jax.lax.stop_gradient, y_support)
            k_ss = kernel_fn(x_support, x_support)
            k_ts = kernel_fn(x_target, x_support)
            
            # Regularization
            reg_term = jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0]
            
            # Weight Decay
            weight_decay_term = weight_decay * jnp.eye(k_ss.shape[0])
            
            # Add regularization and weight decay to k_ss
            k_ss_reg = k_ss + reg_term + weight_decay_term
            
            # Compute predictions
            pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, sym_pos=True))
            
            # Compute mean squared error loss
            mse_loss = 0.5 * jnp.mean((pred - y_target) ** 2)
            
            # Compute accuracy
            acc = jnp.mean(jnp.argmax(pred, axis=1) == jnp.argmax(y_target, axis=1))
            
            return mse_loss, acc

        return loss_acc_fn

    def get_update_functions(self, init_params, kernel_fn, lr):
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(init_params)
        loss_acc_fn = self.make_loss_acc_fn(kernel_fn)
        value_and_grad = jax.value_and_grad(lambda params, x_target, y_target: loss_acc_fn(params['x'],
                                                                                           params['y'],
                                                                                           x_target,
                                                                                           y_target), has_aux=True)

        @jax.jit
        def update_fn(step, opt_state, params, x_target, y_target):
            (loss, acc), dparams = value_and_grad(params, x_target, y_target)
            return opt_update(step, dparams, opt_state), (loss, acc)

        return opt_state, get_params, update_fn

    def distill(self, log_freq=20, seed=1,  X_TRAIN_RAW=None, LABELS_TRAIN=None, X_TEST_RAW=None, LABELS_TEST=None,
                additional_message = None):

        import pdb; pdb.set_trace()

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)  

        if isinstance(X_TRAIN_RAW, torch.Tensor):
            X_TRAIN_RAW = X_TRAIN_RAW.detach().cpu().numpy()

        if isinstance(LABELS_TRAIN, torch.Tensor):
            LABELS_TRAIN = LABELS_TRAIN.detach().cpu().numpy()

        if isinstance(X_TEST_RAW, torch.Tensor):
            X_TEST_RAW = X_TEST_RAW.detach().cpu().numpy()

        if isinstance(LABELS_TEST, torch.Tensor):
            LABELS_TEST = LABELS_TEST.detach().cpu().numpy()
        

        best_val_result = -999
        best_image_syn = None
        best_label_syn = None


        channel_means, channel_stds = self.get_normalization_data(X_TRAIN_RAW)

        X_TRAIN, X_TEST = self.normalize(X_TRAIN_RAW, channel_means, channel_stds), self.normalize(
            X_TEST_RAW, channel_means, channel_stds)
        Y_TRAIN, Y_TEST = self.one_hot(
            LABELS_TRAIN, 10), self.one_hot(LABELS_TEST, 10)

        _, _, kernel_fn = self.get_kernel_fn(
            self.ARCHITECTURE, self.DEPTH, self.WIDTH, self.PARAMETERIZATION)
        KERNEL_FN = jax.jit(functools.partial(kernel_fn, get='ntk'))

        channel_means, channel_stds = self.get_normalization_data(X_TRAIN_RAW)

        _,_,  labels_init, x_init_raw, y_init = self.class_balanced_sample(
            self.SUPPORT_SIZE, LABELS_TRAIN, X_TRAIN_RAW, Y_TRAIN, seed=seed)
        x_init = self.normalize(
            x_init_raw, channel_means, channel_stds)
        params_init = {'x': x_init, 'y': y_init}
        params_init_raw = {'x': x_init_raw, 'y': labels_init}

        opt_state, get_params, update_fn = self.get_update_functions(
            params_init, KERNEL_FN, self.LEARNING_RATE)
        params = get_params(opt_state)
        loss_acc_fn = self.make_loss_acc_fn(KERNEL_FN)

        # compute in batches for expensive kernels
        test_loss, test_acc = loss_acc_fn(
            params['x'], params['y'], X_TEST, Y_TEST)
        print('initial test loss:', test_loss)
        print('initial test acc:', test_acc)
        best_data_synthetic = None
        for i in range(1, self.itr+1):
            # full batch gradient descent
            _, _,_, x_target_batch, y_target_batch = self.class_balanced_sample(
                self.TARGET_BATCH_SIZE, LABELS_TRAIN, X_TRAIN, Y_TRAIN)
            opt_state, aux = update_fn(
                i, opt_state, params, x_target_batch, y_target_batch)
            train_loss, train_acc = aux
            params = get_params(opt_state)
            if i % log_freq == 0:
                print(f'----step {i}:')
                print('train loss:', train_loss)
                print('train acc:', train_acc)
                print(additional_message)
                # compute in batches for expensive kernels
                test_loss, test_acc = loss_acc_fn(
                    params['x'], params['y'], X_TEST, Y_TEST)
                if test_acc > best_val_result:
                    best_val_result = test_acc
                    # x_tensor = torch.tensor(params['x'])
                    best_data_synthetic = params
                    x_np = np.array(params['x'])
                    y_np = np.array(params['y'])
                    torch.save(y_np, f'{self.save_path}/x_distill.pt')
                    torch.save(y_np, f'{self.save_path}/y_distill.pt')

                print('test loss:', test_loss)
                print('test acc:', test_acc)
        _, classes_response,_, sample_raw, sample_init, sample_final = self.class_balanced_sample(
                10, params_init_raw['y'], params_init_raw['x'], params_init['x'], best_data_synthetic['x'], seed=3, save_img_mode=True)
        self.save_image_synthetic(sample_raw, sample_init, sample_final,len(classes_response))
        # return params, params_init, params_init_raw
        print(f'best_val_result {best_val_result}')

        # return params['x'],params['y']
        return best_data_synthetic['x'],best_data_synthetic['y']


    def run(self):
        X_TRAIN_RAW, LABELS_TRAIN, X_TEST_RAW, LABELS_TEST = self.get_tfds_dataset(
            self.DATASET)
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        number_dataset_need_get = 500
        # _, _, LABELS_TRAIN, X_TRAIN_RAW = self.class_balanced_sample(
        #     500, LABELS_TRAIN, LABELS_TRAIN, X_TRAIN_RAW)

        # _, _, LABELS_TEST, X_TEST_RAW = self.class_balanced_sample(
        #     500, LABELS_TEST, LABELS_TEST, X_TEST_RAW)

        # params_final, params_init, params_init_raw = self.distill(
        #     X_TRAIN_RAW=X_TRAIN_RAW, LABELS_TRAIN=LABELS_TRAIN, X_TEST_RAW=X_TEST_RAW, LABELS_TEST=LABELS_TEST)
        x_predict, y_predict = self.distill(
            X_TRAIN_RAW=X_TRAIN_RAW, LABELS_TRAIN=LABELS_TRAIN, X_TEST_RAW=X_TEST_RAW, LABELS_TEST=LABELS_TEST)
        # images = params_final['x']
        # print(images.shape)
        # labels = params_final['y']
        # print(labels.shape)

        # _, _, sample_raw, sample_init, sample_final = self.class_balanced_sample(
        #     10, params_init_raw['y'], params_init_raw['x'], params_init['x'], params_final['x'], seed=3)
        # class_names = ['airplane', 'automobile', 'bird', 'cat',
        #                'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # fig = plt.figure(figsize=(33, 10))
        # fig.suptitle(
        #     'Image comparison.\n\nRow 1: Original uint8.  Row2: Original normalized.  Row 3: KIP learned images.', fontsize=16, y=1.02)
        # for i, img in enumerate(sample_raw):
        #     ax = plt.subplot(3, 10, i+1)
        #     ax.set_title(class_names[i])
        #     plt.imshow(np.squeeze(img))

        # for i, img in enumerate(sample_init, 1):
        #     plt.subplot(3, 10, 10+i)
        #     plt.imshow(np.squeeze(img))

        # for i, img in enumerate(sample_final, 1):
        #     plt.subplot(3, 10, 20+i)
        #     plt.imshow(np.squeeze(img))
        # plt.savefig('synthetic.png')

    # params_final, params_init, params_init_raw = train(300)


# if __name__ == "__main__":
#     distiller = Distiller(itr=300, TARGET_BATCH_SIZE=5000)
#     distiller.run()
