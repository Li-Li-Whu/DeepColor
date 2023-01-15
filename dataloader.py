import math
from paddle.io import Dataset
import numpy as np
from PIL import Image
import glob
import random
import cv2
import paddle


#random.seed(1143)


def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
    train_list = image_list_lowlight
    random.shuffle(train_list)

    return train_list

class ImageDataset(Dataset):
    def __init__(self, root='/path/to/FlyingChairs_release/data', mode='train',prob_threshold=0.5):
        self.root_path = root
        self.mode = mode
        self.prob_threshold = prob_threshold
        self.visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-0.1, 0.1),
            hue_range=(0.8, 1.2),
            saturation_range=(0.95, 1.05)
        )  # 创建生成器

        self.train_list = populate_train_list(root)
        self.train_list.sort()
        self.size = 128
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))
        self.ref_list = []

        for i in range(0, len(self.train_list)):
            self.ref_list.append(self.train_list[i])

        self.data_list = self.train_list
        print("Total training examples:", len(self.ref_list))

    def __getitem__(self, index):
        ref_img_path = self.ref_list[index]
        ref_img = Image.open(ref_img_path)
        pic_name = ref_img_path.split('VOC\\')[-1].split('.jpg')[0]
        visual_effect = next(self.visual_effect_generator)
        tgt_img = visual_effect(np.asarray(ref_img)).astype(np.uint8)
        tgt_img = Image.fromarray(tgt_img)
        #img_size = ref_img.size
        if self.mode == 'test':
            probability = 1
        else:
            probability = random.random()
        if probability < self.prob_threshold:
            angle = random.randint(1,3)*90
            ref_img = ref_img.rotate(int(angle), expand=True)
            tgt_img = tgt_img.rotate(int(angle), expand=True)

        ref_img = ref_img.resize((self.size, self.size), Image.ANTIALIAS)
        tgt_img = tgt_img.resize((self.size, self.size), Image.ANTIALIAS)
        ref_img = self.norm_img(np.asarray(ref_img), self.mean_I, self.std_I).astype(np.float32)
        tgt_img = self.norm_img(np.asarray(tgt_img), self.mean_I, self.std_I).astype(np.float32)
        ref_img = ref_img.transpose((2, 0, 1))
        tgt_img = tgt_img.transpose((2, 0, 1))

        if self.mode == 'test':
            return ref_img, tgt_img, pic_name
        else:
            return ref_img, tgt_img

    def __len__(self):
        return len(self.ref_list)

    def norm_img(self, img, mean, std):
        img = (img - mean) / std
        return img




class VisualEffect:
    """
    生成从给定间隔均匀采用的视觉效果参数
    参数：
    contrast_factor: 对比度因子：调整对比度的因子区间，应该在0-3之间
    brightness_delta: 亮度增量：添加到像素的量在-1和1之间的间隔
    hue_delta:色度增量：为添加到色调通道的量在-1和1之间的间隔
    saturation_factor:饱和系数：因子乘以每个像素的饱和值的区间
    """
    def __init__(
            self,
            contrast_factor,
            brightness_delta,
            hue_range,
            saturation_factor
    ):
        self.contrast_factor = contrast_factor
        self.brightness_delta = brightness_delta
        self.hue_range = hue_range
        self.saturation_factor = saturation_factor

    def __call__(self, image):
        """
        将视觉效果应用到图片上
        """
        # if self.contrast_factor:
        #     image = self.adjust_contrast(image, self.contrast_factor)
        #
        # if self.brightness_delta:
        #     image = self.adjust_brightness(image, self.brightness_delta)

        if self.hue_range or self.saturation_factor:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 色彩空间转化
            if self.hue_range:
                image = self.adjust_hue(image, self.hue_range)
            # if self.saturation_factor:
            #     image = self.adjust_saturation(image, self.saturation_factor)

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 色彩空间转化
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def adjust_saturation(self, image, factor):
        """
        调整图片的饱和度
        """
        image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
        return image

    def adjust_hue(self, image, hue_range):
        """
        调整图片的色度
        添加到色调通道的量在-1和1之间的间隔。
        如果值超过180，则会旋转这些值。
        """
        delta0 = _uniform(hue_range)
        delta1 = _uniform(hue_range)
        delta2 = _uniform(hue_range)
        image = image.transpose(2,0,1)
        tensor = paddle.to_tensor(np.asarray(image)) / 255
        #tensor = paddle.transpose(tensor,(2, 0, 1)).float() / 255
        #tensor = paddle.transpose(tensor, (2, 0, 1))
        tensor[0, :, :] = tensor[0, :, :] * delta0
        tensor[1, :, :] = tensor[1, :, :] * delta1
        tensor[2, :, :] = tensor[2, :, :] * delta2
        image = paddle.clip(tensor * 255, min=0, max=255).numpy().transpose(1,2,0)#.byte().permute(1, 2, 0).cpu().numpy()
        # image[..., 0] = image[..., 0] * delta0
        # image[..., 1] = image[..., 1] * delta1
        # image[..., 2] = image[..., 2] * delta2
        return image

    def adjust_contrast(self, image, factor):
        """
        调整一张图像的对比度
        """
        mean = image.mean(axis=0).mean(axis=0)
        return self._clip((image - mean) * factor + mean)

    def adjust_brightness(self, image, delta):
        """
        调整一张图片的亮度
        """
        return self._clip(image + delta * 255)

    def _clip(self, image):
        """
        剪辑图像并将其转换为np.uint8
        """
        return np.clip(image, 0, 255).astype(np.uint8)


def _uniform(val_range):
    """
    随机返回值域之间的数值
    """
    return np.random.uniform(val_range[0], val_range[1])


def _check_range(val_range, min_val=None, max_val=None):
    """
    检查间隔是否有效
    """
    if val_range[0] > val_range[1]:
        raise ValueError('interval lower bound > upper bound')
    if min_val is not None and val_range[0] < min_val:
        raise ValueError('invalid interval lower bound')
    if max_val is not None and val_range[1] > max_val:
        raise ValueError('invalid interval upper bound')


def random_visual_effect_generator(
        contrast_range=(0.9, 1.1),
        brightness_range=(-.1, .1),
        hue_range=(0.9, 1.1),
        saturation_range=(0.95, 1.05)):
    _check_range(contrast_range, 0)
    _check_range(brightness_range, -1, 1)
    _check_range(hue_range, 0)
    _check_range(saturation_range, 0)

    def _generate():
        while True:
            yield VisualEffect(
                contrast_factor=_uniform(contrast_range),
                brightness_delta=_uniform(brightness_range),
                hue_range=hue_range,
                saturation_factor=_uniform(saturation_range)
            )

    return _generate()