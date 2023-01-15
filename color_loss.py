import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle_msssim import ssim


# 色彩恒常性损失
class L_color(nn.Layer):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, img1, img2):
        # b, c, h, w = img1.shape

        mean_rgb_1 = paddle.mean(img1, [2, 3], keepdim=True)
        mean_rgb_2 = paddle.mean(img2, [2, 3], keepdim=True)
        mr_1, mg_1, mb_1 = paddle.split(mean_rgb_1, 3, axis=1)
        mr_2, mg_2, mb_2 = paddle.split(mean_rgb_2, 3, axis=1)
        l2_loss_fn = nn.MSELoss(reduction='none')
        Dr = l2_loss_fn(mr_1, mr_2)
        Dg = l2_loss_fn(mg_1, mg_2)
        Db = l2_loss_fn(mb_1, mb_2)
        k = (Dr + Dg + Db) / 3

        return paddle.sqrt(k)  # 8*1*1*1


# 空间一致性损失
class L_spa(nn.Layer):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.kernel_left = paddle.unsqueeze(paddle.unsqueeze(paddle.to_tensor([[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], stop_gradient=True), 0), 0)
        self.kernel_right = paddle.unsqueeze(paddle.unsqueeze(paddle.to_tensor([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]], stop_gradient=True), 0), 0)
        self.kernel_up = paddle.unsqueeze(paddle.unsqueeze(paddle.to_tensor([[0.0, -1.0, 0.0], [.0, 1.0, 0.0], [0.0, 0.0, 0.0]], stop_gradient=True), 0), 0)
        self.kernel_down = paddle.unsqueeze(paddle.unsqueeze(paddle.to_tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], stop_gradient=True), 0), 0)
        # self.weight_left = nn.ParameterList(data=kernel_left, requires_grad=False)
        # self.weight_right = nn.ParameterList(data=kernel_right, requires_grad=False)
        # self.weight_up = nn.ParameterList(data=kernel_up, requires_grad=False)
        # self.weight_down = nn.ParameterList(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2D(4)

    def forward(self, org, enhance, num):
        b, c, h, w = org.shape

        # org_mean = torch.mean(org, 1, keepdim=True)  # batch_size*1*256*256
        # enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org)  # batch_size * 6 * 64 * 64
        enhance_pool = self.pool(enhance)

        # weight_diff = torch.max(
        #     torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
        #                                                       torch.FloatTensor([0]).cuda()),
        #     torch.FloatTensor([0.5]).cuda())
        # E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        E = None
        for i in range(3 * num):
            # 后面，包括E，都是batch_size*1*64*64
            org_pool_channel = org_pool[:, i, :, :].unsqueeze(1)
            enhance_pool_channel = enhance_pool[:, i, :, :].unsqueeze(1)

            D_org_letf = F.conv2d(org_pool_channel, self.kernel_left, padding=1)
            D_org_right = F.conv2d(org_pool_channel, self.kernel_right, padding=1)
            D_org_up = F.conv2d(org_pool_channel, self.kernel_up, padding=1)
            D_org_down = F.conv2d(org_pool_channel, self.kernel_down, padding=1)

            D_enhance_letf = F.conv2d(enhance_pool_channel, self.kernel_left, padding=1)
            D_enhance_right = F.conv2d(enhance_pool_channel, self.kernel_right, padding=1)
            D_enhance_up = F.conv2d(enhance_pool_channel, self.kernel_up, padding=1)
            D_enhance_down = F.conv2d(enhance_pool_channel, self.kernel_down, padding=1)

            D_left = paddle.pow(D_org_letf - D_enhance_letf, 2)
            D_right = paddle.pow(D_org_right - D_enhance_right, 2)
            D_up = paddle.pow(D_org_up - D_enhance_up, 2)
            D_down = paddle.pow(D_org_down - D_enhance_down, 2)
            E_i = (D_left + D_right + D_up + D_down)
            # E = 25*(D_left + D_right + D_up +D_down)
            if i == 0:
                E = E_i
            else:
                E = paddle.concat(x=[E, E_i], axis=1)

        return E


# 与原图像的损失
class L_cons(nn.Layer):
    def __init__(self):
        super(L_cons, self).__init__()

    def forward(self, ref_imgs, tgt_imgs):
        # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        l1_loss_fn = nn.L1Loss(reduction='mean')
        loss = l1_loss_fn(ref_imgs, tgt_imgs)
        return loss


# 两个变换矩阵需互逆
class L_inv(nn.Layer):
    def __init__(self):
        super(L_inv, self).__init__()

    def forward(self, predict_vectors_12, predict_vectors_21):
        predict_vectors_12 = paddle.unsqueeze(predict_vectors_12, 1)
        predict_vectors_12 = paddle.reshape(predict_vectors_12, (-1, 3, 4))
        predict_vectors_21 = paddle.unsqueeze(predict_vectors_21, 1)
        predict_vectors_21 = paddle.reshape(predict_vectors_21, (-1, 3, 4))

        batch = predict_vectors_12.shape[0]
        A_12 = paddle.zeros(shape=[batch, 4, 4], dtype='float32')
        A_12[:, 3, 3] = 1
        A_12[:, 0: 3, :] = predict_vectors_12

        A_21 = paddle.zeros(shape=[batch, 4, 4], dtype='float32')
        A_21[:, 3, 3] = 1
        A_21[:, 0: 3, :] = predict_vectors_21

        E = paddle.zeros(shape=[batch, 4, 4], dtype='float32')
        E[:, 0, 0] = 1
        E[:, 1, 1] = 1
        E[:, 2, 2] = 1
        E[:, 3, 3] = 1

        A = A_12 * A_21
        L_I = A - E

        loss = paddle.sqrt(paddle.trace(paddle.matmul(L_I, L_I, transpose_x=False, transpose_y=True), axis1=1, axis2=2)) / 9

        return loss


class L_SSIM(nn.Layer):
    def __init__(self):
        super(L_SSIM, self).__init__()

    def forward(self, img1, img2):
        return ssim(img1, img2, size_average=True)


def compute_output_imgs_3(predict_vectors, ref_imgs, num):
    predict_vectors = paddle.unsqueeze(predict_vectors, 1)
    predict_vectors = paddle.reshape(predict_vectors, (-1, 3 * num, 4))

    residual_imgs = None
    for i in range(num):
        r_vector = predict_vectors[:, i + 0, :].unsqueeze(-1).unsqueeze(-1)
        g_vector = predict_vectors[:, i + 1, :].unsqueeze(-1).unsqueeze(-1)
        b_vector = predict_vectors[:, i + 2, :].unsqueeze(-1).unsqueeze(-1)

        r = ref_imgs[:, i + 0, :, :]
        g = ref_imgs[:, i + 1, :, :]
        b = ref_imgs[:, i + 2, :, :]

        r_output = r * r_vector[:, 0] + g * r_vector[:, 1] + b * r_vector[:, 2] + r_vector[:, 3]
        g_output = r * g_vector[:, 0] + g * g_vector[:, 1] + b * g_vector[:, 2] + g_vector[:, 3]
        b_output = r * b_vector[:, 0] + g * b_vector[:, 1] + b * b_vector[:, 2] + b_vector[:, 3]

        # residual_img = torch.stack((r_output, g_output, b_output), 1)
        residual_img = paddle.concat(x=[paddle.unsqueeze(r_output, 1), paddle.unsqueeze(g_output, 1), paddle.unsqueeze(b_output, 1)], axis=1)
        if i == 0:
            residual_imgs = residual_img
        else:
            residual_imgs = paddle.concat(x=[residual_imgs, residual_img], axis=1)

    # return residual_imgs + ref_imgs[:, 0:3 * num, :, :]
    return residual_imgs