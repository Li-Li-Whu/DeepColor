import os
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.99999"
os.environ["FLAGS_eager_delete_tensor_gb"] = "0"
import paddle
import paddle.fluid as fluid
from paddle.io import DataLoader
import argparse
from model import AlexNet
from dataloader import ImageDataset
import color_loss


parser = argparse.ArgumentParser(description='PWCNet_paddle')
parser.add_argument('--data_root', default='./VOC2012/JPEGImages/', help='the path of selected datasets')
parser.add_argument('--model_out_dir', default='./out/', help='the path of selected datasets')
parser.add_argument('--loss', default='l2', help='loss type : first train with l2 and finetune with l1')
parser.add_argument('--train_val_txt', default='', help='the path of selected train_val_txt of dataset')
parser.add_argument('--numEpoch', '-e', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
parser.add_argument('--load_pretrain', type=bool, default=False, help='whether to load pretrain model')
parser.add_argument('--pretrain_dir', type=str, default="snapshots/epoch_4.pth", help='path to the pretrained model weights')
parser.add_argument('--optimize', default=None, help='path to the pretrained optimize weights')
parser.add_argument('--use_multi_gpu',action = 'store_true', help='Enable multi gpu mode')

args = parser.parse_args()


L_color = color_loss.L_color()
L_spa = color_loss.L_spa()
L_cons = color_loss.L_cons()
L_inv = color_loss.L_inv()
L_ssim = color_loss.L_SSIM()


def main():
    print(args)
    if args.use_multi_gpu:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    else:
        place = fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place=place):
        if args.use_multi_gpu:
            strategy = fluid.dygraph.parallel.prepare_context()
        model = AlexNet(num_classes=12)

        if args.load_pretrain:
            print('-----------load pretrained model:', args.pretrain_dir)
            # pd_pretrain, _ = fluid.dygraph.load_dygraph(args.pretrain_dir)
            model_dict = model.state_dict()
            param_state_dict = paddle.load(args.pretrain_dir)['model']
            # param_state_dict = match_state_dict(model_dict, param_state_dict)
            model.set_dict(param_state_dict)

        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.0001, parameter_list=model.parameters(), regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0004))
        if args.optimize:
            print('-----------load pretrained optimizer:', args.optimize)
            adam_pretrain, _ = fluid.dygraph.load_dygraph(args.optimize)
            optimizer.set_dict(adam_pretrain)
        if args.use_multi_gpu:
            model = fluid.dygraph.parallel.DataParallel(model, strategy)

        train_dataset = ImageDataset(root=args.data_root, mode='train')
        data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        epoch_num = args.numEpoch


        model.train()
        for epoch in range(0, epoch_num):
            for batch_id, data in enumerate(data_loader()):
                image1, image2 = data
                org_img_12 = paddle.concat(x=[image1, image2], axis=1)

                logits_12 = model(org_img_12)

                output_img_12 = color_loss.compute_output_imgs_3(logits_12, org_img_12[:, 0:3, :, :], 1)

                temp = org_img_12[:, 0: 3, :, :]
                img_batch_inv = org_img_12[:, 3: 6, :, :]
                org_img_21 = paddle.concat(x=[img_batch_inv, temp], axis=1)

                logits_21 = model(org_img_21)
                output_img_21 = color_loss.compute_output_imgs_3(logits_21, org_img_21[:, 0:3, :, :], 1)

                loss_L1_12 = L_cons(output_img_12, org_img_12[:, 3:6, :, :])
                loss_L1_21 = L_cons(output_img_21, org_img_21[:, 3:6, :, :])
                loss_spa_12 = paddle.mean(L_spa(org_img_12[:, 0:3, :, :], output_img_12, 1))
                loss_spa_21 = paddle.mean(L_spa(org_img_21[:, 0:3, :, :], output_img_21, 1))
                loss_col_12 = paddle.mean(L_color(output_img_12, org_img_12[:, 3:6, :, :]))
                loss_col_21 = paddle.mean(L_color(output_img_21, org_img_21[:, 3:6, :, :]))
                loss_ssim_12 = L_ssim(output_img_12, org_img_12[:, 0:3, :, :])
                loss_ssim_21 = L_ssim(output_img_21, org_img_21[:, 0:3, :, :])
                loss_inv = paddle.mean(L_inv(logits_12, logits_21))

                loss = (loss_col_12 + loss_col_21) * 1 + (loss_spa_12 + loss_spa_21) * 1 \
                       + ((1 - loss_ssim_12) + (1 - loss_ssim_21)) * 1 + loss_inv * 1

                loss.backward()

                optimizer.minimize(loss)
                model.clear_gradients()

                if ((batch_id + 1) % 10) == 0:
                    print("epoch ",epoch)
                    print("Loss at epoch ", epoch, ", iteration", batch_id + 1, ":", loss.item())
                    print("loss_L1: ", loss_L1_12.item())
                    print("loss_spa: ", loss_spa_12.item())
                    print("loss_col: ", loss_col_12.item())
                    print("loss_ssim: ", loss_ssim_12.item())
                    print("loss_inv: ", loss_inv.item())
                    # print("loss_min: ", loss_min.item())

            if (epoch % 1) == 0:
                obj = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': 100}
                path = args.model_out_dir + "epoch_" + str(epoch) + '.pth'
                paddle.save(obj, path)


if __name__ == '__main__':
    main()