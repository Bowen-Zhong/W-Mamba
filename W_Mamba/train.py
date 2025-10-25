import csv
import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import warnings
import re

from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from Metric.eval_one_image import evaluation_one
# from loss import grad_loss, ints_loss

from dataloader_HMIF import TrainData, TestData
from args_setting import args
from natsort import natsorted
import glob

# from net_DFTv2P_fusion_strategy import net_pyramid as net
from Double_SS2D import  W_Mamba

from utils.util_train import tensorboard_load

# from FourierBranch_ab import net_pyramid as net
from loss import compute_loss

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def write_results_to_csv(results, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        # 创建一个写入器对象
        writer = csv.DictWriter(csvfile,
                                fieldnames=['Image', 'EN','SF', 'AG', 'SD', 'CC', 'SCD', 'VIF', 'MSE', 'PSNR', 'MI', 'SSIM', 'MS_SSIM', 'Qabf'])
        # 写入标题头
        writer.writeheader()
        # 写入数据
        for result in results:
            writer.writerow(result)
def Mytrain(model_pretrain=None):
    # 设置随机数种子
    logs_path = './modelsave/' + args.model + '/' + args.task
    writer = SummaryWriter(logs_path)
    print('Tensorboard 构建完成，进入路径：' + logs_path)
    print('然后使用该指令查看训练过程：tensorboard --logdir=./')
    setup_seed()
    model_path = './modelsave/' + args.model + '/' + args.task + '/'
    os.makedirs(model_path, exist_ok=True)
    # os.makedirs('./modelsave')

    lr = args.lr

    # device handling
    if args.DEVICE == 'cpu':
        device = 'cpu'
    else:
        device = args.DEVICE
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # prepare model folder
    temp_dir = './temp/' + args.model + '/' + args.task
    os.makedirs(temp_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    train_set = TrainData(transform=transform)
    train_loader = data.DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=0,
                                   pin_memory=True)
    # model = net(img_size=args.imgsize, dim=256)
    # print('train datasets lenth:', len(train_loader))
    model = W_Mamba()


    if model_pretrain is not None:
        model.load_state_dict(torch.load(model_pretrain, map_location=args.DEVICE))

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    cont_training = False
    epoch_start = 0
    if cont_training:
        epoch_start = 930
        model_dir = './modelsave/' + args.model + '/' + args.task + '/'
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        model.load_state_dict(best_model)

    loss_plt = []
    for epoch in range(epoch_start, args.epoch):
        loss_mean = []
        for idx, datas in enumerate(tqdm(train_loader, desc='[Epoch--%d]' % (epoch + 1))):
            # for idx, datas in tqdm(train_loader):
            model.train()
            # print(len(data))
            img1, img2 = datas
            # 训练模型
            model, img_fusion, loss_per_img = train(model, img1, img2, lr, device)
            loss_mean.append(loss_per_img)

        # print loss
        sum_list = 0
        for item in loss_mean:
            sum_list += item
        sum_per_epoch = sum_list / len(loss_mean)
        print('\tLoss:%.5f' % sum_per_epoch)
        loss_plt.append(sum_per_epoch.detach().cpu().numpy())

        # save info to txt file
        strain_path = temp_dir + '/temp_loss.txt'
        Loss_file = 'Epoch--' + str(epoch + 1) + '\t' + 'Loss:' + str(sum_per_epoch.detach().cpu().numpy())
        with open(strain_path, 'a') as f:
            f.write(Loss_file + '\r\n')

        max_model_num = 2000
        # save model 测试
        if (epoch + 1) % 20 == 0 or epoch + 1 == args.epoch:
            torch.save(model.state_dict(), model_path + str(epoch + 1) + '_' + 'ASFEFusion.pth')
            print('model save in %s' % './modelsave/' + args.model + '/' +  args.task)

            model_lists = natsorted(glob.glob('./modelsave/' + args.model + '/' +  args.task + '/*'))
            while len(model_lists) > max_model_num:
                os.remove(model_lists[0])
                model_lists = natsorted(glob.glob('./modelsave/' + args.model + '/' +  args.task + '/*'))


            img_save_dir = './result/' + args.model + '/' + args.task
            os.makedirs(img_save_dir, exist_ok=True)
            model.eval()
            model = model.to(device)
        # net.load_state_dict(torch.load(model_path_final, map_location=args.DEVICE))

            transform = transforms.Compose([transforms.ToTensor()])
            test_set = TestData(transform)
            test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False,
                                      num_workers=1, pin_memory=False)
            with torch.no_grad():
                if args.task == 'CT-MRI':
                    for batch, [img_name, img1, img2] in enumerate(test_loader):  # CT-MRI Fusion
                        print("test for image %s" % img_name[0])
                        img1 = img1.to(device)
                        img2 = img2.to(device)
                        fused_img = model(img1, img2)
                        fused_img = (fused_img - fused_img.min()) / (fused_img.max() - fused_img.min()) * 255.
                        fused_img = fused_img.cpu().numpy().squeeze()
                        cv2.imwrite('%s/%s' % (img_save_dir, img_name[0]), fused_img)
                else:
                    for batch, [img_name, img1_Y, img2, img1_CrCb] in enumerate(test_loader):  # PET/SPECT-MRI-MRI Fusion
                        print("test for image %s" % img_name[0])

                        img1_Y = img1_Y.to(device)
                        img2 = img2.to(device)

                        fused_img_Y = model(img1_Y, img2)

                        fused_img_Y = (fused_img_Y - fused_img_Y.min()) / (fused_img_Y.max() - fused_img_Y.min()) * 255.
                        fused_img_Y = fused_img_Y.cpu().numpy()

                        fused_img = np.concatenate((fused_img_Y, img1_CrCb), axis=1).squeeze()
                        fused_img = np.transpose(fused_img, (1, 2, 0))
                        fused_img = fused_img.astype(np.uint8)
                        fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)

                        cv2.imwrite('%s/%s' % (img_save_dir, img_name[0]), fused_img)
                Fuse_path = img_save_dir
                dir_prefix = './HMIFDatasets/' + args.task + '/test/'
                IR_path = dir_prefix + args.task.split('-')[0]
                VIS_path = dir_prefix + 'MRI'
                print(Fuse_path, IR_path, VIS_path)
                result_path = './assessment_indexes/' + args.model + '/' + args.task + str(epoch)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                IR_image_list = os.listdir(IR_path)
                VIS_image_list = os.listdir(VIS_path)
                Fuse_image_list = os.listdir(Fuse_path)
              
            # EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(ir_name, vi_name, f_name)
                evaluation_results = []
                num = 0

                for IR_image_name, VIS_image_name, Fuse_image_name in zip(IR_image_list, VIS_image_list, Fuse_image_list):
                    num += 1
                    IR_image_path = os.path.join(IR_path, IR_image_name)
                    print(IR_image_path)
                    VIS_image_path = os.path.join(VIS_path, VIS_image_name)
                    print(VIS_image_path)
                    Fuse_image_path = os.path.join(Fuse_path, Fuse_image_name)
                    print(Fuse_image_path)

                    EN, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, MI, SSIM, MS_SSIM, Qabf = evaluation_one(IR_image_path, VIS_image_path,
                                                                                   Fuse_image_path)
                    # 将指标添加到列表中
                    evaluation_results.append({
                        'Image': num,
                        'EN': round(EN, 4),
                        'SF': round(SF, 4),
                        'AG': round(AG, 4),
                        'SD': round(SD, 4),
                        'CC': round(CC, 4),
                        'SCD': round(SCD, 4),
                        'VIF': round(VIF, 4),
                        'MSE': round(MSE, 4),
                        'PSNR': round(PSNR, 4),
                        'MI': round(MI, 4),
                        'SSIM': round(SSIM, 4),
                        'MS_SSIM': round(MS_SSIM, 4),
                        'Qabf': round(Qabf, 4),
                    })
                # 计算所有图像的平均评估指标
                average_results = {
                    'Image': 'Average',
                    'EN': round(np.mean([results['EN'] for results in evaluation_results]), 4),
                    'SF': round(np.mean([results['SF'] for results in evaluation_results]), 4),
                    'AG': round(np.mean([results['AG'] for results in evaluation_results]), 4),
                    'SD': round(np.mean([results['SD'] for results in evaluation_results]), 4),
                    'CC': round(np.mean([results['CC'] for results in evaluation_results]), 4),
                    'SCD': round(np.mean([results['SCD'] for results in evaluation_results]), 4),
                    'VIF': round(np.mean([results['VIF'] for results in evaluation_results]), 4),
                    'MSE': round(np.mean([results['MSE'] for results in evaluation_results]), 4),
                    'PSNR': round(np.mean([results['PSNR'] for results in evaluation_results]), 4),
                    'MI': round(np.mean([results['MI'] for results in evaluation_results]), 4),
                    'SSIM': round(np.mean([results['SSIM'] for results in evaluation_results]), 4),
                    'MS_SSIM': round(np.mean([results['MS_SSIM'] for results in evaluation_results]), 4),
                    'Qabf': round(np.mean([results['Qabf'] for results in evaluation_results]), 4),
                }

            # 将平均结果添加到列表中
                evaluation_results.append(average_results)
                writer.add_scalar('EN', average_results["EN"].item(), global_step=epoch)
                writer.add_scalar('SF', average_results["SF"].item(), global_step=epoch)
                writer.add_scalar('AG', average_results["AG"].item(), global_step=epoch)
                writer.add_scalar('SD', average_results["SD"].item(), global_step=epoch)
                writer.add_scalar('CC', average_results["CC"].item(), global_step=epoch)
                writer.add_scalar('SCD', average_results["SCD"].item(), global_step=epoch)
                writer.add_scalar('VIF', average_results["VIF"].item(), global_step=epoch)
                writer.add_scalar('MSE', average_results["MSE"].item(), global_step=epoch)
                writer.add_scalar('PSNR', average_results["PSNR"].item(), global_step=epoch)
                writer.add_scalar('MI', average_results["MI"].item(), global_step=epoch)
                writer.add_scalar('SSIM', average_results["SSIM"].item(), global_step=epoch)
                writer.add_scalar('MS_SSIM', average_results["MS_SSIM"].item(), global_step=epoch)
                writer.add_scalar('Qabf', average_results["Qabf"].item(), global_step=epoch)
                #tensorboard_load(writer, average_results, args.epoch)
            # 将结果保存到CSV文件
                csv_file_path = os.path.join(result_path, 'evaluation_results.csv')
                write_results_to_csv(evaluation_results, csv_file_path)

                print(f"评估结果已保存到CSV文件：{csv_file_path}")

    writer.close()
    # 输出损失函数曲线
    plt.figure()
    x = range(0, args.epoch)  # x和y的维度要一样
    y = loss_plt
    plt.plot(x, y, 'r-')  # 设置输出样式
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(model_path + '/loss.png')  # 保存训练损失曲线图片
    plt.show()  # 显示曲线


def train(model, img1, img2, lr, device):
    model.to(device)
    model.train()

    img1 = img1.to(device)
    img2 = img2.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr)

    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    img_fusion = model(img1, img2)
    # img_fusion = img_fusion.cpu()
    img_fusion = img_fusion.to(device)

    img_cat = torch.cat([img1, img2], dim=1)

    #loss_total = compute_loss(img_fusion, img_cat, img1, img2)
    #img1是CT，img2是MRI
    loss_total = compute_loss(img_fusion, img_cat, img1, img2)
    # loss_total = torch.from_numpy(loss_total.numpy())

    opt.zero_grad()
    loss_total.backward()
    opt.step()

    return model, img_fusion, loss_total




if __name__ == '__main__':
    Mytrain(model_pretrain=None)
