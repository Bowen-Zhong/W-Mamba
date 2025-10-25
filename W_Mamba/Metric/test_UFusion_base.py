import os
import numpy as np
import torch
import re
import cv2
from time import time
from dataloader import TestData
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
from args_setting import args
from natsort import natsorted
from eval_one_image import evaluation_one, write_results_to_csv

from network import UFusion_base

DEVICE = args.DEVICE
EPS = 1e-8

def Mytest(model_test=None, img_save_dir=None):
    os.makedirs('./result/' + args.model + '/' + args.task, exist_ok=True)
    if model_test is None:
        model_dir = './modelsave/' + args.model + '/' + args.task + '/'
        #model_path_final = './modelsave/' + args.model + '/' + args.task + '/' + natsorted(os.listdir(model_dir))[-1]
        model_path_final = './modelsave/' + args.model + '/' + args.task + '/' + '{}_{}.pth'.format(args.epoch, args.model)
    else:
        model_path_final = model_test

    if img_save_dir is None:
        img_save_dir = './result/' + args.model + '/' + args.task
    else:
        img_save_dir = img_save_dir

    os.makedirs(img_save_dir, exist_ok=True)

    net = UFusion_base()
    net.eval()
    net = net.to(DEVICE)
    net.load_state_dict(torch.load(model_path_final, map_location=args.DEVICE))

    transform = transforms.Compose([transforms.ToTensor()])
    test_set = TestData(transform)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False,
                                  num_workers=1, pin_memory=False)
    with torch.no_grad():
        if args.task == 'CT-MRI':
            for batch, [img_name, img1, img2] in enumerate(test_loader):  # CT-MRI Fusion
                print("test for image %s" % img_name[0])
                img1 = img1.to(DEVICE)
                img2 = img2.to(DEVICE)
                fused_img = net(img1, img2)
                fused_img = (fused_img - fused_img.min()) / (fused_img.max() - fused_img.min()) * 255.
                fused_img = fused_img.cpu().numpy().squeeze()
                cv2.imwrite('%s/%s' % (img_save_dir, img_name[0]), fused_img)
        else:
            for batch, [img_name, img1_Y, img2, img1_CrCb] in enumerate(test_loader):  # PET/SPECT-MRI Fusion
                print("test for image %s" % img_name[0])

                img1_Y = img1_Y.to(DEVICE)
                img2 = img2.to(DEVICE)

                fused_img_Y = net(img1_Y, img2)

                fused_img_Y = (fused_img_Y - fused_img_Y.min()) / (fused_img_Y.max() - fused_img_Y.min()) * 255.
                fused_img_Y = fused_img_Y.cpu().numpy()

                fused_img = np.concatenate((fused_img_Y, img1_CrCb), axis=1).squeeze()
                fused_img = np.transpose(fused_img, (1, 2, 0))
                fused_img = fused_img.astype(np.uint8)
                fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)

                cv2.imwrite('%s/%s' % (img_save_dir, img_name[0]), fused_img)

    print('test results in ./%s/' % img_save_dir)
    print('Finish!')

    model_name = args.model
    IR_path = "./datasets/PET-MRI/test/ori_PET"
    VIS_path = "./datasets/PET-MRI/test/MRI"
    Fuse_path = "./result/{}/PET-MRI".format(model_name)
    result_path = './result/{}/scores/'.format(model_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    IR_image_list = os.listdir(IR_path)
    VIS_image_list = os.listdir(VIS_path)
    Fuse_image_list = os.listdir(Fuse_path)
    print(IR_image_list)
    print(VIS_image_list)
    print(Fuse_image_list)
    #IR_image_list = sorted(IR_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
    #VIS_image_list = sorted(VIS_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
    #Fuse_image_list = sorted(Fuse_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
    evaluation_results = []
    num = 0
    for IR_image_name, VIS_image_name,Fuse_image_name in zip(IR_image_list, VIS_image_list,Fuse_image_list):
        num += 1
        IR_image_path = os.path.join(IR_path, IR_image_name)
        print(IR_image_path)
        VIS_image_path = os.path.join(VIS_path, VIS_image_name)
        print(VIS_image_path)
        Fuse_image_path = os.path.join(Fuse_path,Fuse_image_name)
        print(Fuse_image_path)

        EN, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, MI, SSIM, MS_SSIM, Qabf = evaluation_one(IR_image_path, VIS_image_path, Fuse_image_path)
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
            'MI':round(MI,4),
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

    # 将结果保存到CSV文件
    csv_file_path = os.path.join(result_path, 'evaluation_results_{}_{}.csv'.format(model_name, args.epoch))
    write_results_to_csv(evaluation_results, csv_file_path)

    print(f"评估结果已保存到CSV文件：{csv_file_path}")

if __name__ == '__main__':
    '''Mytest(
        model_test="./modelsave/CT-MRI/2000_UNet.pth",
        img_save_dir='result/CT-MRI')'''  # 修改img_type
    Mytest()
