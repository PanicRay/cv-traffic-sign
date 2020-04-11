from __future__ import print_function
import argparse
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler


from data import get_train_test_set
from myNN import resnet18, resnet34, resnet50, resnet101, resnet152, GoogLeNet
from myFPN import FPN101

import os


torch.set_default_tensor_type(torch.FloatTensor)


def train(args, train_loader, valid_loader, model, criterion, optimizer, device, cuda=False):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    if args.checkpoint != '':
        model.load_state_dict(torch.load(args.checkpoint, map_location={'cuda:0': 'cuda' if cuda else 'cpu'}))
        print('Training from checkpoint: %s' % args.checkpoint)

    epoch = args.epochs

    for epoch_id in range(epoch):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        ######################
        # training the model #
        ######################
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            target = batch['cate']

            # ground truth
            input_img = img.to(device)
            target_label = target.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get output
            output_label = model(input_img)

            # get loss
            loss = criterion(output_label, target_label)

            # do BP automatically
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                        epoch_id,
                        batch_idx * len(img),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item()
                    )
                )
                with open(os.path.join(args.save_directory, 'train_losses.txt'), 'a+') as f:
                    f.write('Train Epoch: {} pts_loss{:.10f}\n'.format(epoch_id, loss.item()))

        ######################
        # validate the model #
        ######################
        valid_mean_pts_loss = 0.0

        model.eval()  # prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0

            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                target = batch['cate']

                input_img = valid_img.to(device)
                target_label = target.to(device)

                output_label = model(input_img)

                # get loss
                valid_loss = criterion(output_label, target_label)

                valid_mean_pts_loss += valid_loss.item()

            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            print('Valid: pts_loss: {:.6f}'.format(
                    valid_mean_pts_loss
                )
            )
            with open(os.path.join(args.save_directory, 'valid_losses.txt'), 'a+') as f:
                f.write('Eval Epoch: pts_loss{:.10f}\n'.format(valid_mean_pts_loss))
        print('====================================================')
        # save model
        if args.save_model and epoch_id % args.save_interval == 0:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)

def main_test():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',				
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',		
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alg', type=str, default='adam',
                        help='select optimzer SGD, adam, or other')
    parser.add_argument('--loss', type=str, default='CE',
                        help='loss function')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    parser.add_argument('--net', type=str, default='resnet101',
                        help='DefaultNet, ResNet***[18,34,50,101,152], MobileNet or GoogLeNet')
    parser.add_argument('--angle', type=float, default=30,
                        help='max (30) angle range to rotate original image on both side')
    parser.add_argument('--save-interval', type=int, default=20,
                        help='after # of epoch, save the current Model')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='continuing the training from specified checkpoint')
    args = parser.parse_args()
    ###################################################################################
    torch.manual_seed(args.seed)
    # For single GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    # cuda:0
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('===> Loading Datasets')
    train_set, test_set = get_train_test_set(args.net, args.angle)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    # 输出类别定义
    categories = 62
    print('===> Building Model')
    # For single GPU
    if args.net == 'ResNet18' or args.net == 'resnet18':
        model = resnet18()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)
    elif args.net == 'ResNet34' or args.net == 'resnet34':
        model = resnet34()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)
    elif args.net == 'ResNet50' or args.net == 'resnet50':
        model = resnet50()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)
    elif args.net == 'ResNet101' or args.net == 'resnet101':
        model = resnet101()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)
    elif args.net == 'ResNet152' or args.net == 'resnet152':
        model = resnet152()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)
    elif args.net == 'GoogLeNet' or args.net == 'googlenet':
        model = GoogLeNet(num_classes=categories).to(device)
    else:
        model = resnet50()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, categories)
        model = model.to(device)
    '''elif args.net == 'FPN':
        model = FPN101(args.batch_size).to(device)
    else:
        model = Net(args.use_bn).to(device)'''
    ####################################################################
    if args.loss == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    ####################################################################
    # Freeze all layer except ip3, if current mode is finetune
    if args.phase == 'Finetune':
        start, end = (args.layer_lockdown.split(':'))
        for param in list(model.parameters())[start:end]:
            param.requires_grad = False
    if args.alg == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.alg == 'adam' or args.alg == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ####################################################################
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train(args, train_loader, valid_loader, model, criterion, optimizer, device, cuda=use_cuda)
        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        # how to do test?
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        # how to do finetune?
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        # how to do predict?
    else:
        print('===> Verify')



if __name__ == '__main__':
    main_test()


