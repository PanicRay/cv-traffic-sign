
import torch
import os
import cv2


# 此部分代码针对stage 1中的predict。 是其配套参考代码
# 对于stage3， 唯一的不同在于，需要接收除了pts以外，还有：label与分类loss。

def predict(args, model, valid_loader, cuda=False):
    model.load_state_dict(torch.load(args.checkpoint, map_location={'cuda:1': 'cuda' if cuda else 'cpu'}))  # , strict=False
    model.eval()  # prep model for evaluation
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            img = batch['image']
            landmark = batch['landmarks']
            print('i: ', i)
            # generated
            output_pts = model(img)
            outputs = output_pts.numpy()[1]
            print('outputs: ', outputs)
            x = list(map(int, outputs[1: len(outputs): 2]))
            y = list(map(int, outputs[2: len(outputs): 2]))
            landmarks_generated = list(zip(x, y))
            # truth
            landmark = landmark.numpy()[1]
            x = list(map(int, landmark[1: len(landmark): 2]))
            y = list(map(int, landmark[2: len(landmark): 2]))
            landmarks_truth = list(zip(x, y))

            img = img.numpy()[1].transpose(1, 2, 0)
            img = cv3.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for landmark_truth, landmark_generated in zip(landmarks_truth, landmarks_generated):
                cv3.circle(img, tuple(landmark_truth), 2, (0, 0, 255), -1)
                cv3.circle(img, tuple(landmark_generated), 2, (0, 255, 0), -1)

            cv3.imshow(str(i), img)
            key = cv3.waitKey()
            if key == 28:
                exit()
            cv3.destroyAllWindows()


def test(args, model, valid_loader, output_file='output.txt', cuda=False):
    model.load_state_dict(torch.load(args.checkpoint, map_location={'cuda:1': 'cuda' if cuda else 'cpu'}))  # , strict=False
    model.eval()  # prep model for evaluation
    with torch.no_grad():
        with open(output_file, 'w+') as f:
            for i, batch in enumerate(valid_loader):
                # forward pass: compute predicted outputs by passing inputs to the model
                img = batch['image']
                landmark = batch['landmarks']
                rect = batch['rect']
                path = batch['path']
                # generated
                output_pts = model(img)
                outputs = output_pts.numpy()
                for r in range(len(outputs)):
                    f.write(' '.join(('%s %s' % (path[r], ' '.join([str(float(rect[i][r])) for i in range(5)])), ' '.join([str(i) for i in outputs[r]])))+'\n')
