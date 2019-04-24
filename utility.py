import json
import torch
import os, sys
import time
import numpy as np
import imageio
import torch
import scipy
import scipy.misc
from shutil import copyfile
from PIL import Image, ImageDraw


def loadParams(jsonFile):
    with open(jsonFile) as f:
        return json.load(f)


def saveArgs(chkDir, args):
    if not os.path.exists(chkDir):
        os.makedirs(chkDir)
    os.makedirs(chkDir + '/models')
    os.makedirs(chkDir + '/dataset')

    fileList = args['projectFiles']
    for f in fileList:
        copyfile(f, chkDir + '/' + f)


class Record(object):
    def __init__(self):
        self.loss = 0
        self.count = 0

        self.miniloss = 0
        self.minicount = 0

    def add(self, value):
        self.loss += value
        self.count += 1
        self.minicount += 1
        self.miniloss += value

    def mean(self):
        return self.loss / self.count

    def minimean(self):
        ans = self.miniloss / self.minicount
        self.minicount = 0
        self.miniloss = 0
        return ans


class TestResult():
    def __init__(self):
        self.loss = torch.zeros(15)
        self.count = 0

    def add(self, n, value):
        self.loss[n] += value

    def mean(self):
        return self.loss / self.count


class Timer(object):
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.begin = time.time()

    def stop(self):
        self.end = time.time()
        self.interval = self.end - self.begin


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __def__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


# class AverageMeter(object):
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")


def printInfo(recorder, epoch=None, preffix='', suffix=''):
    s = preffix
    if epoch is not None:
        s = s + 'Epoch: {epoch}     '.format(epoch=epoch)
    if 'Time' in recorder:
        s = s + 'Time: {time.sum:.3f}   '.format(time=recorder['Timr'])
    if 'MSE_Loss' in recorder:
        s = s + 'MSE Loss: {loss.avg:.5f}   '.format(loss=recorder['MSE_Loss'])

    s = s + suffix
    print(s)


def saveCheckpoint(chkDir, stats, mode='newest'):
    if not os.path.exists(chkDir):
        os.makedirs(chkDir)
    chkFile = chkDir + '/' + mode + '_checkpoint.tar'
    torch.save(stats, chkFile)


def saveResult(chkDir, resultDict, mode='newest', num=None):
    if not os.path.exists(chkDir):
        os.makedirs(chkDir)
        resultFile = chkDir + '/' + mode + '_result.h5'

        with h5py.File(resultFile, 'w') as hdf:
            for key, values in resultDict.items():
                n = len(values) if num is None else num
                for i in range(n):
                    hdf.create_dataset(key + '/' + str(i), data=values[i])


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images) - 1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding:
            (i + 1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.Tensor) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images) - 1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding:
            (i + 1) * y_dim + i * padding].copy_(image)
        return result


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
             hasattr(arg, "__iter__")))


def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1)
        images.append((img.numpy() * 255).astype(np.uint8))
    imageio.mimsave(filename, images, duration=duration)


def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1).numpy() * 255
        images.append(img.astype(np.uint8))
    imageio.mimsave(filename, images, duration=duration)


def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    return scipy.misc.toimage((tensor.numpy() * 255).astype(np.uint8),
                              high=255 * tensor.numpy().max(),
                              channel_axis=0)


def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)


def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x * 255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0, 0, 0))
    img = np.asarray(pil)
    return torch.Tensor(img / 255.).transpose(1, 2).transpose(0, 1)
