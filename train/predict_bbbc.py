import argparse
import collections
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import UNET.torch_models as module_arch
import UNET.data_loaders as data_loaders
from UNET.utils import ConfigParser
from UNET.utils import prepare_device
from UNET.torch_models import UNetModel
from torchsummary import summary
from skimage.io import imread
torch.cuda.empty_cache()

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def plot(output):
    output = F.softmax(output,dim=1)
    idx = output.argmax(dim=1)
    mask = F.one_hot(idx,num_classes=3)
    mask = torch.permute(mask,(0,3,1,2))
    fig, ax = plt.subplots(1,2,sharex=True,sharey=True)
    ax[0].imshow(mask[0,0,:,:],cmap='gray')
    ax[1].imshow(output[0,0,:,:],cmap='gray')
    plt.show()

def main(config,path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.init_obj('arch', module_arch)
    model.to(device=device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    data_loader = config.init_obj('data_loader', data_loaders)
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data).cpu()
            plot(output)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    path = 'saved/models/UNetModel/1222_162246/model_best.pth'
    main(config,path)
