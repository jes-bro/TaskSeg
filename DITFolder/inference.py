from utils.distributed import *
import torch.multiprocessing as mp
from utils.ckpt import *
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.logging import *
import argparse
import time
from pathlib import Path
from utils import config
import ffmpeg
from natsort import natsorted
import math
from datasets.dataloader import loader,RefCOCODataSet
from datasets.inf_dataloader_human import InferenceDataSet
from tensorboardX import SummaryWriter
from utils.utils import *
import torch.optim as Optim
from importlib import import_module
from utils.misc import NestedTensor
from pytorch_pretrained_bert.tokenization import BertTokenizer

class ModelLoader:
    def __init__(self, __C):

        self.model_use = __C.MODEL
        model_moudle_path = 'models.' + self.model_use + '.net'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, __arg1, __arg2, __arg3):
        return self.model_moudle.Net(__arg1, __arg2, __arg3)
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([0, 1, 1, 0.6])  # Cyan
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def denormalize(image, mean, std):
    """
    Denormalizes a tensor that was normalized with the given mean and std.

    Args:
        tensor (torch.Tensor): Normalized tensor.
        mean (list of float): Mean values used for normalization.
        std (list of float): Standard deviation values used for normalization.

    Returns:
        torch.Tensor: Denormalized tensor.
    """
    channels = image.shape[0]
    for c in range(channels):
        image[c] = image[c] * std[c] + mean[c]
    
    return image

def perform_inference(__C,
             net,
             loader,
             ema=None):
    # breakpoint()
    if ema is not None:
        ema.apply_shadow()
    net.eval()
    with th.no_grad():
        for ith_batch, data in enumerate(loader):
            # breakpoint()
            ref_iter, image_iter, mask_id, ref_mask_iter = data
            ref_iter = ref_iter.cuda(non_blocking=True)
            image_iter = image_iter.cuda(non_blocking=True)
            ref_mask_iter = ref_mask_iter.cuda(non_blocking=True)
            # lang_iter = NestedTensor(ref_iter,ref_mask_iter)
            lang_iter = NestedTensor(ref_iter.unsqueeze(0),ref_mask_iter.unsqueeze(0))
            # breakpoint()
            mask= net(image_iter,lang_iter)
            
            mask=mask.cpu().numpy()
            # breakpoint()
            mask_dir = "/home/jess/TaskSeg/final_masks-5/segs/pred_segs-USBINFHH"
            os.makedirs(mask_dir, exist_ok=True)
            for i, mask_pred in enumerate(mask):
                plt.figure(figsize=(10,10))
                if isinstance(image_iter[i].cpu(), torch.Tensor):
                    img = denormalize(image_iter[i], __C.MEAN, __C.STD)
                    # breakpoint()
                    img = image_iter[i].permute(1, 2, 0).cpu().numpy()  # Convert and transpose if it's a tensor
                else:
                    img = denormalize(image_iter[i]) # Already a NumPy array, just use it directly
                plt.figure()
                plt.imshow(img)
                plt.title(f"Image {i+1}", fontsize=18)
                plt.axis('off')
                plt.savefig(os.path.join(mask_dir, f'image_{mask_id}.png'))
                plt.close()

                # Plot and save the mask
                plt.figure()
                plt.imshow(img)
                show_mask(mask_pred, plt.gca()) # Assuming show_mask is a function that overlays the mask
                plt.title(f"Mask {i+1}", fontsize=18)
                plt.axis('off')
                plt.savefig(os.path.join(mask_dir, f'mask_{mask_id}.png'))
                plt.close()

def main_worker(gpu,__C,images,texts):
    if __C.MULTIPROCESSING_DISTRIBUTED:
        if __C.DIST_URL == "env://" and __C.RANK == -1:
            __C.RANK = int(os.environ["RANK"])
        if __C.MULTIPROCESSING_DISTRIBUTED:
            __C.RANK = __C.RANK* len(__C.GPU) + gpu
        dist.init_process_group(backend=dist.Backend('NCCL'), init_method=__C.DIST_URL, world_size=__C.WORLD_SIZE, rank=__C.RANK)
    data = []
    for image, text in zip(images, texts):
        new_datapoint = {"image":image,"text":text}
        data.append(new_datapoint)
    dataset=InferenceDataSet(__C,data,split='train')
    # breakpoint()
    net= ModelLoader(__C).Net(
        __C,
        dataset.pretrained_emb,
        dataset.token_size
    )

    eval_str = 'params, lr=%f'%__C.LR
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])

    if __C.MULTIPROCESSING_DISTRIBUTED:
        torch.cuda.set_device(gpu)
        net = DDP(net.cuda(), device_ids=[gpu],find_unused_parameters=True)
    elif len(gpu)==1:
        net.cuda()
    else:
        net = DP(net.cuda())
 #       net = DP(net)
    if main_process(__C, gpu):
        print(__C)
        # print(net)
        total = sum([param.nelement() for param in net.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6))  # 每一百万为一个单位


    if os.path.isfile(__C.RESUME_PATH):
        checkpoint = torch.load(__C.RESUME_PATH,map_location=lambda storage, loc: storage.cuda() )
        net.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        if main_process(__C,gpu):
            print("==> loaded checkpoint from {}\n".format(__C.RESUME_PATH) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))
    # breakpoint()
    perform_inference(__C,net,dataset)

def load_rgbs_from_directory(category_name):
    rgb_save_dir = Path(category_name)

    rgb_images = []

    for rgb_file in natsorted(rgb_save_dir.glob('*.jpg')):
        if rgb_file.exists():
            rgb_image = plt.imread(rgb_file)
            rgb_images.append(rgb_image)
        else:
            print(f"Warning: Missing data for {rgb_file.stem}")

    return rgb_images

def get_segmentation(image,text):
    __C = config.load_cfg_from_cfg_file("/home/jess/TaskSeg/DIT/config/e2e.yaml")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in __C.GPU)
    setup_unique_version(__C)
    seed_everything(__C.SEED)
    N_GPU=len(__C.GPU)
    __C.RESUME_PATH=__C.CHECKPOINT_PATH
    if not os.path.exists(os.path.join(__C.LOG_PATH,str(__C.VERSION))):
        os.makedirs(os.path.join(__C.LOG_PATH,str(__C.VERSION),'ckpt'),exist_ok=True)
    if N_GPU == 1:
        __C.MULTIPROCESSING_DISTRIBUTED = False
    else:
        # turn on single or multi node multi gpus training
        __C.MULTIPROCESSING_DISTRIBUTED = True
        __C.WORLD_SIZE *= N_GPU
        __C.DIST_URL = f"tcp://127.0.0.1:{find_free_port()}"
    if __C.MULTIPROCESSING_DISTRIBUTED:
        mp.spawn(main_worker, args=(__C,), nprocs=N_GPU, join=True)
    else:
        main_worker(__C.GPU, __C, image, text)



def convert_vid_to_rgbs(vid_path: str):
    # Use ffmpeg to decode the video
    process = (
        ffmpeg
        .input(vid_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    # Get video metadata (e.g., width, height)
    probe = ffmpeg.probe(vid_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    # Read raw video frames
    frame_size = width * height * 3
    frames = []
    while True:
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        frames.append(frame)

    return frames

text = "The bag of loose tea leaves."
# images = convert_vid_to_rgbs("/home/jess/TaskSeg/DIT/non-pov-13.mp4")
images = load_rgbs_from_directory("/run/user/1008/doc/416d53ea/TaskSeg/tsg/jpgs/USBINF-same-front-90-132-2024-08-07 23:48:06.820491")
breakpoint()
texts = [text for _ in images]
get_segmentation(images, texts)