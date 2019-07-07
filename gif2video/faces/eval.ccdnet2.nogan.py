from utils import *

parser = argparse.ArgumentParser(description='PyTorch Training')
# data
parser.add_argument('--inputRoot', default='/nfs/bigbrain/yangwang/Gif2Video/data,FF/face_gif_image/expand1.5_size256_s1_g32_nodither/', type=str, help='root of gifs')
parser.add_argument('--tCrop', default=5, type=int, help='sequence length')
parser.add_argument('--sScale', default=1, type=int, help='spatial upsampling for input')
parser.add_argument('--sCrop', default=256, type=int, help='spatial patch size')
parser.add_argument('--tStride', default=10, type=int, help="temporal downsampling")
parser.add_argument('--workers', default=2, type=int, help='number of workers for dataloader')
parser.add_argument('--BtSz', default=8, type=int, help='batch size')
# model
parser.add_argument('--color_model_file', default='/nfs/bigfovea/yangwang/Gif2Video/degif_c/exp,FF/CMD_nogan/results/g32_nodither_pt256_bt8_tr0.1/idl100_1,gdl100_1,nogan/ckpt/ep-0060.pt', type=str, help='')
parser.add_argument('--color_model2_file', default='/nfs/bigfovea/yangwang/Gif2Video/degif_c/exp,FF,recurrent/CMD_nogan/results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,nogan/unroll1/ckpt/ep-0050.pt', type=str, help='')
parser.add_argument('--color_model_key', default='model_netG', type=str, help='')
parser.add_argument('--maxFlow', default=30, type=float, help='maximum flow value, use for rescaling')
# init & checkpoint
parser.add_argument('--initModel', default='', help='init model in absence of checkpoint')
parser.add_argument('--initNonAdv', action='store_true', default=False, help='init from a non-adv checkpoint')
parser.add_argument('--checkpoint', default=0, type=int, help='resume from a checkpoint')
# other mode
parser.add_argument('--evalMode', action='store_true', default=False, dest='evalMode', help='evaluation mode')
parser.add_argument('--visMode', action='store_true', default=False, dest='visMode', help='visualization mode')
parser.add_argument('--visDir', default='visual', type=str, help="dir to store visualization")
parser.add_argument('--visNum', default=10, type=int, help="number of videos to visualize")
# misc
parser.add_argument('--seed', default=1, type=int, help='random seed for torch/numpy')
opts = parser.parse_args()
manual_seed(opts.seed)

global color_model
global color_model2
color_model = models.UNet_rgb(3, 3, ch=64)
color_model.load_state_dict(torch.load(opts.color_model_file)[opts.color_model_key])
color_model.eval()
color_model = nn.DataParallel(color_model.to(DEVICE))
color_model2 = models.UNet_rgb(3*4, 3, ch=64)
color_model2.load_state_dict(torch.load(opts.color_model2_file)[opts.color_model_key])
color_model2.eval()
color_model2 = nn.DataParallel(color_model2.to(DEVICE))

def create_model():
    model = edict()
    model.netG = models.netSlomo(maxFlow=opts.maxFlow)
    model.netD = models.NLayerDiscriminator(in_ch=12, ndf=64, n_layers=3)
    for key in model.keys(): 
        model[key] = nn.DataParallel(model[key].to(DEVICE))
    return model

def create_dataloader():
    evalSet = datasets.FaceForensics_ct_eval_color(inputRoot=opts.inputRoot, tStride=opts.tStride, tCrop=opts.tCrop, sScale=opts.sScale)
    evalLD = DD.DataLoader(evalSet, batch_size=1,
        sampler=DD.sampler.SequentialSampler(evalSet),
        num_workers=opts.workers, pin_memory=True, drop_last=False)
    return None, evalLD

def resume_checkpoint(epoch, model, optimizer):
    print('resume model & optimizer from epoch %d'%(epoch))
    ckpt_file = '%s/ckpt/ep-%04d.pt'%(opts.saveDir, epoch)
    if os.path.isfile(ckpt_file):
        L = torch.load(ckpt_file)
        for key in model.keys():
            model[key].module.load_state_dict(L['model_'+key])
        for key in optimizer.keys():
            optimizer[key].load_state_dict(L['optimizer_'+key])
    else:
        print('checkpoint "%s" not found'%(ckpt_file))
        quit()

def initialize(model, initModel):
    if initModel == '':
        print('no further initialization')
        return
    elif os.path.isfile(initModel):
        if opts.initNonAdv:
            L = torch.load(initModel)
            model.netG.module.load_state_dict(L['model'], strict=False)
        else:
            L = torch.load(initModel)
            for key in model.keys():
                model[key].module.load_state_dict(L['model_'+key], strict=False)
        print('model initialized using [%s]'%(initModel))
    else:
        print('[%s] not found'%(initModel))
        quit()

################################################
def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def iterative_input(fakeB, realA, colors):
    # fakeB_gif = fakeB
    B, C, H, W = fakeB.shape
    fakeB_gif = []
    for i in range(B):
        _fakeB, _realA = fakeB[i].detach(), realA[i].detach()
        _fakeB = _fakeB.view(C, H*W).transpose(0, 1)
        _colors = colors[i].detach()
        dist = pairwise_distances(_fakeB, _colors)
        argmin = dist.min(dim=1)[1]
        _fakeB_gif = _colors[argmin].transpose(0, 1).view(1, C, H, W)
        fakeB_gif.append(_fakeB_gif)
    fakeB_gif = torch.cat(fakeB_gif, dim=0)
    new_input = torch.cat([fakeB, realA, fakeB_gif, realA - fakeB_gif], dim=1)
    return new_input
################################################

def evaluate(epoch, evalLD, model):
    # switch to evaluate mode (Dropout, BatchNorm, etc)
    netG = model.netG
    netG.eval()

    tags = ['PSNR', 'PSNR_gif', 'SSIM', 'SSIM_gif']
    epL = AverageMeters(tags)
    for i, (gif0s, gif1s, targets, color0s, color1s) in progressbar.progressbar(enumerate(evalLD), max_value=len(evalLD)):
        # i, (gif0s, gif1s, targets, color0s, color1s) = 0, next(iter(evalLD))
        # gif0s, gif1s: 1, T, C, H, W
        # targets: 1, T, L, C, H, W
        # color0s, color1s: 1, T, 32, 3
        _, T, L, C, H, W = targets.size()
        for j in range(T):
            gif0, gif1, target, color0, color1 = gif0s[:, j], gif1s[:, j], targets[:, j], color0s[:, j], color1s[:, j]
            gif0, gif1, target, color0, color1 = list(map(lambda x: preprocess(x).to(DEVICE), (gif0, gif1, target, color0, color1)))
            ts = np.linspace(0, 1, L)[1:L-1].tolist()
            with torch.no_grad():
                ################################################
                I0 = color_model(gif0).tanh()
                for _ in range(1):
                    new_input = iterative_input(I0, gif0, color0)
                    I0 = (I0 + color_model2(new_input)).tanh()
                I1 = color_model(gif1).tanh()
                for _ in range(1):
                    new_input = iterative_input(I1, gif1, color1)
                    I1 = (I1 + color_model2(new_input)).tanh()
                ################################################
                Its, F01, F10, Ft1s, Ft0s, Vt0s = model.netG(gif0, gif1, I0, I1, ts)
                pred = torch.cat((I0.unsqueeze(dim=1), Its, I1.unsqueeze(dim=1)), dim=1)
                pred_gif = torch.cat(list((gif0 if t<=0.5 else gif1).unsqueeze(1) for t in np.linspace(0, 1, L).tolist()), dim=1)
            comp_psnr = lambda x, y: rmse2psnr((x - y).abs().pow(2).mean().pow(0.5).item(), maxVal=2.0)
            psnr = comp_psnr(pred, target)
            psnr_gif = comp_psnr(pred_gif, target)

            tensor2im = lambda x: np.moveaxis(x.cpu().numpy(), 0, 2)
            ssim, ssim_gif = 0.0, 0.0
            for k in range(L):
                ssim += comp_ssim(tensor2im(pred[0, k]), tensor2im(target[0, k]), data_range=2.0, multichannel=True)/L
                ssim_gif += comp_ssim(tensor2im(pred_gif[0, k]), tensor2im(target[0, k]), data_range=2.0, multichannel=True)/L

            values = [psnr, psnr_gif, ssim, ssim_gif]
            assert len(tags) == len(values)
            for tag, value in zip(tags, values):
                epL[tag].update(value, 1.0/T)

    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Evaluate_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, -1, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('eval_epoch/'+tag, value, epoch)

def main_eval():
    _, evalLD = create_dataloader()
    model = create_model()
    initialize(model, opts.initModel)
    evaluate(-1, evalLD, model)

def main_vis():
    mkdir(opts.visDir)
    print('==> load model')
    model = create_model()
    initialize(model, opts.initModel)
    netG = model.netG
    netG.eval()
    print('==> create data loader')
    visSet = datasets.FaceForensics_ct_eval_color(inputRoot=opts.inputRoot, tStride=opts.tCrop, tCrop=opts.tCrop, sScale=opts.sScale)
    for i, (gif0s, gif1s, targets, color0s, color1s) in progressbar.progressbar(enumerate(visSet), max_value=min(opts.visNum, len(visSet))):
        # i, (gif0s, gif1s, targets, color0s, color1s) = 0, next(iter(visSet))
        # gif0s, gif1s: T, C, H, W
        # targets: T, L, C, H, W
        if i >= opts.visNum: break
        T, L, C, H, W = targets.size()
        ims_target = np.moveaxis(targets.view(T*L, C, H, W).numpy().astype(np.uint8), 1, 3)
        ims_gif, ims_pred = [], []
        for j in range(T):
            gif0, gif1, color0, color1 = gif0s[j:j+1], gif1s[j:j+1], color0s[j:j+1], color1s[j:j+1]
            gif0, gif1, color0, color1 = list(map(lambda x: preprocess(x).to(DEVICE), (gif0, gif1, color0, color1)))
            ts = np.linspace(0, 1, L)[1:L-1].tolist()
            with torch.no_grad():
                ################################################
                I0 = color_model(gif0).tanh()
                for _ in range(1):
                    new_input = iterative_input(I0, gif0, color0)
                    I0 = (I0 + color_model2(new_input)).tanh()
                I1 = color_model(gif1).tanh()
                for _ in range(1):
                    new_input = iterative_input(I1, gif1, color1)
                    I1 = (I1 + color_model2(new_input)).tanh()
                ################################################
                Its, _, _, _, _, _ = model.netG(gif0, gif1, I0, I1, ts)
            pred = torch.cat((I0.unsqueeze(dim=1), Its, I1.unsqueeze(dim=1)), dim=1)
            # pred_gif = torch.cat(list((gif0 if t<=0.5 else gif1).unsqueeze(1) for t in np.linspace(0, 1, L).tolist()), dim=1)
            pred_gif = torch.cat(list((gif0 if t<0.999 else gif1).unsqueeze(1) for t in np.linspace(0, 1, L).tolist()), dim=1)
            pred = np.moveaxis(postprocess(pred[0]).cpu().numpy().astype(np.uint8), 1, 3)
            pred_gif = np.moveaxis(postprocess(pred_gif[0]).cpu().numpy().astype(np.uint8), 1, 3)
            ims_gif.append(pred_gif)
            ims_pred.append(pred)
        ims_gif = np.concatenate(ims_gif, axis=0)
        ims_pred = np.concatenate(ims_pred, axis=0)
        ims_error = np.abs(ims_target.astype(float) - ims_pred.astype(float))
        ims_error = np.tile(ims_error.mean(axis=3, keepdims=True), 3)
        ims_error = (ims_error / 20.0 * 255.0).astype(np.uint8)
        ims_row1 = np.concatenate([ims_gif,  ims_target], axis=2)
        ims_row2 = np.concatenate([ims_pred, ims_error ], axis=2)
        ims_four = np.concatenate([ims_row1, ims_row2  ], axis=1)
        imageio.mimwrite('{}/{:04d}_result.mp4'.format(opts.visDir, i+1), ims_four, fps=50)

if __name__ == '__main__':
    if opts.evalMode:
        main_eval()
    if opts.visMode:
        main_vis()
