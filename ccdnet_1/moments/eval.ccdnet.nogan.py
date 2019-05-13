from utils import *

parser = argparse.ArgumentParser(description='PyTorch Training')
# data
parser.add_argument('--inputRoot', default='/nfs/bigbrain/yangwang/Gif2Video/data,video2gif/gifs/size360_div12_s1_g32_nodither/', type=str, help='root of gifs')
parser.add_argument('--workers', default=2, type=int, help='number of workers for dataloader')
parser.add_argument('--PtSz', default=256, type=int, help='patch size')
parser.add_argument('--sScale', default=1, type=int, help='spatial upsampling for input')
parser.add_argument('--BtSz', default=8, type=int, help='batch size')
parser.add_argument('--tDown', default=8, type=int, help="temporal downsampling")
# model
parser.add_argument('--base_model_file', default='/mnt/disk/data/yangwang/Gif2Video/degif_c/exp,FF/CMD_nogan/results/g32_nodither_pt256_bt8_tr0.1/idl100_1,gdl100_1,nogan/ckpt/ep-0060.pt', type=str, help='')
parser.add_argument('--base_model_key', default='model_netG', type=str, help='')
parser.add_argument('--unroll', default=1, type=int, help='')
# loss
# init & checkpoint
parser.add_argument('--initModel', default='', help='init model in absence of checkpoint')
parser.add_argument('--initNonAdv', action='store_true', default=False, help='init from a non-adv checkpoint')
parser.add_argument('--checkpoint', default=0, type=int, help='resume from a checkpoint')
# save & display
# other mode
parser.add_argument('--evalMode', action='store_true', default=False, dest='evalMode', help='evaluation mode')
parser.add_argument('--visMode', action='store_true', default=False, dest='visMode', help='visualization mode')
parser.add_argument('--visDir', default='visual', type=str, help="dir to store visualization")
parser.add_argument('--visNum', default=1000, type=int, help="number of videos to visualize")
# misc
parser.add_argument('--seed', default=1, type=int, help='random seed for torch/numpy')
opts = parser.parse_args()
manual_seed(opts.seed)

global base_model
base_model = models.UNet(3, 3, ch=64)
base_model.load_state_dict(torch.load(opts.base_model_file)[opts.base_model_key])
base_model.eval()
base_model = nn.DataParallel(base_model.to(DEVICE))

def create_model():
    model = edict()
    model.netG = models.UNet(in_ch=3*4, out_ch=3, ch=64)
    model.netD = models.NLayerDiscriminator(in_ch=12, ndf=64, n_layers=3)
    for key in model.keys(): 
        model[key] = nn.DataParallel(model[key].to(DEVICE))
    return model

def create_dataloader():
    evalSet = datasets.Video2Gif_eval_color(inputRoot=opts.inputRoot, subset='test', tStride=opts.tDown, sScale=opts.sScale)
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
        _colors = colors[i, :32].detach()
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
    for i, (inputs, targets, colors) in progressbar.progressbar(enumerate(evalLD), max_value=len(evalLD)):
        # i, (inputs, targets, colors) = 0, next(iter(evalLD))
        _, T, _, H, W = inputs.size()
        for j in range(T):
            input, target, _colors = inputs[:, j], targets[:, j], colors[:, j]
            input, target, _colors = list(map(lambda x: preprocess(x).to(DEVICE), (input, target, _colors)))
            ################################################
            realA, realB = input, target
            with torch.no_grad():
                initB = base_model(realA).tanh()
                fakeB = None
                for _ in range(opts.unroll):
                    if fakeB is None: fakeB = initB
                    new_input = iterative_input(fakeB, realA, _colors)
                    fakeB = (fakeB + model.netG(new_input)).tanh()
            pred = fakeB
            ################################################
            comp_psnr = lambda x, y: rmse2psnr((x - y).abs().pow(2).mean().pow(0.5).item(), maxVal=2.0)
            psnr = comp_psnr(pred, target)
            psnr_gif = comp_psnr(input, target)

            tensor2im = lambda x: np.moveaxis(x[0].cpu().numpy(), 0, 2)
            ssim = comp_ssim(tensor2im(pred), tensor2im(target), data_range=2.0, multichannel=True)
            ssim_gif = comp_ssim(tensor2im(input), tensor2im(target), data_range=2.0, multichannel=True)

            values = [psnr, psnr_gif, ssim, ssim_gif]
            assert len(tags) == len(values)
            for tag, value in zip(tags, values):
                epL[tag].update(value, 1.0/T)

    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Evaluate_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, -1, state))

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
    visSet = datasets.Video2Gif_eval_color(inputRoot=opts.inputRoot, subset='test', tStride=opts.tDown, sScale=opts.sScale)
    for i, samples in progressbar.progressbar(enumerate(visSet), max_value=len(visSet)):
        # i, sample = 0, next(iter(visSet))
        if i >= opts.visNum: break
        ims_input = np.moveaxis(samples[0].numpy(), 1, 3)
        ims_target = np.moveaxis(samples[1].numpy(), 1, 3)
        ims_pred = []
        btSz = samples[0].shape[0]
        sBtSz = 4
        for sbStart in range(0, btSz, sBtSz):
            sbEnd = min(sbStart+sBtSz-1, btSz-1)
            inputs, targets, colors = list(map(lambda x: preprocess(x[sbStart:sbEnd+1]).to(DEVICE), samples[:3]))
            ################################################
            realA, realB = inputs, targets
            with torch.no_grad():
                initB = base_model(realA).tanh()
                fakeB = None
                for _ in range(opts.unroll):
                    if fakeB is None: fakeB = initB
                    new_input = iterative_input(fakeB, realA, colors)
                    fakeB = (fakeB + model.netG(new_input)).tanh()
            predicts = fakeB
            ################################################
            predicts = postprocess(predicts).cpu().numpy()
            predicts = predicts.astype(np.uint8)
            predicts = np.moveaxis(predicts, 1, 3)
            ims_pred.append(predicts)
        ims_pred = np.concatenate(ims_pred, axis=0)
        ims_error = np.abs(ims_target.astype(float) - ims_pred.astype(float))
        ims_error = np.tile(ims_error.mean(axis=3, keepdims=True), 3)
        ims_error = (ims_error / 20.0 * 255.0).astype(np.uint8)
        ims_row1 = np.concatenate([ims_input, ims_target], axis=2)
        ims_row2 = np.concatenate([ims_pred,  ims_error ], axis=2)
        ims_four = np.concatenate([ims_row1,  ims_row2  ], axis=1)
        imageio.mimwrite('{}/{:04d}_result.mp4'.format(opts.visDir, i+1), ims_four)

if __name__ == '__main__':
    if opts.evalMode:
        main_eval()
    if opts.visMode:
        main_vis()
