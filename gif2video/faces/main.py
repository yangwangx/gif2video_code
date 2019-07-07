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
parser.add_argument('--trRatio', default=1, type=float, help='ratio of training data per epoch')
parser.add_argument('--pool_size', default=60, type=int, help="image pool size")
parser.add_argument('--OneBatch', default=False, action='store_true', dest='OneBatch', help='debug with one batch')
# model
parser.add_argument('--color_model_file', default='/nfs/bigfovea/yangwang/Gif2Video/degif_c/exp,FF/CMD_gan/results/g32_nodither_pt256_bt8_tr0.1/idl100_1,gdl100_1,gan_d1_g1/lr0.0002/ckpt/ep-0030.pt', type=str, help='')
parser.add_argument('--color_model_key', default='model_netG', type=str, help='')
parser.add_argument('--maxFlow', default=30, type=float, help='maximum flow value, use for rescaling')
# loss
parser.add_argument('--nogan', default=False, action='store_true', dest='nogan', help='do not use gan')
parser.add_argument('--gan_loss', default='GAN', type=str, choices=['GAN', 'LSGAN'], help='which GAN Loss')
parser.add_argument('--w_dgan', default=1, type=float, help='weight for discriminator loss')
parser.add_argument('--w_ggan', default=1, type=float, help='weight for generator loss')
parser.add_argument('--w_idl', default=0.5, type=float, help='weight for image difference loss')
parser.add_argument('--w_gdl', default=0.5, type=float, help='weight for gradient difference loss')
parser.add_argument('--w_warp', default=0.5, type=float, help='weight for image difference loss')
parser.add_argument('--w_smooth', default=1, type=float, help='weight for gradient difference loss')
parser.add_argument('--L_warp_outlier', default=40.0, type=float, help='initial outlier value for warp loss')
# optimizer
parser.add_argument('--solver', default='adam', choices=['adam','sgd'], help='which solver')
parser.add_argument('--MM', default=0.9, type=float, help='momentum')
parser.add_argument('--Beta', default=0.999, type=float, help='beta for adam')
parser.add_argument('--WD', default=1e-4, type=float, help='weight decay')
parser.add_argument('--GC', default=1.0, type=float, help='gradient clipping')
# learning rate
parser.add_argument('--LRPolicy', default='constant', type=str, choices=['constant', 'step', 'steps', 'exponential',], help='learning rate policy')
parser.add_argument('--gamma', default=0.1, type=float, help='decay rate for learning rate')
parser.add_argument('--LRStart', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--LRStep', default=100, type=int, help='steps to change learning rate')
parser.add_argument('--LRSteps', default=[], type=int, nargs='+', dest='LRSteps', help='epochs before cutting learning rate')
parser.add_argument('--nEpoch', default=500, type=int, help='total epochs to run')
# init & checkpoint
parser.add_argument('--initModel', default='', help='init model in absence of checkpoint')
parser.add_argument('--initNonAdv', action='store_true', default=False, help='init from a non-adv checkpoint')
parser.add_argument('--checkpoint', default=0, type=int, help='resume from a checkpoint')
# save & display
parser.add_argument('--saveDir', default='results/default/', help='directory to save/log experiments')
parser.add_argument('--saveStep', default=10, type=int, help='epoch step for snapshot')
parser.add_argument('--evalStep', default=50, type=int, help='epoch step for evaluation')
parser.add_argument('--dispIter', default=20, type=int, help='batch step for tensorboard')
# other mode
parser.add_argument('--evalMode', action='store_true', default=False, dest='evalMode', help='evaluation mode')
parser.add_argument('--visMode', action='store_true', default=False, dest='visMode', help='visualization mode')
parser.add_argument('--visDir', default='visual', type=str, help="dir to store visualization")
parser.add_argument('--visNum', default=1000, type=int, help="number of videos to visualize")
parser.add_argument('--applyMode', action='store_true', default=False, dest='applyMode', help='apply model to one gif')
parser.add_argument('--applyFile', default='', type=str, help='path to gif')
parser.add_argument('--applyT', default=2, type=int, help='factor for temporal interpolation')
# misc
parser.add_argument('--seed', default=1, type=int, help='random seed for torch/numpy')
opts = parser.parse_args()
manual_seed(opts.seed)

global color_model
color_model = models.UNet_rgb(3, 3, ch=64)
color_model.load_state_dict(torch.load(opts.color_model_file)[opts.color_model_key])
color_model.eval()
color_model = nn.DataParallel(color_model.to(DEVICE))

def create_model():
    model = edict()
    model.netG = models.netSlomo(maxFlow=opts.maxFlow)
    model.netD = models.NLayerDiscriminator(in_ch=12, ndf=64, n_layers=3)
    for key in model.keys(): 
        model[key] = nn.DataParallel(model[key].to(DEVICE))
    return model

def create_optimizer(model):
    optimizer = edict()
    if opts.solver == 'sgd':
        solver = partial(torch.optim.SGD, lr=opts.LRStart, momentum=opts.MM, weight_decay=opts.WD)
    elif opts.solver == 'adam':
        solver = partial(torch.optim.Adam, lr=opts.LRStart, betas=(opts.MM, opts.Beta), weight_decay=opts.WD)
    else:
        raise ValueError('optim solver "{}" not defined'.format(opts.solver))
    for key in model.keys():
        optimizer[key] = solver(model[key].parameters())
    return optimizer

def create_scheduler(optimizer):
    scheduler = edict()
    for key in optimizer.keys():
        op = optimizer[key]
        if opts.LRPolicy == 'constant':
            scheduler[key] = optim.lr_scheduler.ExponentialLR(op, gamma=1.0)
        elif opts.LRPolicy == 'step':
            scheduler[key] = optim.lr_scheduler.StepLR(op, opts.LRStep, gamma=opts.gamma)
        elif opts.LRPolicy == 'steps':
            scheduler[key] = optim.lr_scheduler.MultiStepLR(op, opts.LRSteps, gamma=opts.gamma)
        elif opts.LRPolicy == 'exponential':
            scheduler[key] = optim.lr_scheduler.ExponentialLR(op, gamma=opts.gamma)
        else:
            raise ValueError('learning rate policy "{}" not defined'.format(opts.LRPolicy))
    return scheduler

def create_dataloader():
    trSet = datasets.FaceForensics_ct_train(inputRoot=opts.inputRoot, tCrop=opts.tCrop, sScale=opts.sScale, sCrop=opts.sCrop)
    trLD = DD.DataLoader(trSet, batch_size=opts.BtSz,
        sampler= DD.sampler.SubsetRandomSampler([0]*opts.BtSz) if opts.OneBatch else DD.sampler.RandomSampler(trSet),
        num_workers=opts.workers, pin_memory=True, drop_last=True)
    evalSet = datasets.FaceForensics_ct_eval(inputRoot=opts.inputRoot, tStride=opts.tStride, tCrop=opts.tCrop, sScale=opts.sScale)
    evalLD = DD.DataLoader(evalSet, batch_size=1,
        sampler=DD.sampler.SequentialSampler(evalSet),
        num_workers=opts.workers, pin_memory=True, drop_last=False)
    return trLD, evalLD

def mkdir_save(state, state_file):
    mkdir(os.path.dirname(state_file))
    torch.save(state, state_file)

def save_checkpoint(epoch, model=None, optimizer=None):
    print('save model & optimizer @ epoch %d'%(epoch))
    ckpt_file = '%s/ckpt/ep-%04d.pt'%(opts.saveDir, epoch)
    state = {}
    # an error occurs when I use edict
    # possible reason: optim.state_dict() has an 'state' attribute
    state['epoch'] = epoch
    for key in model.keys():
        state['model_'+key] = model[key].module.state_dict()
    for key in optimizer.keys():
        state['optimizer_'+key] = optimizer[key].state_dict()
    mkdir_save(state, ckpt_file)

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

def board_vis(epoch, frm1, frm0, frm10, frm01, F01, F10, Vt0s, gif, target, imgs):
    B, L, C, H, W = target.shape
    # I0 and I1
    im0, im1 = frm0[:1].detach(), frm1[:1].detach()
    im0_warp, im1_warp = frm10[:1].detach(), frm01[:1].detach()
    im0_err, im1_err = (im0 - im0_warp).abs(), (im1 - im1_warp).abs()
    im01_diff = (im0 - im1).abs()
    x = torch.cat((im0, im1, im0_warp, im1_warp, im0_err, im1_err, im01_diff), dim=0)
    x = vutils.make_grid(x, nrow=2, normalize=True)
    opts.board.add_image('train_batch/i0_i1', x, epoch)
    # flow
    flow01, flow10 = F01[:1].detach(), F10[:1].detach()
    flow01 = torch.cat([flow01, flow01.new_zeros(1, 1, H, W)], dim=1)
    flow10 = torch.cat([flow10, flow10.new_zeros(1, 1, H, W)], dim=1)
    x = torch.cat([flow01, flow10], dim=0)
    x = vutils.make_grid(x, nrow=2, normalize=True, range=(-1, 1))
    opts.board.add_image('train_batch/f01_f10', x, epoch)
    # vis_map
    vis0s = Vt0s[0].detach().expand(-1, 3, -1, -1)
    vis1s = 1 - vis0s
    x = torch.cat([vis0s, vis1s], dim=0)
    x = vutils.make_grid(x, nrow=L-2, normalize=True)
    opts.board.add_image('train_batch/vis0_vis1', x, epoch)
    # interp
    ims_gif = gif[0].detach()
    ims_gt = target[0].detach()
    ims_est = imgs[0].detach()
    ims_err = (ims_est - ims_gt).abs()
    x = torch.cat((ims_gif, ims_gt, ims_est, ims_err), dim=0)
    x = vutils.make_grid(x, nrow=L, normalize=True)
    opts.board.add_image('train_batch/recover', x, epoch)

def train(epoch, trLD, model, optimizer, fakeABPool):
    # switch to train mode (Dropout, BatchNorm, etc)
    for key in model.keys():
        model[key].train()

    tags = ['D_gan', 'D_real', 'D_fake', 'D_acc'] + ['L_gan', 'L_idl', 'L_gdl', 'L_warp', 'L_smooth', 'L_total']
    epL = AverageMeters(tags)
    N = max(1, round(len(trLD) * opts.trRatio))
    for i, samples in progressbar.progressbar(enumerate(islice(trLD, N)), max_value=N):
        # i, samples = 0, next(iter(trLD))
        btSz = samples[0].shape[0]
        gif, target = list(map(lambda x: preprocess(x).to(DEVICE), samples))
        B, L, C, H, W = gif.shape
        gif0, gif1 = gif[:, 0], gif[:, -1]
        frm0, frm1, frm_ts = target[:, 0], target[:, -1], target[:, 1:L-1]
        ts = np.linspace(0, 1, L)[1:L-1].tolist()
        with torch.no_grad():
            I0, I1 = color_model(gif0).tanh(), color_model(gif1).tanh()
        Its, F01, F10, Ft1s, Ft0s, Vt0s = model.netG(gif0, gif1, I0, I1, ts)
        imgs = torch.cat((I0.unsqueeze(dim=1), Its, I1.unsqueeze(dim=1)), dim=1)
        D_input = lambda A, B: torch.cat((A, B, pad_tl(diff_xy(B))), dim=1)
        realAB = D_input(gif.view(B*L, -1, H, W), target.view(B*L, -1, H, W))
        fakeAB = D_input(gif.view(B*L, -1, H, W), imgs.view(B*L, -1, H, W))

        # (1) Update D network
        optimizer.netD.zero_grad()
        fakeAB_ = fakeABPool.query(fakeAB.detach()).to(DEVICE)
        real_logits = model.netD(realAB)
        fake_logits = model.netD(fakeAB_)
        d_gan, d_real, d_fake = compute_D_loss(real_logits, fake_logits, method=opts.gan_loss)
        d_acc, _, _ = compute_D_acc(real_logits, fake_logits, method=opts.gan_loss)

        loss_d = d_gan * opts.w_dgan
        loss_d.backward()
        if d_acc.item() < 0.75:
            nn.utils.clip_grad_norm_(model.netD.parameters(), opts.GC)
            optimizer.netD.step()

        # (2) Update G network
        optimizer.netG.zero_grad()
        fake_logits = model.netD(fakeAB)
        L_gan = compute_G_loss(fake_logits, method=opts.gan_loss)
        L_idl = 127.5*(f_idl(I0, frm0) + f_idl(I1, frm1) + f_idl(Its, frm_ts))
        L_gdl = 127.5*(f_gdl(I0, frm0) + f_gdl(I1, frm1) + f_gdl(Its, frm_ts))
        L_smooth = opts.maxFlow*(f_smooth(F01) + f_smooth(F10))
        frm10 = models.backwarp(frm1, F01*opts.maxFlow)
        frm01 = models.backwarp(frm0, F10*opts.maxFlow)
        frm1ts = torch.cat(list(models.backwarp(frm1, Ft1s[:, i]*opts.maxFlow).unsqueeze(1) for i in range(Ft1s.shape[1])), dim=1)
        frm0ts = torch.cat(list(models.backwarp(frm0, Ft0s[:, i]*opts.maxFlow).unsqueeze(1) for i in range(Ft0s.shape[1])), dim=1)
        L_warp = 127.5*(f_idl(frm10, frm0) + f_idl(frm01, frm1) + f_idl(frm1ts, frm_ts) + f_idl(frm0ts, frm_ts))

        Loss_g = L_gan * opts.w_ggan + L_idl * opts.w_idl + L_gdl * opts.w_gdl + L_warp * opts.w_warp + L_smooth * opts.w_smooth
        Loss_g.backward()
        if d_acc.item() > 0.25 and L_warp < opts.L_warp_outlier:
            nn.utils.clip_grad_norm_(model.netG.parameters(), opts.GC)
            optimizer.netG.step()

        # tags = ['D_gan', 'D_real', 'D_fake', 'D_acc'] + ['L_gan', 'L_idl', 'L_gdl', 'L_warp', 'L_smooth', 'L_total']
        values = list(map(lambda x: x.item(), [d_gan, d_real, d_fake, d_acc, L_gan, L_idl, L_gdl, L_warp, L_smooth, Loss_g]))
        assert len(tags) == len(values)
        for tag, value in zip(tags, values):
            epL[tag].update(value, btSz)
            if opts.board is not None and i%opts.dispIter==0:
                opts.board.add_scalar('train_batch/'+tag, value, epoch-1+float(i+1)/N)

        if opts.board is not None and i%opts.dispIter==0:
            board_vis(epoch, frm1, frm0, frm10, frm01, F01, F10, Vt0s, gif, target, imgs)

    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Train_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('train_epoch/'+tag, value, epoch)

    opts.L_warp_outlier = epL['L_warp'].avg * 1.5
    print('outlier threshold for L_warp is set to {}'.format(opts.L_warp_outlier))

def train_nogan(epoch, trLD, model, optimizer):
    # switch to train mode (Dropout, BatchNorm, etc)
    for key in model.keys():
        model[key].train()

    tags = ['L_idl', 'L_gdl', 'L_warp', 'L_smooth', 'L_total']
    epL = AverageMeters(tags)
    N = max(1, round(len(trLD) * opts.trRatio))
    for i, samples in progressbar.progressbar(enumerate(islice(trLD, N)), max_value=N):
        # i, samples = 0, next(iter(trLD))
        btSz = samples[0].shape[0]
        gif, target = list(map(lambda x: preprocess(x).to(DEVICE), samples))
        B, L, C, H, W = gif.shape
        gif0, gif1 = gif[:, 0], gif[:, -1]
        frm0, frm1, frm_ts = target[:, 0], target[:, -1], target[:, 1:L-1]
        ts = np.linspace(0, 1, L)[1:L-1].tolist()
        with torch.no_grad():
            I0, I1 = color_model(gif0).tanh(), color_model(gif1).tanh()
        Its, F01, F10, Ft1s, Ft0s, Vt0s = model.netG(gif0, gif1, I0, I1, ts)
        imgs = torch.cat((I0.unsqueeze(dim=1), Its, I1.unsqueeze(dim=1)), dim=1)

        # Update G network
        optimizer.netG.zero_grad()
        L_idl = 127.5*(f_idl(I0, frm0) + f_idl(I1, frm1) + f_idl(Its, frm_ts))
        L_gdl = 127.5*(f_gdl(I0, frm0) + f_gdl(I1, frm1) + f_gdl(Its, frm_ts))
        L_smooth = opts.maxFlow*(f_smooth(F01) + f_smooth(F10))
        frm10 = models.backwarp(frm1, F01*opts.maxFlow)
        frm01 = models.backwarp(frm0, F10*opts.maxFlow)
        frm1ts = torch.cat(list(models.backwarp(frm1, Ft1s[:, i]*opts.maxFlow).unsqueeze(1) for i in range(Ft1s.shape[1])), dim=1)
        frm0ts = torch.cat(list(models.backwarp(frm0, Ft0s[:, i]*opts.maxFlow).unsqueeze(1) for i in range(Ft0s.shape[1])), dim=1)
        L_warp = 127.5*(f_idl(frm10, frm0) + f_idl(frm01, frm1) + f_idl(frm1ts, frm_ts) + f_idl(frm0ts, frm_ts))

        Loss_g = L_idl * opts.w_idl + L_gdl * opts.w_gdl + L_warp * opts.w_warp + L_smooth * opts.w_smooth
        Loss_g.backward()
        if L_warp < opts.L_warp_outlier:
            nn.utils.clip_grad_norm_(model.netG.parameters(), opts.GC)
            optimizer.netG.step()

        # tags = ['L_idl', 'L_gdl', 'L_warp', 'L_smooth', 'G_Loss']
        values = list(map(lambda x: x.item(), [L_idl, L_gdl, L_warp, L_smooth, Loss_g]))
        assert len(tags) == len(values)
        for tag, value in zip(tags, values):
            epL[tag].update(value, btSz)
            if opts.board is not None and i%opts.dispIter==0:
                opts.board.add_scalar('train_batch/'+tag, value, epoch-1+float(i+1)/N)

        if opts.board is not None and i%opts.dispIter==0:
            board_vis(epoch, frm1, frm0, frm10, frm01, F01, F10, Vt0s, gif, target, imgs)

    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Train_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('train_epoch/'+tag, value, epoch)

    opts.L_warp_outlier = epL['L_warp'] * 1.5
    print('outlier threshold for L_warp is set to {}'.format(opts.L_warp_outlier))

def evaluate(epoch, evalLD, model):
    # switch to evaluate mode (Dropout, BatchNorm, etc)
    netG = model.netG
    netG.eval()

    tags = ['PSNR', 'PSNR_gif', 'SSIM', 'SSIM_gif']
    epL = AverageMeters(tags)
    for i, (gif0s, gif1s, targets) in progressbar.progressbar(enumerate(evalLD), max_value=len(evalLD)):
        # i, (gif0s, gif1s, targets) = 0, next(iter(evalLD))
        # gif0s, gif1s: 1, T, C, H, W
        # targets: 1, T, L, C, H, W
        _, T, L, C, H, W = targets.size()
        for j in range(T):
            gif0, gif1, target = gif0s[:, j], gif1s[:, j], targets[:, j]
            gif0, gif1, target = list(map(lambda x: preprocess(x).to(DEVICE), (gif0, gif1, target)))
            ts = np.linspace(0, 1, L)[1:L-1].tolist()
            with torch.no_grad():
                I0, I1 = color_model(gif0).tanh(), color_model(gif1).tanh()
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
    print('Evaluate_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('eval_epoch/'+tag, value, epoch)

def main_train():
    print('==> create dataset loader')
    trLD, evalLD = create_dataloader()
    fakeABPool = ImagePool(opts.pool_size)

    print('==> create model, optimizer, scheduler')
    model = create_model()
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)

    print('==> initialize with checkpoint or initModel ?')
    FIRST_EPOCH = 1 # do not change
    USE_CKPT = opts.checkpoint >= FIRST_EPOCH
    if USE_CKPT:
        resume_checkpoint(opts.checkpoint, model, optimizer)
        start_epoch = opts.checkpoint + 1
    else:
        initialize(model, opts.initModel)
        start_epoch = FIRST_EPOCH

    print('==> start training from epoch %d'%(start_epoch))
    for epoch in range(start_epoch, FIRST_EPOCH + opts.nEpoch):
        for key in scheduler.keys():
            scheduler[key].step(epoch-1)
            print('Epoch {}: learning rate of {} is set to {}'.format(epoch, key, scheduler[key].get_lr()))
        if opts.nogan:
            train_nogan(epoch, trLD, model, optimizer)
        else:
            train(epoch, trLD, model, optimizer, fakeABPool)
        if not opts.OneBatch and epoch%opts.saveStep==0:
            save_checkpoint(epoch, model, optimizer)
        if not opts.OneBatch and epoch%opts.evalStep==0:
            evaluate(epoch, evalLD, model)

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
    visSet = datasets.FaceForensics_ct_eval(inputRoot=opts.inputRoot, tStride=opts.tCrop, tCrop=opts.tCrop, sScale=opts.sScale)
    for i, (gif0s, gif1s, targets) in progressbar.progressbar(enumerate(visSet), max_value=min(opts.visNum, len(visSet))):
        # i, (gif0s, gif1s, targets) = 0, next(iter(visSet))
        # gif0s, gif1s: T, C, H, W
        # targets: T, L, C, H, W
        if i >= opts.visNum: break
        T, L, C, H, W = targets.size()
        ims_target = np.moveaxis(targets.view(T*L, C, H, W).numpy().astype(np.uint8), 1, 3)
        ims_gif, ims_pred = [], []
        for j in range(T):
            gif0, gif1 = gif0s[j:j+1], gif1s[j:j+1]
            gif0, gif1 = list(map(lambda x: preprocess(x).to(DEVICE), (gif0, gif1)))
            ts = np.linspace(0, 1, L)[1:L-1].tolist()
            with torch.no_grad():                
                I0, I1 = color_model(gif0).tanh(), color_model(gif1).tanh()
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
        ims_row1 = np.concatenate([ims_gif, ims_target], axis=2)
        ims_row2 = np.concatenate([ims_pred,  ims_error ], axis=2)
        ims_four = np.concatenate([ims_row1,  ims_row2  ], axis=1)
        fourcc = cv2.VideoWriter_fourcc(*'x264') #(*'DIVX')  # 'x264' doesn't work
        fps = 25.0 
        _, H_, W_, _ = ims_four.shape
        video = cv2.VideoWriter('{}/{:04d}_result.mp4'.format(opts.visDir, i+1), fourcc, fps, (W_, H_))
        for j, im in enumerate(ims_four):
            video.write(im[:, :, ::-1])
        video.release()

def main_apply():
    print('==> read gif frames')
    ims = imageio.mimread(opts.applyFile)
    L, H, W = len(ims), ims[0].shape[0], ims[0].shape[1]
    for i in range(L):
        if ims[i].ndim == 2:
            ims[i] = np.broadcast_to(np.expand_dims(ims[i], 2), list(ims[i].shape) + [3])
        elif ims[i].ndim == 3:
            ims[i] = ims[i][:,:,:3]    
    print('==> load model')
    model = create_model()
    initialize(model, opts.initModel)
    netG = model.netG
    netG.eval()
    print('==> processing')
    ims_gif, ims_pred = [], []
    for i in range(L-1):
        gif0, gif1 = ims[i], ims[i+1]
        im2cutensor = lambda im: preprocess(torch.ByteTensor(np.moveaxis(im, 2, 0)).view(1, 3, H, W)).to(DEVICE)
        gif0 = im2cutensor(gif0)
        gif1 = im2cutensor(gif1)
        ts = np.linspace(0, 1, opts.applyT)[1:opts.applyT].tolist()
        with torch.no_grad():
            I0, I1 = color_model(gif0).tanh(), color_model(gif1).tanh()
            Its, _, _, _, _, _ = model.netG(gif0, gif1, I0, I1, ts)
        pred = torch.cat((I0.unsqueeze(dim=1), Its, I1.unsqueeze(dim=1)), dim=1)
        pred_gif = torch.cat(list((gif0 if t<=0.5 else gif1).unsqueeze(1) for t in np.linspace(0, 1, opts.applyT+1).tolist()), dim=1)
        #pred_gif = torch.cat(list((gif0 if t<0.999 else gif1).unsqueeze(1) for t in np.linspace(0, 1, opts.applyT+1).tolist()), dim=1)
        pred = np.moveaxis(postprocess(pred[0][:-1]).cpu().numpy().astype(np.uint8), 1, 3)
        pred_gif = np.moveaxis(postprocess(pred_gif[0][:-1]).cpu().numpy().astype(np.uint8), 1, 3)
        ims_gif.append(pred_gif)
        ims_pred.append(pred)
    ims_gif = np.concatenate(ims_gif, axis=0)
    ims_pred = np.concatenate(ims_pred, axis=0)
    ims_row = np.concatenate([ims_gif, ims_pred], axis=2)
    imageio.mimwrite('{}_t{}.mp4'.format(opts.applyFile, opts.applyT), ims_row)

if __name__ == '__main__':
    trainMode = True
    if opts.evalMode:
        opts.board = None
        trainMode = False
        main_eval()
    if opts.visMode:
        opts.board = None
        trainMode = False
        main_vis()
    if opts.applyMode:
        opts.board = None
        trainMode = False
        main_apply()
    if trainMode:
        opts.board = SummaryWriter(os.path.join(opts.saveDir, 'board'))
        print_options(opts, parser)
        main_train()
        opts.board.close()
