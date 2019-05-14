from utils import *

# data
parser.add_argument('--dither_mode', default='nodither', type=str, choices=['dither','nodither'], help='dither mode of input gifs')
parser.add_argument('--PtSz', default=256, type=int, help='patch size')
parser.add_argument('--tDown', default=8, type=int, help="temporal downsampling")
parser.add_argument('--pool_size', default=60, type=int, help="image pool size")
# model
parser.add_argument('--base_model_file', default='/mnt/disk/data/yangwang/Gif2Video/degif_c/exp,FF/CMD_nogan/results/g32_nodither_pt256_bt8_tr0.1/idl100_1,gdl100_1,nogan/ckpt/ep-0060.pt', type=str, help='')
parser.add_argument('--base_model_key', default='model_netG', type=str, help='')
parser.add_argument('--unroll', default=2, type=int, help='')
# loss
parser.add_argument('--w_idl', default=100, type=float, help='weight for image difference loss')
parser.add_argument('--p_idl', default=1, type=float, help='p-norm for image difference loss')
parser.add_argument('--w_gdl', default=100, type=float, help='weight for gradient difference loss')
parser.add_argument('--p_gdl', default=1, type=float, help='p-norm for gradient difference loss')
parser.add_argument('--w_ggan', default=1, type=float, help='weight for generator loss')
parser.add_argument('--w_dgan', default=1, type=float, help='weight for discriminator loss')
parser.add_argument('--gan_loss', default='GAN', type=str, choices=['GAN', 'LSGAN'], help='which GAN Loss')
parser.add_argument('--nogan', default=False, action='store_true', dest='nogan', help='do not use gan')
#
opts = parser.parse_args()
dataRoot = '/nfs/bigbrain/yangwang/Gif2Video/gif2video_data/gif_faces/'
opts.inputRoot = dataRoot + 'face_gif_image/expand1.5_size256_s1_g32_' + opts.dither_mode
manual_seed(opts.seed)

global base_model
base_model = models.UNet(3, 3, ch=64)
base_model.load_state_dict(torch.load(opts.base_model_file)[opts.base_model_key])
base_model.eval()
base_model = nn.DataParallel(base_model.to(DEVICE))

def create_dataloader():
    trSet = torchdata.gif_faces_train(inputRoot=opts.inputRoot, patchSize=opts.PtSz)
    trLD = DD.DataLoader(trSet, batch_size=opts.BtSz,
        sampler= DD.sampler.SubsetRandomSampler([0]*opts.BtSz) if opts.OneBatch else DD.sampler.RandomSampler(trSet),
        num_workers=opts.workers, pin_memory=True, drop_last=True)
    evalSet = torchdata.gif_faces_test(inputRoot=opts.inputRoot, tDown=opts.tDown)
    evalLD = DD.DataLoader(evalSet, batch_size=1,
        sampler=DD.sampler.SequentialSampler(evalSet),
        num_workers=opts.workers, pin_memory=True, drop_last=False)
    return trLD, evalLD

def create_model():
    model = edict()
    if opts.dither_mode == 'nodither':
        iter_ch = 3*4
    elif opts.dither_mode == 'dither':
        iter_ch = 3*2
    model.netG = torchmodel.UNet_simple(in_ch=iter_ch, out_ch=3, ch=64)
    model.netD = torchmodel.NLayerDiscriminator(in_ch=12, ndf=64, n_layers=3)
    for key in model.keys(): 
        model[key] = model[key].to(DEVICE)
        if DEVICE != "cpu": model[key] = nn.DataParallel(model[key])
    return model

def board_vis(epoch, gif, pred, target):
    B, C, H, W = target.shape
    error = (pred - target).abs()
    x = torch.cat((gif, target, pred, error), dim=0)
    x = vutils.make_grid(x, nrow=B, normalize=True)
    opts.board.add_image('train_batch/gif_target_pred_error', x, epoch)

################################################
if opts.dither_mode == 'nodither':
    def iterative_input(fakeB, realA, colors, nColor):
        # fakeB_gif = fakeB
        B, C, H, W = fakeB.shape
        fakeB_gif = []
        for i in range(B):
            _fakeB, _realA = fakeB[i].detach(), realA[i].detach()
            _fakeB = _fakeB.view(C, H*W).transpose(0, 1)
            _colors = colors[i, :nColor[i].item()].detach()
            dist = pairwise_distances(_fakeB, _colors)
            argmin = dist.min(dim=1)[1]
            _fakeB_gif = _colors[argmin].transpose(0, 1).view(1, C, H, W)
            fakeB_gif.append(_fakeB_gif)
        fakeB_gif = torch.cat(fakeB_gif, dim=0)
        new_input = torch.cat([fakeB, realA, fakeB_gif, realA - fakeB_gif], dim=1)
        return new_input
elif opts.dither_mode == 'dither':
    def iterative_input(fakeB, realA, colors, nColor):
        new_input = torch.cat([fakeB, realA], dim=1)
        return new_input
################################################

def train(epoch, trLD, model, optimizer, fakeABPool):
    # switch to train mode (Dropout, BatchNorm, etc)
    for key in model.keys(): model[key].train()

    tags = ['D_gan', 'D_real', 'D_fake', 'D_acc', 'G_gan', 'G_idl', 'G_gdl', 'G_total']
    epL = AverageMeters(tags)
    N = max(1, round(len(trLD) * opts.trRatio))
    for i, samples in progressbar.progressbar(enumerate(islice(trLD, N)), max_value=N):
        # i, samples = 0, next(iter(trLD))
        btSz = samples[0].shape[0]
        realA, realB, colors = list(map(lambda x: preprocess(x).to(DEVICE), samples[:3]))
        nColor = samples[3].to(DEVICE)
        ################################################
        with torch.no_grad():
            initB = base_model(realA).tanh()
        fakeBs = []
        for _ in range(opts.unroll):
            fakeB = initB if not fakeBs else fakeBs[-1]
            new_input = iterative_input(fakeB, realA, colors, nColor)
            fakeB = (fakeB + model.netG(new_input)).tanh()
            fakeBs.append(fakeB)
        ################################################
        D_input = lambda A, B: torch.cat((A, B, pad_tl(diff_xy(B))), dim=1)
        realAB = D_input(realA, realB)
        fakeAB = D_input(realA, fakeBs[-1])

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
            optimizer.netD.step()

        # (2) Update G network
        optimizer.netG.zero_grad()
        fake_logits = model.netD(fakeAB)
        g_gan = compute_G_loss(fake_logits, method=opts.gan_loss)
        g_idl = sum(Lp(opts.p_idl)(fakeB, realB) for fakeB in fakeBs) / opts.unroll
        g_gdl = sum(2 * Lp(opts.p_gdl)(diff_xy(fakeB).abs(), diff_xy(realB).abs()) for fakeB in fakeBs) / opts.unroll

        loss_g = g_gan * opts.w_ggan + g_idl * opts.w_idl + g_gdl * opts.w_gdl
        loss_g.backward()
        if d_acc.item() > 0.25:
            optimizer.netG.step()

        # tags = ['D_gan', 'D_real', 'D_fake', 'D_acc', 'G_gan', 'G_idl', 'G_gdl', 'G_total']
        values = list(map(lambda x: x.item(), [d_gan, d_real, d_fake, d_acc, g_gan, g_idl, g_gdl, loss_g]))
        assert len(tags) == len(values)
        for tag, value in zip(tags, values):
            epL[tag].update(value, btSz)
            if opts.board is not None and i%opts.dispIter==0:
                opts.board.add_scalar('train_batch/'+tag, value, epoch-1+float(i+1)/N)

        if opts.board is not None and i%opts.dispIter==0:
            board_vis(epoch, realA, fakeB, realB)

    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Train_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('train_epoch/'+tag, value, epoch)

def train_nogan(epoch, trLD, model, optimizer):
    # switch to train mode (Dropout, BatchNorm, etc)
    for key in model.keys(): model[key].train()

    tags = ['G_idl', 'G_gdl', 'G_total']
    epL = AverageMeters(tags)
    N = max(1, round(len(trLD) * opts.trRatio))
    for i, samples in progressbar.progressbar(enumerate(islice(trLD, N)), max_value=N):
        # i, samples = 0, next(iter(trLD))
        btSz = samples[0].shape[0]
        realA, realB, colors = list(map(lambda x: preprocess(x).to(DEVICE), samples[:3]))
        nColor = samples[3].to(DEVICE)
        ################################################
        with torch.no_grad():
            initB = base_model(realA).tanh()
        fakeBs = []
        for _ in range(opts.unroll):
            fakeB = initB if not fakeBs else fakeBs[-1]
            new_input = iterative_input(fakeB, realA, colors, nColor)
            fakeB = (fakeB + model.netG(new_input)).tanh()
            fakeBs.append(fakeB)
        ################################################

        # Update G network
        optimizer.netG.zero_grad()
        g_idl = sum(Lp(opts.p_idl)(fakeB, realB) for fakeB in fakeBs) / opts.unroll
        g_gdl = sum(2 * Lp(opts.p_gdl)(diff_xy(fakeB).abs(), diff_xy(realB).abs()) for fakeB in fakeBs) / opts.unroll

        loss_g = g_idl * opts.w_idl + g_gdl * opts.w_gdl
        loss_g.backward()
        optimizer.netG.step()

        # tags = ['G_idl', 'G_gdl', 'G_total']
        values = list(map(lambda x: x.item(), [g_idl, g_gdl, loss_g]))
        assert len(tags) == len(values)
        for tag, value in zip(tags, values):
            epL[tag].update(value, btSz)
            if opts.board is not None and i%opts.dispIter==0:
                opts.board.add_scalar('train_batch/'+tag, value, epoch-1+float(i+1)/N)

        if opts.board is not None and i%opts.dispIter==0:
            board_vis(epoch, realA, fakeB, realB)

    # logging
    state = edict({k:v.avg for k, v in epL.items()})
    print('Train_Summary: Epoch [{:03d}/{:03d}], {}'.format(epoch, opts.nEpoch, state))
    if opts.board is not None:
        for tag, value in state.items():
            opts.board.add_scalar('train_epoch/'+tag, value, epoch)

def evaluate(epoch, evalLD, model):
    # switch to evaluate mode (Dropout, BatchNorm, etc)
    netG = model.netG
    netG.eval()

    tags = ['PSNR', 'PSNR_gif', 'SSIM', 'SSIM_gif']
    epL = AverageMeters(tags)
    for i, (inputs, targets, colors, nColor) in progressbar.progressbar(enumerate(evalLD), max_value=len(evalLD)):
        # i, (inputs, targets, colors, nColor) = 0, next(iter(evalLD))
        _, T, _, H, W = inputs.size()
        for j in range(T):
            input, target, _colors, _nColor = inputs[:, j], targets[:, j], colors[:, j], nColor[:, j]
            input, target, _colors = list(map(lambda x: preprocess(x).to(DEVICE), (input, target, _colors)))
            _nColor = _nColor.to(DEVICE)
            ################################################
            realA, realB = input, target
            with torch.no_grad():
                initB = base_model(realA).tanh()
                fakeB = None
                for _ in range(opts.unroll):
                    if fakeB is None: fakeB = initB
                    new_input = iterative_input(fakeB, realA, _colors, _nColor)
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
    optimizer = create_optimizer(model, opts)
    scheduler = create_scheduler(optimizer, opts)

    print('==> initialize with checkpoint or initModel ?')
    FIRST_EPOCH = 1 # do not change
    USE_CKPT = opts.checkpoint >= FIRST_EPOCH
    if USE_CKPT:
        resume_checkpoint(opts.checkpoint, model, optimizer, opts)
        start_epoch = opts.checkpoint + 1
    else:
        initialize(model, opts.initModel)
        start_epoch = FIRST_EPOCH

    print('==> start training from epoch %d'%(start_epoch))
    for epoch in range(start_epoch, FIRST_EPOCH + opts.nEpoch):
        print('\nEpoch {}:\n'.format(epoch))
        for key in scheduler.keys():
            scheduler[key].step(epoch-1)
            lr = scheduler[key].optimizer.param_groups[0]['lr']
            print('learning rate of {} is set to {}'.format(key, lr))
            if opts.board is not None: opts.board.add_scalar('lr_schedule/'+key, lr, epoch)
        if opts.nogan:
            train_nogan(epoch, trLD, model, optimizer)
        else:
            train(epoch, trLD, model, optimizer, fakeABPool)
        if not opts.OneBatch and epoch%opts.saveStep==0:
            save_checkpoint(epoch, model, optimizer, opts)
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
    visSet = torchdata.gif_faces_test(inputRoot=opts.inputRoot, tDown=opts.tDown)
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
            nColor = samples[3][sbStart:sbEnd+1].to(DEVICE)
            ################################################
            realA, realB = inputs, targets
            with torch.no_grad():
                initB = base_model(realA).tanh()
                fakeB = None
                for _ in range(opts.unroll):
                    if fakeB is None: fakeB = initB
                    new_input = iterative_input(fakeB, realA, colors, nColor)
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
        # imageio.mimwrite('{}/{:04d}_result.mp4'.format(opts.visDir, i+1), ims_four)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
        fps = 25.0 / opts.tDown 
        _, H_, W_, _ = ims_four.shape
        video = cv2.VideoWriter('{}/{:04d}_result.mp4'.format(opts.visDir, i+1), fourcc, fps, (W_, H_))
        for j, im in enumerate(ims_four):
            video.write(im[:, :, ::-1])
        video.release()

def main_apply():
    print('==> read gif frames')
    ims = imageio.mimread(opts.applyFile)
    L, H, W = len(ims), ims[0].shape[0], ims[0].shape[1]
    print('==> load model')
    model = create_model()
    initialize(model, opts.initModel)
    netG = model.netG
    netG.eval()
    print('==> processing')
    ims_gif, ims_pred = [], []
    for i in range(L):
        # make sure it's 3 channels
        im = ims[i]
        if im.ndim == 2:
            im = np.broadcast_to(np.expand_dims(im, 2), list(im.shape) + [3])
        elif im.ndim == 3:
            im = im[:,:,:3]
        ims_gif.append(im.copy())
        # gif2video
        im = torch.ByteTensor(np.moveaxis(im, 2, 0)).view(1, 3, H, W)
        im = preprocess(im).to(DEVICE)
        with torch.no_grad():
            predict = netG(im).tanh()
        predict = postprocess(predict[0]).cpu().numpy()
        predict = predict.astype(np.uint8)
        predict = np.moveaxis(predict, 0, 2)
        ims_pred.append(predict)
    ims_gif = np.asarray(ims_gif)
    ims_pred = np.asarray(ims_pred)
    ims_row = np.concatenate([ims_gif, ims_pred], axis=2)
    imageio.mimwrite('{}.mp4'.format(opts.applyFile), ims_row)

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
        options_text = print_options(opts, parser)
        opts.board.add_text('options', options_text, opts.checkpoint)
        main_train()
        opts.board.close()