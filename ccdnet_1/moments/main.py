from utils import *

parser = argparse.ArgumentParser(description='PyTorch Training')
# data
parser.add_argument('--inputRoot', default='/nfs/bigbrain/yangwang/Gif2Video/data,Video2Gif/gifs/size360_div12_s1_g32_nodither/', type=str, help='root of gifs')
parser.add_argument('--subset_train', default='train', type=str, help='which subset for training')
parser.add_argument('--subset_eval', default='valid', type=str, help='which subset for evaluation')
parser.add_argument('--workers', default=2, type=int, help='number of workers for dataloader')
parser.add_argument('--PtSz', default=256, type=int, help='patch size')
parser.add_argument('--sScale', default=1, type=int, help='spatial upsampling for input')
parser.add_argument('--BtSz', default=8, type=int, help='batch size')
parser.add_argument('--trRatio', default=1, type=float, help='ratio of training data per epoch')
parser.add_argument('--tDown', default=8, type=int, help="temporal downsampling")
parser.add_argument('--pool_size', default=60, type=int, help="image pool size")
parser.add_argument('--OneBatch', default=False, action='store_true', dest='OneBatch', help='debug with one batch')
# model
# loss
parser.add_argument('--w_idl', default=100, type=float, help='weight for image difference loss')
parser.add_argument('--p_idl', default=1, type=float, help='p-norm for image difference loss')
parser.add_argument('--w_gdl', default=100, type=float, help='weight for gradient difference loss')
parser.add_argument('--p_gdl', default=1, type=float, help='p-norm for gradient difference loss')
parser.add_argument('--w_ggan', default=1, type=float, help='weight for generator loss')
parser.add_argument('--w_dgan', default=1, type=float, help='weight for discriminator loss')
parser.add_argument('--gan_loss', default='GAN', type=str, choices=['GAN', 'LSGAN'], help='which GAN Loss')
parser.add_argument('--nogan', default=False, action='store_true', dest='nogan', help='do not use gan')
# optimizer
parser.add_argument('--solver', default='adam', choices=['adam','sgd'], help='which solver')
parser.add_argument('--MM', default=0.5, type=float, help='momentum')
parser.add_argument('--Beta', default=0.999, type=float, help='beta for adam')
parser.add_argument('--WD', default=1e-4, type=float, help='weight decay')
# learning rate
parser.add_argument('--LRPolicy', default='constant', type=str, choices=['constant', 'step', 'steps', 'exponential',], help='learning rate policy')
parser.add_argument('--gamma', default=1.0, type=float, help='decay rate for learning rate')
parser.add_argument('--LRStart', default=0.0002, type=float, help='initial learning rate')
parser.add_argument('--LRStep', default=10, type=int, help='steps to change learning rate')
parser.add_argument('--LRSteps', default=[], type=int, nargs='+', dest='LRSteps', help='epochs before cutting learning rate')
parser.add_argument('--nEpoch', default=50, type=int, help='total epochs to run')
# init & checkpoint
parser.add_argument('--initModel', default='', help='init model in absence of checkpoint')
parser.add_argument('--initNonAdv', action='store_true', default=False, help='init from a non-adv checkpoint')
parser.add_argument('--checkpoint', default=0, type=int, help='resume from a checkpoint')
# save & display
parser.add_argument('--saveDir', default='results/default/', help='directory to save/log experiments')
parser.add_argument('--saveStep', default=5, type=int, help='epoch step for snapshot')
parser.add_argument('--evalStep', default=10, type=int, help='epoch step for evaluation')
parser.add_argument('--dispIter', default=50, type=int, help='batch step for tensorboard')
# other mode
parser.add_argument('--evalMode', action='store_true', default=False, dest='evalMode', help='evaluation mode')
parser.add_argument('--visMode', action='store_true', default=False, dest='visMode', help='visualization mode')
parser.add_argument('--visDir', default='visual', type=str, help="dir to store visualization")
parser.add_argument('--visNum', default=1000, type=int, help="number of videos to visualize")
parser.add_argument('--applyMode', action='store_true', default=False, dest='applyMode', help='apply model to one gif')
parser.add_argument('--applyFile', default='', type=str, help='path to gif')
# misc
parser.add_argument('--seed', default=1, type=int, help='random seed for torch/numpy')
opts = parser.parse_args()
manual_seed(opts.seed)

def create_model():
    model = edict()
    model.netG = models.UNet(in_ch=3, out_ch=3, ch=64)
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
            scheduler[key] = opts.lr_scheduler.MultiStepLR(op, opts.LRSteps, gamma=opts.gamma)
        elif opts.LRPolicy == 'exponential':
            scheduler[key] = optim.lr_scheduler.ExponentialLR(op, gamma=opts.gamma)
        else:
            raise ValueError('learning rate policy "{}" not defined'.format(opts.LRPolicy))
    return scheduler

def create_dataloader():
    trSet = datasets.Video2Gif_train(inputRoot=opts.inputRoot, subset=opts.subset_train, sCrop=opts.PtSz, sScale=opts.sScale)
    trLD = DD.DataLoader(trSet, batch_size=opts.BtSz,
        sampler= DD.sampler.SubsetRandomSampler([0]*opts.BtSz) if opts.OneBatch else DD.sampler.RandomSampler(trSet),
        num_workers=opts.workers, pin_memory=True, drop_last=True)
    evalSet = datasets.Video2Gif_eval(inputRoot=opts.inputRoot, subset=opts.subset_eval, tStride=opts.tDown, sScale=opts.sScale)
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

def board_vis(epoch, gif, pred, target):
    B, C, H, W = target.shape
    error = (pred - target).abs()
    x = torch.cat((gif, target, pred, error), dim=0)
    x = vutils.make_grid(x, nrow=B, normalize=True)
    opts.board.add_image('train_batch/gif_target_pred_error', x, epoch)

def train(epoch, trLD, model, optimizer, fakeABPool):
    # switch to train mode (Dropout, BatchNorm, etc)
    for key in model.keys():
        model[key].train()

    tags = ['D_gan', 'D_real', 'D_fake', 'D_acc', 'G_gan', 'G_idl', 'G_gdl', 'G_total']
    epL = AverageMeters(tags)
    N = max(1, round(len(trLD) * opts.trRatio))
    for i, samples in progressbar.progressbar(enumerate(islice(trLD, N)), max_value=N):
        # i, samples = 0, next(iter(trLD))
        btSz = samples[0].shape[0]
        realA, realB = list(map(lambda x: preprocess(x).to(DEVICE), samples))
        fakeB = model.netG(realA).tanh()
        D_input = lambda A, B: torch.cat((A, B, pad_tl(diff_xy(B))), dim=1)
        realAB = D_input(realA, realB)
        fakeAB = D_input(realA, fakeB)

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
        g_idl = Lp(opts.p_idl)(fakeB, realB)
        g_gdl = 2 * Lp(opts.p_gdl)(diff_xy(fakeB).abs(), diff_xy(realB).abs())

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
    for key in model.keys():
        model[key].train()

    tags = ['G_idl', 'G_gdl', 'G_total']
    epL = AverageMeters(tags)
    N = max(1, round(len(trLD) * opts.trRatio))
    for i, samples in progressbar.progressbar(enumerate(islice(trLD, N)), max_value=N):
        # i, samples = 0, next(iter(trLD))
        btSz = samples[0].shape[0]
        realA, realB = list(map(lambda x: preprocess(x).to(DEVICE), samples))
        fakeB = model.netG(realA).tanh()

        # Update G network
        optimizer.netG.zero_grad()
        g_idl = Lp(opts.p_idl)(fakeB, realB)
        g_gdl = 2 * Lp(opts.p_gdl)(diff_xy(fakeB).abs(), diff_xy(realB).abs())

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
    for i, (inputs, targets) in progressbar.progressbar(enumerate(evalLD), max_value=len(evalLD)):
        # i, (inputs, targets) = 0, next(iter(evalLD))
        _, T, _, H, W = inputs.size()
        for j in range(T):
            input, target = inputs[:, j], targets[:, j]
            input, target = list(map(lambda x: preprocess(x).to(DEVICE), (input, target)))
            with torch.no_grad():
                pred = netG(input).tanh()
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
    visSet = datasets.Video2Gif_eval(inputRoot=opts.inputRoot, subset=opts.subset_eval, tStride=opts.tDown, sScale=opts.sScale)
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
            inputs, targets = list(map(lambda x: preprocess(x[sbStart:sbEnd+1]).to(DEVICE), samples))
            with torch.no_grad():
                predicts = netG(inputs).tanh()
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
        # Video2Gif
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
        print_options(opts, parser)
        main_train()
        opts.board.close()
