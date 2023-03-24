import time, os, signal

import torch as tc
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, RandomSampler
import torch.utils.data.distributed as dd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import detect_anomaly, grad

from src.data.factory import fetch_dataset, DATASETS
from src.model.factory import *
from src.data.dataset import TinyImages
from src.utils.helper import *
from src.utils.evaluate import validate
from src.utils.printer import sprint, dprint
from src.utils.adversary import pgd, perturb
from src.utils.awp import AdvWeightPerturb as AWP

def train(args):
    start_epoch = 0
    best_acc1, best_pgd, best_fgsm = 0, 0, 0
    checkpoint = None
    
    fargs = args.func_arguments(fetch_dataset, DATASETS, postfix='data')
    train_set = fetch_dataset(train=True, **fargs)
    train_sampler = dd.DistributedSampler(train_set) if args.parallel else None
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              pin_memory=True,
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              drop_last=True)

    if args.rst:
        edata_set = TinyImages(fargs.root, train_set.transform, train_set.target_transform)
        edata_sampler = dd.DistributedSampler(edata_set) if args.parallel else None
        edata_loader = DataLoader(edata_set,
                                  batch_size=args.batch_size,
                                  shuffle=(train_sampler is None),
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  sampler=edata_sampler,
                                  drop_last=True)
    else:
        edata_loader = None
    
    val_set = fetch_dataset(train=False, split=args.world_size, **fargs)
    if args.world_size > 1:
        total_samples = sum([len(vs) for vs in val_set])
        val_set = val_set[args.rank]
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)

    
    if args.resume is not None:
        resume_file = args.path('trained', "{}/{}_end".format(args.logbook, args.resume))
        if os.path.isfile(resume_file):
            checkpoint = tc.load(resume_file, map_location='cpu')
            best_acc1 = checkpoint['best_acc1']
            best_fgsm = checkpoint['best_fgsm']
            best_pgd = checkpoint['best_pgd']
            start_epoch = checkpoint['epoch']
        else:
            raise Exception("Resume point not exists: {}".format(args.resume))
        checkpoint = (args.resume, checkpoint)
        
    fargs = args.func_arguments(fetch_model, ARCHS, postfix='arch')
    if checkpoint is not None:
        fargs['checkpoint'] = checkpoint
    model = fetch_model(**fargs).to(args.device)
    if args.parallel:
        if args.using_cpu():
            model = DDP(model)
        else:
            model = DDP(model, device_ids=[args.rank], output_device=args.rank)

    if len(args.awp_gamma) != 0:
        tmp_model = fetch_model(**fargs).to(args.device)
        tmp_model.train()
        awp_proxy = fetch_model(**fargs).to(args.device)
        awp_opt = tc.optim.SGD(awp_proxy.parameters(), lr=args.awp_eta)
        awp = AWP(model=model, proxy=awp_proxy, proxy_optim=awp_opt, gamma=args.awp_gamma)
    else:
        awp = None
        tmp_model = None
        tmp_opt = None
        
    criterion = nn.CrossEntropyLoss()

    fargs = args.func_arguments(fetch_optimizer, OPTIMS, checkpoint=checkpoint)
    optimizer = fetch_optimizer(params=model.parameters(), **fargs)

    if tmp_model is not None:
        tmp_opt = fetch_optimizer(params=tmp_model.parameters(), **fargs)
    
    # free the memory taken by the checkpoint
    checkpoint = None

    
    if args.advt:
        dprint('adversary', **{k:getattr(args, k, None)
                               for k in ['eps', 'eps_step', 'max_iter', 'random_init', 'eval_iter']})
    dprint('data loader', batch_size=args.batch_size, num_workers=args.num_workers)
    sprint("=> Start training!", split=True)

    for epoch in range(start_epoch, args.epochs):
        if args.parallel:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr, args.annealing)

        update(train_loader, model, criterion, optimizer, epoch, args, awp, edata_loader, tmp_model, tmp_opt)
        
        model.eval()
        acc1, ig, fgsm, pgd = validate(val_loader, model, criterion, args)
        
        if args.rank is not None and args.rank != 0: continue
        # execute only on the main process
        
        best_acc1 = max(acc1, best_acc1)
        best_fgsm = max(fgsm, best_fgsm)
        best_pgd = max(pgd, best_pgd)

        print(" ** Acc@1: {:.2f} | FGSM: {:.2f} | PGD: {:.2f}".format(best_acc1, best_fgsm, best_pgd))
        
        if args.logging:
            logger = args.logger
            acc_info = '{:3.2f} E: {} IG: {:.2e} FGSM: {:3.2f} PGD: {:3.2f}'.format(acc1, epoch+1, ig, fgsm, pgd)
            logger.update('checkpoint', end=acc_info)
            state_dict = model.module.state_dict() if args.parallel else model.state_dict()
            state = {
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'best_acc1': best_acc1,
                'best_pgd': best_pgd,
                'best_fgsm' : best_fgsm,
                'optimizer' : optimizer.state_dict(),
            }

            if acc1 >= best_acc1:
                logger.update('checkpoint', acc=acc_info, save=True)

            lid = args.log_id
            fname = "{}/{}".format(args.logbook, lid)
            ck_path = args.path('trained', fname+"_end")
            tc.save(state, ck_path)

            if pgd >= best_pgd:
                logger.update('checkpoint', pgd=acc_info, save=True)
                shutil.copyfile(ck_path, args.path('trained', fname+'_pgd'))

            
def update(loader, model, criterion, optimizer, epoch, args, awp, edata_loader, tmp_model, tmp_opt):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    igs = AverageMeter('IG', ':.2e')
    accs = AverageMeter('Acc@', ':6.2f')
    robs = AverageMeter('Rob@', ':6.2f')
    meters = [batch_time, losses, igs, accs, robs]
    progress = ProgressMeter(len(loader), meters, prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    if edata_loader is not None:
        edata_loader = iter(edata_loader)

    # calc the stage of learning rate decay in piecewise schedule
    lr_stage = 0
    for a in args.annealing:
        if epoch < int(a): break
        else: lr_stage += 1

    if len(args.lam) == 0:
        lam = 0
    else:
        # get the value of lam according to learning rate stage
        lam = args.lam[lr_stage] if lr_stage < len(args.lam) else args.lam[-1]

    if len(args.awp_gamma) == 0:
        gamma = 0
    else:
        # get the value of awp gamma according to learning rate stage
        gamma = args.awp_gamma[lr_stage] if lr_stage < len(args.awp_gamma) else args.awp_gamma[-1]
        
    if args.warm_start and epoch < 5:
        # warmup training: linearly increase epsilon and step size
        # disable smoothness regularization
        factor = epoch / 5
        eps = args.eps * factor
        step = args.eps_step * factor

        lam = 0
        awp_enabled = False
    else:
        eps, step = args.eps, args.eps_step
        awp_enabled = awp is not None and gamma != 0
        if awp is not None:
            awp.gamma = gamma
        
    end = time.time()
    niter = len(loader)
    for i, (aug, tgt) in enumerate(loader, 1):
        if edata_loader is not None:
            # load extra data if semi-supervised learning (RST) enabled
            eaug, etgt = next(edata_loader)
            aug = tc.cat((aug, eaug))
            tgt = tc.cat((tgt, etgt))
            
        aug = aug.to(args.device, non_blocking=True)
        tgt = tgt.to(args.device, non_blocking=True)        
        prt = tc.zeros_like(aug)
        
        if args.random_init: prt.uniform_(-eps, eps)
        
        batch_size = len(aug)

        # forward pass on clean examples
        cle = tc.clamp(aug+prt, 0, 1).requires_grad_(True)
        prt = cle - aug
        lgt_cle = model(cle)

        if awp_enabled:
            # clone the underlying model into tmp_model
            tmp_model.load_state_dict(model.state_dict())

        # calc clean accuracy
        acc = accuracy(lgt_cle, tgt)[0]
        accs.update(acc, batch_size)

        # instance-wise CE loss on clean examples
        loss_cle = cross_entropy(lgt_cle, tgt, reduction='none')        
        loss = tc.mean(loss_cle)

        # gradients of clean loss w.r.t. the input
        # retain the computation graph if smothness reg enabled yet AWP not
        # because the clean logits will not be recomputed using tmp_model
        ig = grad(loss, cle, retain_graph=lam!=0 and not awp_enabled)[0]
        ig_norm = tc.norm(ig, p=1)
        
        if args.advt:
            adv, prt = pgd(aug, tgt, model, criterion,
                           eps, step, args.max_iter, pert=prt, ig=ig)
                        
            if awp_enabled:
                # calc adversarial weight perturbation
                awp_prt = awp.calc_awp(inputs_adv=adv, targets=tgt)
                # perturb the underlying model
                awp.perturb(awp_prt)
                
                lgt_adv = model(adv)
                # instance-wise CE loss on adversarial examples
                loss_adv = cross_entropy(lgt_adv, tgt, reduction='none')
            else:
                awp_prt = None
                
                lgt_adv = model(adv)
                loss_adv = cross_entropy(lgt_adv, tgt, reduction='none')
                
            rob = accuracy(lgt_adv, tgt)[0]
            robs.update(rob, batch_size)
            loss = tc.mean(loss_adv)
                
            if lam != 0:
                # apply smoothness regularization
                if awp_prt is not None:
                    # recompute logits on clean examples using tmp_model
                    # which is exactly the same model as the unperturbed underlying one
                    # the original clean logits cannot be used because
                    # the underlying is perturbed, i.e., parameters modified in-place
                    lgt_cle = tmp_model(cle)

                # instance-wise logit variation between clean and adv examples
                smooth = (lgt_adv-lgt_cle)**2
                smooth = smooth.sum(dim=1)

                # instance-wise adversarial vulnerability
                av = (loss_adv - loss_cle).detach()
                # index of sorted av
                _, topk_idx = tc.topk(av, av.size(0), largest=False)
                # linear weight
                w = tc.tensor(range(1, av.size(0)+1)).float().cuda()
                w /= av.size(0)
                w[topk_idx] = w.clone()

                smooth *= w.detach()
                
                loss += tc.sum(smooth) * lam / batch_size
                                
        # measure accuracy and record loss
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        if awp_prt is not None and lam != 0:
            tmp_opt.zero_grad()
        loss.backward()

        if awp_prt is not None and lam != 0:
            for p1, p2 in zip(model.parameters(), tmp_model.parameters()):
                # merge gradients from tmp_model into the underlying one
                p1.grad += p2.grad

        optimizer.step()

        if awp_prt is not None:
            # remove awp from the underlying model
            awp.restore(awp_prt)
            
        igs.update(ig_norm, batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i == 1 or i % args.log_pbtc == 0 or i == niter:
            progress.display(i)
        
        if args.rank != 0: continue

def adjust_learning_rate(optimizer, epoch, lr, annealing):
    decay = 0
    for a in annealing:
        if epoch < int(a): break
        else: decay += 1
    
    lr *= 0.1 ** decay

    params = optimizer.param_groups
    if lr != params[0]['lr']:
        sprint("Learning rate now is {:.0e}".format(lr))
    
    for param in params: param['lr'] = lr
