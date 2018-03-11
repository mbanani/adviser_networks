import argparse
import os
import sys
import shutil
import time

import numpy as np
from IPython import embed

import torch

from util                       import tf_logger, Paths
from util                       import get_data_loaders, fg_metrics
from models                     import ch_alexBird, alexBird
from util.torch_utils           import to_var, save_checkpoint
from torch.optim.lr_scheduler   import MultiStepLR

def main(args):
    initialization_time = time.time()


    print "#############  Read in Database   ##############"
    train_loader, valid_loader, test_loader = get_data_loaders( dataset     = args.dataset,
                                                                batch_size  = args.batch_size,
                                                                num_workers = args.num_workers,
                                                                model       = args.model,
                                                                flip        = args.flip,
                                                                num_classes = args.num_classes,
                                                                valid       = 0.0,
                                                                parallel    = args.world_size > 1)

    print "#############  Initiate Model     ##############"
    if args.model == 'alexBird':
        model = alexBird()
        args.no_keypoint = True
    elif args.model == 'ch_alexBird':
        model = ch_alexBird()
        args.no_keypoint = False
    else:
        print "Error: unknown model choice. Exiting."
        exit()

    # Loss
    if args.loss == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
        criterion.cuda()
    else:
        print "Error: unknown loss choice. Exiting."
        exit()

    if args.just_attention:
        assert False, "to do! implement just attention for ch_models"
        params = list(model.map_linear.parameters()) +list(model.cls_linear.parameters())
        params = params + list(model.kp_softmax.parameters()) +list(model.fusion.parameters())
    else:
        params = list(model.parameters())

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr = args.lr, betas = (0.9, 0.999), eps=1e-8, weight_decay=0)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum = 0.9, weight_decay = 0.0005)
        scheduler = MultiStepLR( optimizer,
                                 milestones=range(0, args.num_epochs, 5),
                                 gamma=0.95)
    else:
        assert False, "Error: Unknown choice for optimizer."


    if args.resume is not None:
        print "Loading pretrained Module at %s " % (args.resume)
        checkpoint      = torch.load(args.resume)
        args.best_loss  = checkpoint['val_loss']
        args.best_acc   = checkpoint['val_acc']
        start_epoch     = checkpoint['epoch']
        start_step      = checkpoint['step']
        state_dict      = checkpoint['state_dict']

        print "Pretrained Model Val Accuracy is %f " % (args.best_acc)
        model.load_state_dict(state_dict)
    else:
        start_epoch     = 0
        start_step      = 0

    # if args.world_size > 1:
    #     print "Parallelizing Model"
    #     if torch.cuda.is_available():
    #         model = torch.nn.DataParallel(model, device_ids = range(0, args.world_size)).cuda()

    if torch.cuda.is_available():
        # Train on GPU if available
        model.cuda()


    print "Time to initialize take: ", time.time() - initialization_time
    print "#############  Start Training     ##############"
    total_step = len(train_loader)

    for epoch in range(0, args.num_epochs):

        if epoch % args.eval_epoch == 0:
            if args.evaluate_train:
                _, _ = eval_step(   model       = model,
                                    data_loader = train_loader,
                                    criterion   = criterion,
                                    step        = epoch * total_step,
                                    datasplit   = "train",
                                    with_dropout = True)


            curr_loss, curr_wacc = eval_step(   model       = model,
                                                data_loader = test_loader,
                                                criterion   = criterion,
                                                step        = epoch * total_step,
                                                datasplit   = "test")

            if valid_loader != None:
                assert False, "proper sampling for validation vs training not implemented yet!"
                curr_loss, curr_wacc = eval_step(   model       = model,
                                                    data_loader = valid_loader,
                                                    criterion   = criterion,
                                                    step        = epoch * total_step,
                                                    datasplit   = "valid")



        if args.evaluate_only:
            exit()

        if epoch % args.save_epoch == 0 and epoch > 0:

            args = save_checkpoint(  model      = model,
                                     optimizer  = optimizer,
                                     curr_epoch = epoch,
                                     curr_step  = (total_step * epoch),
                                     args       = args,
                                     curr_loss  = curr_loss,
                                     curr_acc   = curr_wacc,
                                     filename   = ('model@epoch%d.pkl' %(epoch)))

        if args.optimizer == 'sgd':
            scheduler.step()

        logger.add_scalar_value("Misc/Epoch Number", epoch, step=epoch * total_step)
        train_step( model        = model,
                    train_loader = train_loader,
                    criterion    = criterion,
                    optimizer    = optimizer,
                    epoch        = epoch,
                    step         = epoch * total_step,
                    valid_loader = valid_loader,
                    valid_type   = "valid")

    # Final save of the model
    args = save_checkpoint( model      = model,
                            optimizer  = optimizer,
                            curr_epoch = epoch,
                            curr_step  = (total_step * epoch),
                            args       = args,
                            curr_loss  = curr_loss,
                            curr_acc   = curr_wacc,
                            filename   = ('model@epoch%d.pkl' %(epoch)))

def train_step(model, train_loader, criterion, optimizer, epoch, step, valid_loader = None, valid_type = "valid"):
    model.train()
    total_step      = len(train_loader)
    epoch_time      = time.time()
    batch_time      = time.time()
    processing_time = 0
    loss_sum        = 0.
    counter         = 0

    for i, (images, labels, kp_map, kp_class, key_uid) in enumerate(train_loader):
        counter = counter + 1
        training_time = time.time()

        # Set mini-batch dataset
        images      = to_var(images, volatile=False)
        labels      = to_var(labels)

        if (not args.no_keypoint):
            kp_map      = to_var(kp_map, volatile=False)
            kp_class    = to_var(kp_class, volatile=False)

        # Forward, Backward and Optimize
        model.zero_grad()

        if args.no_keypoint:
            preds = model(images)
        else:
            preds = model(images, kp_map, kp_class)

        loss = criterion(preds, labels)

        curr_loss = loss.data.cpu().item()
        loss_sum += curr_loss

        loss.backward()
        optimizer.step()

        # # Log losses
        logger.add_scalar_value("(" + args.dataset + ") Loss/train", curr_loss , step=step + i)

        processing_time += time.time() - training_time

        # Print log info
        if i % args.log_rate == 0 and i > 0:
            # print "Epoch [%d/%d] Step [%d/%d]" %( epoch, args.num_epochs, i, total_step)
            time_diff = time.time() - batch_time

            curr_batch_time = time_diff / (1.*args.log_rate)
            curr_train_per  = processing_time/time_diff
            curr_epoch_time = (time.time() - epoch_time) * (total_step / (i+1.))
            curr_time_left  = (time.time() - epoch_time) * ((total_step - i) / (i+1.))

            print "Epoch [%d/%d] Step [%d/%d]: Training Loss = %2.5f, Batch Time = %.2f sec, Time Left = %.1f mins." %( epoch, args.num_epochs,
                                                                                                                        i, total_step,
                                                                                                                        loss_sum / float(counter),
                                                                                                                        curr_batch_time,
                                                                                                                        curr_time_left / 60.)


            logger.add_scalar_value("Misc/batch time (s)",    curr_batch_time,        step=step + i)
            logger.add_scalar_value("Misc/Train_%",           curr_train_per,         step=step + i)
            logger.add_scalar_value("Misc/epoch time (min)",  curr_epoch_time / 60.,  step=step + i)
            logger.add_scalar_value("Misc/time left (min)",   curr_time_left / 60.,   step=step + i)

            # Reset counters
            counter = 0
            loss_sum = 0.
            processing_time = 0
            batch_time = time.time()

        if valid_loader != None and i % args.eval_step == 0 and i > 0:
            model.eval()
            _, _ = eval_step(   model       = model,
                                data_loader = valid_loader,
                                criterion   = criterion,
                                step        = epoch * total_step,
                                datasplit   = valid_type)

            model.train()


def eval_step( model, data_loader,  criterion, step, datasplit, with_dropout = False):
    if not with_dropout:
        model.eval()

    total_step      = len(data_loader)
    start_time      = time.time()
    epoch_loss      = 0.
    results_dict    = fg_metrics(args.num_classes)

    for i, (images, labels, kp_map, kp_class, key_uid) in enumerate(data_loader):

        if i % args.log_rate == 0:
            print "Evaluation of %s [%d/%d] Time Elapsed: %f " % (datasplit, i, total_step, time.time() - start_time)

        images = to_var(images, volatile=True)
        labels = to_var(labels, volatile=True)


        if args.no_keypoint:
            preds = model(images)
        else:
            preds = model(images, kp_map, kp_class)

        epoch_loss   += criterion(preds, labels).data.cpu().item()

        # Labels not needed as class can be found in uid
        results_dict.update_dict( key_uid, preds.data.cpu().numpy())


    accuracy1, accuracy5, accuracy10, total_count = results_dict.metrics()
    epoch_loss = float(epoch_loss)

    acc1    = np.sum(accuracy1  * total_count) / np.sum(total_count) * 100
    acc5    = np.sum(accuracy5  * total_count) / np.sum(total_count) * 100
    acc10   = np.sum(accuracy10 * total_count) / np.sum(total_count) * 100

    wacc1   = np.mean(accuracy1 ) * 100
    wacc5   = np.mean(accuracy5 ) * 100
    wacc10  = np.mean(accuracy10) * 100


    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "Accuracy@1       : ", acc1 , " %"
    print "Accuracy@5       : ", acc5 , " %"
    print "Accuracy@10      : ", acc10, " %"
    print ""
    print "W. Accuracy@1    : ", wacc1 , " %"
    print "W. Accuracy@5    : ", wacc5 , " %"
    print "W. Accuracy@10   : ", wacc10, " %"
    print "Loss         : ", epoch_loss
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    logger.add_scalar_value("(" + args.dataset + ") Loss/"              + datasplit, epoch_loss, step=step + i)
    logger.add_scalar_value("(" + args.dataset + ") Accuracy@1/"        + datasplit, acc1,   step=step + i)
    logger.add_scalar_value("(" + args.dataset + ") Accuracy@5/"        + datasplit, acc5,   step=step + i)
    logger.add_scalar_value("(" + args.dataset + ") Accuracy@10/"       + datasplit, acc10,  step=step + i)
    logger.add_scalar_value("(" + args.dataset + ") W. Accuracy@1/"     + datasplit, wacc1,  step=step + i)
    logger.add_scalar_value("(" + args.dataset + ") W. Accuracy@5/"     + datasplit, wacc5,  step=step + i)
    logger.add_scalar_value("(" + args.dataset + ") W. Accuracy@10/"    + datasplit, wacc10, step=step + i)

    assert type(epoch_loss) == float, 'Error: Loss type is not float'
    return epoch_loss, acc10



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # logging parameters
    parser.add_argument('--save_epoch',     type=int , default=10)
    parser.add_argument('--eval_epoch',     type=int , default=1)
    parser.add_argument('--eval_step',      type=int , default=1000)
    parser.add_argument('--log_rate',       type=int, default=10)
    parser.add_argument('--num_workers',    type=int, default=7)

    # training parameters
    parser.add_argument('--num_epochs',     type=int, default=100)
    parser.add_argument('--batch_size',     type=int, default=64)
    parser.add_argument('--lr',             type=float, default=0.01)
    parser.add_argument('--optimizer',      type=str,default='sgd')
    parser.add_argument('--loss',           type=str, default='cross_entropy')

    # experiment details
    parser.add_argument('--dataset',         type=str, default='birdsnapKP')
    parser.add_argument('--model',           type=str, default='alexBird')
    parser.add_argument('--experiment_name', type=str, default= 'Test')
    parser.add_argument('--evaluate_only',   action="store_true",default=False)
    parser.add_argument('--evaluate_train',  action="store_true",default=False)
    parser.add_argument('--flip',            action="store_true",default=False)
    parser.add_argument('--just_attention',  action="store_true",default=False)
    parser.add_argument('--num_classes',     type=int, default=500)
    parser.add_argument('--resume',          type=str, default=None)
    parser.add_argument('--world_size',      type=int, default=1)

    args = parser.parse_args()

    root_dir                    = os.path.dirname(os.path.abspath(__file__))
    experiment_result_dir       = os.path.join(root_dir, os.path.join('experiments',args.dataset))
    args.full_experiment_name   = ("exp_%s_%s_%s" % ( time.strftime("%m_%d_%H_%M_%S"), args.dataset, args.experiment_name) )
    args.experiment_path        = os.path.join(experiment_result_dir, args.full_experiment_name)
    args.best_loss              = sys.float_info.max
    args.best_acc               = 0.


    # Create model directory
    if not os.path.exists(experiment_result_dir):
        os.makedirs(experiment_result_dir)
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)


    # Define Logger
    log_name    = args.full_experiment_name
    logger      = tf_logger(os.path.join(Paths.tensorboard_logdir, log_name))

    main(args)
