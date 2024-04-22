import sys

from torch.cuda import amp
import torch.nn.functional as F
import torch

from build_utils.utils import *
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq, accumulate, img_size,
                    grid_min, grid_max, gs,
                    multi_scale=False, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        accumulate = 1

    mloss = torch.zeros(4).to(device)  # mean losses
    now_lr = 0.
    nb = len(data_loader)  # number of batches
    # imgs: [batch_size, 3, img_size, img_size]
    # targets: [num_obj, 6] , that number 6 means -> (img_index, obj_index, x, y, w, h)
    # paths: list of img path
    for i, (imgs, targets, paths, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # ni 统计从epoch0开始的所有batch数
        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        # Multi-Scale
        if multi_scale:
            if ni % accumulate == 0:  # adjust img_size (67% - 150%) every 1 batch
                img_size = random.randrange(grid_min, grid_max + 1) * gs
            sf = img_size / max(imgs.shape[2:])  # scale factor

            if sf != 1:
                # gs: (pixels) grid size
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        with amp.autocast(enabled=scaler is not None):
            pred = model(imgs)

            # loss
            loss_dict = compute_loss(pred, targets, model)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_items = torch.cat((loss_dict_reduced["box_loss"],
                                loss_dict_reduced["obj_loss"],
                                loss_dict_reduced["class_loss"],
                                losses_reduced)).detach()
        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

        if not torch.isfinite(losses_reduced):
            print('WARNING: non-finite loss, ending training ', loss_dict_reduced)
            print("training image path: {}".format(",".join(paths)))
            sys.exit(1)

        losses *= 1. / accumulate  # scale loss

        # backward
        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        # optimize
        if ni % accumulate == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

        if ni % accumulate == 0 and lr_scheduler is not None:
            lr_scheduler.step()

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, coco=None, device=None):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    if coco is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    result_csv = []
    target_csv = []

    for imgs, targets, paths, shapes, img_index in metric_logger.log_every(data_loader, 100, header):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        # targets = targets.to(device)

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        pred = model(imgs)[0]  # only get inference result
        pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6, multi_label=False)
        model_time = time.time() - model_time

        outputs = []
        for index, p in enumerate(pred):
            if p is None:
                p = torch.empty((0, 6), device=cpu_device)
                boxes = torch.empty((0, 4), device=cpu_device)
            else:
                # xmin, ymin, xmax, ymax
                boxes = p[:, :4]
                # shapes: (h0, w0), ((h / h0, w / w0), pad)
                boxes = scale_coords(imgs[index].shape[1:], boxes, shapes[index][0]).round()

            info = {"boxes": boxes.to(cpu_device),
                    "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64),
                    "scores": p[:, 4].to(cpu_device)}
            outputs.append(info)

        res = {img_id: output for img_id, output in zip(img_index, outputs)}

        for t in targets:
            print(t)
            target_data = []
            xmin = round((t[2].item() - t[4].item() / 2.0) * 512, 4)
            ymin = round((t[3].item() - t[5].item() / 2.0) * 512, 4)
            xmax = round((t[2].item() + t[4].item() / 2.0) * 512, 4)
            ymax = round((t[3].item() + t[5].item() / 2.0) * 512, 4)
            target_data.append(int(t[0].item()))
            target_data.append(xmin)
            target_data.append(ymin)
            target_data.append(xmax)
            target_data.append(ymax)
            target_data.append(round(t[2].item() * 512, 4))  # x_center
            target_data.append(round(t[3].item() * 512, 4))  # y_center
            target_data.append(round((t[4].item() * 512 + t[5].item() * 512) / 2.0, 4))  # 直径
            target_data.append(int(t[1].item()))
            target_csv.append(target_data)

        for k in res:
            # print(k)
            for i in range(len(res[k]['boxes'])):
                result_data = []
                xmin = round(res[k]['boxes'][i][0].item(), 4)  # round(x,4)
                ymin = round(res[k]['boxes'][i][1].item(), 4)
                xmax = round(res[k]['boxes'][i][2].item(), 4)
                ymax = round(res[k]['boxes'][i][3].item(), 4)
                result_data.append(k)
                result_data.append(xmin)
                result_data.append(ymin)
                result_data.append(xmax)
                result_data.append(ymax)
                result_data.append(round(xmin + ((xmax - xmin) / 2.0), 4))  # x_center
                result_data.append(round(ymin + ((ymax - ymin) / 2.0), 4))  # y_center
                result_data.append(res[k]['labels'][i].item())
                result_data.append(res[k]['scores'][i].item())
                result_csv.append(result_data)

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)


    print("target_csv:::")
    print(target_csv)
    print("result_csv:::")
    print(result_csv)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    result_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return result_info, target_csv, result_csv


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
