import torch
import torch.nn as nn
from data import *
import argparse
from utils import MAPEvaluator


parser = argparse.ArgumentParser(description='KON-Face Evaluation')
parser.add_argument('-v', '--version', default='centerface',
                    help='centerface.')
parser.add_argument('-d', '--dataset', default='kon',
                    help='kon.')
parser.add_argument('--trained_model', type=str,
                    default='weights/', 
                    help='Trained state_dict file path to open')
parser.add_argument('-size', '--input_size', default=640, type=int,
                    help='input_size')
parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                    help='conf thresh')
parser.add_argument('-nt', '--nms_thresh', default=0.50, type=float,
                    help='nms thresh')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')
parser.add_argument('-nms', '--use_nms', action='store_true', default=False, 
                    help='use diou nms.')

args = parser.parse_args()



def voc_test(model, val_dataset, device):
    evaluator = MAPEvaluator(device=device,
                             dataset=val_dataset,
                             classname=KON_CLASSES,
                             name='kon',
                             display=False
                             )

    # VOC evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':

    # cuda
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # input size
    input_size = [args.input_size, args.input_size]

    # dataset
    # sorry, I don't build a test dataset, so this code is useless for now.
    val_dataset = KONDetection(root=KON_ROOT, 
                               img_size=input_size[0],
                               image_sets='train',
                               transform=BaseTransform(input_size),
                                )

    # build model
    if args.version == 'centerface':
        from models.centerface import CenterFace

        net = CenterFace(device=device, 
                         input_size=input_size, 
                         conf_thresh=args.conf_thresh, 
                         nms_thresh=args.nms_thresh, 
                         num_classes=5,
                         use_nms=args.use_nms
                         )
        print('Let us test CenterFace......')

    else:
        print('Unknown version !!!')
        exit()

    # load net
    net.load_state_dict(torch.load(args.trained_model, map_location='cuda'))
    net.to(device).eval()
    print('Finished loading model!')
    
    # evaluation
    with torch.no_grad():
        voc_test(net, val_dataset, device)
