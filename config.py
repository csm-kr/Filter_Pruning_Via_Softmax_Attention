import torch
import argparse
import yaml
import sys

device_ids = [0]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')


def parser(args):
    # 1. argparse
    parser = argparse.ArgumentParser(args)
    parser.add_argument('--config_path', type=str, default='./yaml_files/mnist.yaml')   # FIXME

    yaml_path = './yaml_files/mnist.yaml'
    # yaml_path = './yaml_files/cifar10.yaml.yaml'
    # yaml_path = './yaml_files/fashionmnist.yaml'

    with open(yaml_path) as config_file:
        config = yaml.safe_load(config_file)
        print(config)

        training_parameters = config['training_parameters']
        milestone = training_parameters['milestone']

        parser.add_argument('--milestone_0', type=int, default=milestone[0])
        parser.add_argument('--milestone_1', type=int, default=milestone[1])
        parser.add_argument('--epoch', type=int, default=training_parameters['epoch'])
        parser.add_argument('--lr', type=float, default=training_parameters['lr'])
        parser.add_argument('--n_classes', type=int, default=training_parameters['n_classes'])
        parser.add_argument('--dataset_type', type=str, default=training_parameters['dataset'])
        parser.add_argument('--batch_size', type=str, default=training_parameters['batch_size'])
        parser.add_argument('--model_name', type=str, default=training_parameters['model'])

    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='finetuned_pruned')
    parser.add_argument('--pruned_save_file_name', type=str, default='pruned')
    parser.add_argument('--pruning_ratio', type=float, default=0.875)

    opts = parser.parse_args()
    print(opts)
    return opts


if __name__ == '__main__':
    parser(sys.argv[1:])