import yaml
from datetime import datetime
import torch
import os
# import wandb
import argparse
from torchvision import datasets, transforms
import copy
from PIL import Image
from model import Classifier, OneHotCVAE
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pathlib

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

def get_config_and_setup_dirs(filename):
    with open(filename, 'r') as fp:
        config = yaml.safe_load(fp)
    config = dict2namespace(config)
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join("./results", config.dataset.lower(), timestamp)
    config.log_dir = os.path.join(config.exp_root_dir, 'logs')
    config.ckpt_dir = os.path.join(config.exp_root_dir, 'ckpts')
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    # wandb_id = wandb.util.generate_id()
    # config.wandb_id = wandb_id

    with open(os.path.join(config.exp_root_dir, 'config.yaml'), 'w') as fp:
        yaml.dump(config, fp)
    
    return config

def get_config_and_setup_dirs_final(working_dir):
    with open(working_dir, 'r') as fp:
        config = yaml.safe_load(fp)
    config = dict2namespace(config)
    
    # timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join(f"./{config.working_dir}", config.dataset.lower(), "initial")
    config.log_dir = os.path.join(config.exp_root_dir, 'logs')
    config.ckpt_dir = os.path.join(config.exp_root_dir, 'ckpts')
    if not os.path.exists(config.exp_root_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)

    # wandb_id = wandb.util.generate_id()
    # config.wandb_id = wandb_id

    with open(os.path.join(config.exp_root_dir, f'config_initial.yaml'), 'w') as fp:
        yaml.dump(config, fp)
    
    return config

def setup_dirs_final(config, working_dir, filename):
    
    # timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join(f"./{working_dir}", config.dataset.lower(), filename)
    config.log_dir = os.path.join(config.exp_root_dir, 'logs')
    config.ckpt_dir = os.path.join(config.exp_root_dir, 'ckpts')
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    # wandb_id = wandb.util.generate_id()
    # config.wandb_id = wandb_id

    with open(os.path.join(config.exp_root_dir, f'config_{filename}.yaml'), 'w') as fp:
        yaml.dump(config, fp)
    
    return config

def setup_dirs(config):
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join("./results", config.dataset.lower(), timestamp)
    config.log_dir = os.path.join(config.exp_root_dir, 'logs')
    config.ckpt_dir = os.path.join(config.exp_root_dir, 'ckpts')
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    # wandb_id = wandb.util.generate_id()
    # config.wandb_id = wandb_id

    with open(os.path.join(config.exp_root_dir, 'config.yaml'), 'w') as fp:
        yaml.dump(config, fp)
    
    return config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def cycle(dl):
    while True:
        for data in dl:
            yield data

def find_indices_to_drop(model_state_dict, layer_name, hyperparam_k = 0.1): # hardcoded for now ... , may write comprehensive code for this later!
    if 'fc31' in layer_name or 'fc32' in layer_name or 'fc6' in layer_name:
        return []
    
    # find the L2 norm of each neuron and store it
    l2_norm = []
    for i in range(model_state_dict[layer_name + '.weight'].size()[0]):
        l2_norm.append(torch.norm(model_state_dict[layer_name + '.weight'][i], 2))
        # Add the contribution from layer_name.bias
        l2_norm[-1] += torch.norm(model_state_dict[layer_name + ".bias"][i], 2)

    # find the threshold i.e. the value of the bottom k% of L2 norm
    if int(hyperparam_k * len(l2_norm)) == 0:
        return []
    else:
        threshold = torch.kthvalue(torch.tensor(l2_norm), int(hyperparam_k * len(l2_norm)))[0]
    # find the indices of neurons with L2 norm less than threshold
    indices = []
    for i in range(len(l2_norm)):
        if l2_norm[i] < threshold:
            indices.append(i)
    return indices

def create_dag(model_state_dict, type=1): # harcoded for now ... , may write comprehensive code for this later!
    dag = {
        "fc1.0.weight": ["fc2.0.weight"],
        "fc2.0.weight": ["fc31.0.weight", "fc32.0.weight"],
        "fc31.0.weight": ["fc4.0.weight"],
        "fc32.0.weight": ["fc4.0.weight"],
        "fc4.0.weight": ["fc5.0.weight"],
        "fc5.0.weight": ["fc6.0.weight"],
    }

    dag2 = {
        "fc1.weight": ["fc2.weight"],
        "fc2.weight": ["fc31.weight", "fc32.weight"],
        "fc31.weight": ["fc4.weight"],
        "fc32.weight": ["fc4.weight"],
        "fc4.weight": ["fc5.weight"],
        "fc5.weight": ["fc6.weight"],
    }
    if type == 1:
        return dag
    return dag2


def prune_model(model_state_dict):
    last_indices = []
    for layer_name in list(model_state_dict.keys()):  # Convert to list to make a static copy of keys
        prune_left = True
        prune_right = True
        print("layer_name ", layer_name)
        if 'fc' in layer_name and 'weight' in layer_name:
            attribute_name = layer_name.split('.')[0]
            base_layer_name = layer_name.rsplit('.', 1)[0]
            indices_to_drop = find_indices_to_drop(model_state_dict, base_layer_name)
            print(f"Pruning {len(indices_to_drop)} neurons from {base_layer_name}")

            device = model_state_dict[base_layer_name + '.weight'].device

            # Creating a mask for all indices initially set to keep (True)
            layer_shape = model_state_dict[base_layer_name + '.weight'].shape
            print("layer_shape ", layer_shape)
            if layer_shape[1] == 794:
                prune_right = False
            if layer_shape[0] == 784:
                prune_left = False

            if prune_left:
                print("inside prune left")
                total_indices = model_state_dict[base_layer_name + '.weight'].shape[0]
                mask = torch.ones(total_indices, dtype=torch.bool, device=device)
                mask[indices_to_drop] = False  # Setting indices to drop to False

                # Applying mask to prune weights
                pruned_weight = model_state_dict[base_layer_name + '.weight'][mask]
                model_state_dict[base_layer_name + '.weight'] = pruned_weight

                # Checking and applying mask to prune bias if it exists
                if base_layer_name + '.bias' in model_state_dict:
                    pruned_bias = model_state_dict[base_layer_name + '.bias'][mask]
                    model_state_dict[base_layer_name + '.bias'] = pruned_bias

                print("Pruned weight shape: ", pruned_weight.shape)
                if base_layer_name + '.bias' in model_state_dict:
                    print("Pruned bias shape: ", pruned_bias.shape)
            if prune_right and last_indices[-1]['total_indices'] == model_state_dict[base_layer_name + '.weight'].shape[1]:
                print("inside prune right")
                total_indices = model_state_dict[base_layer_name + '.weight'].shape[1]
                mask = torch.ones(total_indices, dtype=torch.bool, device=device)
                mask[last_indices[-1]['indices_to_drop']] = False

                # Applying mask to prune weights
                pruned_weight = model_state_dict[base_layer_name + '.weight'][:, mask]
                model_state_dict[base_layer_name + '.weight'] = pruned_weight

                print("Pruned weight shape: ", pruned_weight.shape)
            
            last_indices.append({'total_indices': total_indices, 'indices_to_drop': indices_to_drop})
        # print the final shape of all the layers in the model
    
    #change all the layer names to remove the .0
    for key in list(model_state_dict.keys()):
        if '.0.' in key:
            new_key = key.replace('.0.', '.')
            model_state_dict[new_key] = model_state_dict.pop(key)

    for layer_name in model_state_dict:
        print(f"Layer {layer_name} shape: {model_state_dict[layer_name].shape}")
        
    return model_state_dict


def prune_model_using_dag(model_state_dict, hyperparam_k = 0.1, type=1):
    model_state_dict_copy = copy.deepcopy(model_state_dict)
    dag = create_dag(model_state_dict, type=type)
    if type == 1:
        start_layer = "fc1.0.weight"
    else :
        start_layer = "fc1.weight"
    layers_done = set([])
    if type == 1:
        layers_to_prune = list(dag.keys()) + ["fc6.0.weight"]
    else :
        layers_to_prune = list(dag.keys()) + ["fc6.weight"]
    while len(layers_to_prune) != 0:
        # get the first in the list
        start_layer = layers_to_prune.pop(0)
        prune_right = False
        print("start_layer ", start_layer)
        attribute_name = start_layer.split('.')[0]
        base_layer_name = start_layer.rsplit('.', 1)[0]

        device = model_state_dict[base_layer_name + '.weight'].device

        layer_shape = model_state_dict[base_layer_name + '.weight'].shape
        print("layer_shape ", layer_shape)
        # check if this layer has any parent in the dag
        parent_layer = None
        for key in dag:
            if start_layer in dag[key]:
                parent_layer = key
                break
        if parent_layer is not None:
            if parent_layer in layers_done:
                indices_to_drop2 = find_indices_to_drop(model_state_dict_copy, parent_layer.rsplit('.', 1)[0], hyperparam_k)
                prune_right = True
            else:
                prune_right = False
        
        indices_to_drop = find_indices_to_drop(model_state_dict, base_layer_name, hyperparam_k)
        print(f"Pruning {len(indices_to_drop)} neurons from {base_layer_name}")

        # drop the first dimension
        if len(indices_to_drop) != 0:
            if base_layer_name + '.weight' in model_state_dict:
                total_indices = model_state_dict[base_layer_name + '.weight'].shape[0]
                mask = torch.ones(total_indices, dtype=torch.bool, device=device)
                mask[indices_to_drop] = False

                pruned_weight = model_state_dict[base_layer_name + '.weight'][mask]
                model_state_dict[base_layer_name + '.weight'] = pruned_weight

                if base_layer_name + '.bias' in model_state_dict:
                    pruned_bias = model_state_dict[base_layer_name + '.bias'][mask]
                    model_state_dict[base_layer_name + '.bias'] = pruned_bias

                print("Pruned weight shape: ", pruned_weight.shape)
                if base_layer_name + '.bias' in model_state_dict:
                    print("Pruned bias shape: ", pruned_bias.shape)
        
        # drop the second dimension
        if prune_right:
            print("inside prune right")
            print("parent_layer ", parent_layer)
            if base_layer_name + '.weight' in model_state_dict:
                total_indices = model_state_dict[base_layer_name + '.weight'].shape[1]
                mask = torch.ones(total_indices, dtype=torch.bool, device=device)
                mask[indices_to_drop2] = False

                pruned_weight = model_state_dict[base_layer_name + '.weight'][:, mask]
                model_state_dict[base_layer_name + '.weight'] = pruned_weight

                print("Pruned weight shape: ", pruned_weight.shape)
        layers_done.add(start_layer)
    
    for key in list(model_state_dict.keys()):
        if '.0.' in key:
            new_key = key.replace('.0.', '.')
            model_state_dict[new_key] = model_state_dict.pop(key)
    # print the final shape of all the layers in the model
    for layer_name in model_state_dict:
        print(f"Layer {layer_name} shape: {model_state_dict[layer_name].shape}")

    return model_state_dict

def expand_model(model_state_dict, hyperparam_e = 0.1, hyperparam_perturbation=0.01, type=2): # hardcoded for now ... , may write comprehensive code for this later!
    model_state_dict_copy = copy.deepcopy(model_state_dict)
    dag = create_dag(model_state_dict, type=type)
    if type == 2:
        start_layer = "fc1.weight"
    else : 
        start_layer = "fc1.0.weight"
    layers_done = set([])
    if type == 2:
        layers_to_expand = list(dag.keys()) + ["fc6.weight"]
    else : 
        layers_to_expand = list(dag.keys()) + ["fc6.0.weight"]
    while len(layers_to_expand) != 0:
        start_layer = layers_to_expand.pop(0)
        expand_right = False
        print("start_layer ", start_layer)
        attribute_name = start_layer.split('.')[0]
        base_layer_name = start_layer.rsplit('.', 1)[0]

        device = model_state_dict[base_layer_name + '.weight'].device

        layer_shape = model_state_dict[base_layer_name + '.weight'].shape
        print("layer_shape ", layer_shape)
        # check if this layer has any parent in the dag
        parent_layer = None
        for key in dag:
            if start_layer in dag[key]:
                parent_layer = key
                break
        if parent_layer is not None:
            if parent_layer in layers_done:
                amount_to_expand2 = model_state_dict_copy[parent_layer.rsplit('.', 1)[0] + '.weight'].shape[0]*hyperparam_e
                print("amount_to_expand2 ", amount_to_expand2)
                expand_right = True
            else:
                expand_right = False
        
        amount_to_expand = model_state_dict_copy[base_layer_name + '.weight'].shape[0]*hyperparam_e
        print(f"Expanding {amount_to_expand} neurons from {base_layer_name}")

        # expand the first dimension
        if int(amount_to_expand) != 0 and base_layer_name != 'fc6' and base_layer_name != 'fc31' and base_layer_name != 'fc32':
            if base_layer_name + '.weight' in model_state_dict:
                num_new_neurons = int(amount_to_expand)
                new_weights = (torch.randn(num_new_neurons, model_state_dict[base_layer_name + '.weight'].shape[1])*hyperparam_perturbation).to(device)
                expanded_weights = torch.cat((model_state_dict[base_layer_name + '.weight'], new_weights), dim=0)
                model_state_dict[base_layer_name + '.weight'] = expanded_weights

                if base_layer_name + '.bias' in model_state_dict:
                    new_bias = (torch.randn(num_new_neurons)*hyperparam_perturbation).to(device)
                    expanded_bias = torch.cat((model_state_dict[base_layer_name + '.bias'], new_bias), dim=0)
                    model_state_dict[base_layer_name + '.bias'] = expanded_bias

                print("Expanded weight shape: ", expanded_weights.shape)
                if base_layer_name + '.bias' in model_state_dict:
                    print("Expanded bias shape: ", expanded_bias.shape)
        
        # expand the second dimension
        if expand_right:
            print("inside expand right")
            print("parent_layer ", parent_layer)
            if base_layer_name + '.weight' in model_state_dict:
                num_new_neurons = int(amount_to_expand2)
                new_weights = (torch.randn(model_state_dict[base_layer_name + '.weight'].shape[0], num_new_neurons)*hyperparam_perturbation).to(device)
                expanded_weights = torch.cat((model_state_dict[base_layer_name + '.weight'], new_weights), dim=1)
                model_state_dict[base_layer_name + '.weight'] = expanded_weights

                print("Expanded weight shape: ", expanded_weights.shape)
        layers_done.add(start_layer)

    for key in list(model_state_dict.keys()):
        if '.0.' in key:
            new_key = key.replace('.0.', '.')
            model_state_dict[new_key] = model_state_dict.pop(key)

    # print the final shape of all the layers in the model
    for layer_name in model_state_dict:
        print(f"Layer {layer_name} shape: {model_state_dict[layer_name].shape}")

    return model_state_dict

import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image

def generate_samples(ckpt_folder, sample_path, classes_remembered, classes_not_remembered, n_samples=100, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(os.path.join(ckpt_folder, "ckpt_modified.pt"), map_location=device)
    config = ckpt['config']
    # build model
    vae = OneHotCVAE(x_dim=config.x_dim, h_dim1= ckpt['h_dims1'], h_dim2=ckpt['h_dims2'], z_dim=config.z_dim)
    vae = vae.to(device)
    
    vae.load_state_dict(ckpt['model'])
    vae.eval()
    
    for cls in classes_remembered:
        sample_dir = os.path.join(sample_path, f"{cls}_samples")
        print("sample_dir ", sample_dir)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir, exist_ok=True)
        i = 0
        with torch.no_grad():
            for _ in tqdm.tqdm(range((n_samples // batch_size)*batch_size)):
                z = torch.randn((batch_size, config.z_dim)).to(device)
                c = (torch.ones(batch_size, dtype=int) * cls).to(device)
                c = F.one_hot(c, 10)
                samples = vae.decoder(z, c).view(-1, 1, 28, 28)
                for x in samples:
                    save_image(x, os.path.join(sample_dir, f'{i}.png'))
                    i += 1
    for cls in classes_not_remembered:
        sample_dir = os.path.join(sample_path, f"{cls}_samples")
        print("sample_dir ", sample_dir)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir, exist_ok=True)
        i = 0
        with torch.no_grad():
            for _ in tqdm.tqdm(range((n_samples //batch_size)*batch_size)):
                z = torch.randn((batch_size, config.z_dim)).to(device)
                c = (torch.ones(batch_size, dtype=int) * cls).to(device)
                c = F.one_hot(c, 10)
                samples = vae.decoder(z, c).view(-1, 1, 28, 28)
                for x in samples:
                    save_image(x, os.path.join(sample_dir, f'{i}.png'))
                    i += 1

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, transforms=None, n=None):
        self.transforms = transforms
        
        path = pathlib.Path(img_folder)
        # self.files = sorted([file for ext in IMAGE_EXTENSIONS
        #                for file in path.glob('*.{}'.format(ext))])
        self.folders = sorted([folder for folder in path.iterdir() if folder.is_dir()])
        self.files = []

        for folder in self.folders:
            # self.files.extend(sorted([file for ext in IMAGE_EXTENSIONS
            #     for file in folder.glob('*.{}'.format(ext))]))
            images = sorted([file for ext in IMAGE_EXTENSIONS
                for file in folder.glob('*.{}'.format(ext))])
            # self.files.extend(images)
            # files = [(images, folder.name.split('_')[0])]
            files = [(image, folder.name.split('_')[0]) for image in images]
            self.files.extend(files)
        assert n is None or n <= len(self.files)
        self.n = len(self.files) if n is None else n
        
    def __len__(self):
        return self.n

    def __getitem__(self, i):
        path, label = self.files[i]
        img = Image.open(path).convert('L')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

def GetImageFolderLoader(path, batch_size):

    dataset = ImagePathDataset(
            path,
            transforms=transforms.ToTensor(),
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size
    )
    
    return loader

def evaluate_with_classifier(ckpt_folder, classifier_path, classes_remembered, classes_not_remembered, clean=True, sample_path="./samples", batch_size=32):
    generate_samples(ckpt_folder, sample_path, classes_remembered, classes_not_remembered, n_samples=1000, batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier().to(device)
    model.eval()
    ckpt = torch.load(classifier_path, map_location=device)
    model.load_state_dict(ckpt)
    
    loader = GetImageFolderLoader(sample_path, batch_size=32)
    n_samples = len(loader.dataset)
    
    entropy_cum_sum = 0
    forgotten_prob_cum_sum = 0
    remembered_prob_cum_sum = 0
    print("total samples ", n_samples)
    cum_sum = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0
    }
    for data, label in tqdm.tqdm(iter(loader), total=n_samples//batch_size):
        log_probs = model(data.to(device)) # model outputs log_softmax
        probs = log_probs.exp()
        entropy = -torch.multiply(probs, log_probs).sum(1)
        avg_entropy = torch.sum(entropy)/n_samples
        entropy_cum_sum += avg_entropy.item()
            # forgotten_prob_cum_sum += (probs[:, cls] / (n_samples*len(classes_not_remembered)/(len(classes_remembered) + len(classes_not_remembered)))).sum().item()
            # cum_sum[cls] += probs[:, cls].sum().item() / (batch_size)
            # check if the probs
            # if cls == int(label):
            #     # find the max probability position
            #     max_prob = probs.argmax(1)
            #     # increase the count of the class for which the max prob is cls
            #     count = (max_prob == cls).sum().item()
            #     cum_sum[cls] += count
        for i in range(probs.shape[0]):
            max_prob = probs[i].argmax()
            if max_prob == int(label[i]):
                cum_sum[max_prob.item()] += 1
            # remembered_prob_cum_sum += (probs[:, cls] / (n_samples*len(classes_remembered)/(len(classes_remembered) + len(classes_not_remembered)))).sum().item()
            # cum_sum[cls] += probs[:, cls].sum().item() / (batch_size)
    
    # print the class wise cum sum
    for cls in cum_sum:
        cum_sum[cls] = (cum_sum[cls]/ ((n_samples//batch_size)*batch_size)) * (len(classes_not_remembered) + len(classes_remembered))
        if cls in classes_remembered:
            print(f"REM    : Class {cls} : {cum_sum[cls]}")
        else : 
            print(f"FORGOT : Class {cls} : {cum_sum[cls]}")
    
    # remove the samples folder
    if clean:
        import shutil
        shutil.rmtree(sample_path)
    return cum_sum, entropy_cum_sum