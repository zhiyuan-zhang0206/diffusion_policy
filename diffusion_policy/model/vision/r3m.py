import os
import copy
import gdown
import omegaconf
import hydra
from PIL import Image

import torch
import torchvision.transforms as T
import torch.nn.functional as F


def load_r3m(model_id):
    VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight", "l2dist", "bs"]
    def remove_language_head(state_dict):
        keys = state_dict.keys()
        ## Hardcodes to remove the language head
        ## Assumes downstream use is as visual representation
        for key in list(keys):
            if ("lang_enc" in key) or ("lang_rew" in key):
                del state_dict[key]
        return state_dict

    def cleanup_config(cfg):
        config = copy.deepcopy(cfg)
        keys = config.agent.keys()
        for key in list(keys):
            if key not in VALID_ARGS:
                del config.agent[key]
        config.agent["_target_"] = "r3m.R3M"
        config["device"] = 'cpu'
        
        ## Hardcodes to remove the language head
        ## Assumes downstream use is as visual representation
        config.agent["langweight"] = 0
        return config.agent

    home = os.environ["R3M_HOME"]
    assert home is not None, "R3M_HOME is not set"
    if model_id == "resnet50":
        foldername = "r3m_50"
        modelurl = 'https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA'
        configurl = 'https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8'
    elif model_id == "resnet34":
        foldername = "r3m_34"
        modelurl = 'https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE'
        configurl = 'https://drive.google.com/uc?id=1RY0NS-Tl4G7M1Ik_lOym0b5VIBxX9dqW'
    elif model_id == "resnet18":
        foldername = "r3m_18"
        modelurl = 'https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-'
        configurl = 'https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6'
    else:
        raise NameError('Invalid Model ID')

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
        
    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    r3m_state_dict = remove_language_head(torch.load(modelpath, map_location=torch.device('cpu'))['r3m'])
    filtered_state_dict = {}
    for key, value in r3m_state_dict.items():
        if key.startswith("module"):
            new_key = key.replace("module.", "")
            filtered_state_dict[new_key] = value
    rep.load_state_dict(filtered_state_dict)
    return rep


class R3M(torch.nn.Module):
    def __init__(
        self,
        model_id: str
    ):
        super().__init__()
        r3m = load_r3m(model_id)

        self.model = r3m.convnet
        self.normlayer = r3m.normlayer
        self.feature_size = r3m.outdim

        self.preprocess = T.Compose([
            T.ToTensor(),
            self.normlayer
        ])
    
    def raw_preprocess(self, image: Image):
        # depreciated
        shorter_edge = min(image.size)
        process = T.Compose([
            T.CenterCrop(shorter_edge),
            T.Resize(224),
            T.ToTensor(),
            self.normlayer
        ])
        return process(image)
    
    def forward(self, images):
        return self.model(images)


class R3MImageEncoderWrapper:
    def __init__(self, model_id='resnet34', device:str=None):
        self.model_id = model_id
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)
        self.r3m_model = R3M(model_id).to(self.device).eval()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def __call__(self, images):
        assert isinstance(images, torch.Tensor)
        assert len(images.shape) == 5 or len(images.shape) == 4 # B, L, C, H, W. Note there could be a time dimension L.
        if len(images.shape) == 4:
            B, C, H, W = images.shape
            images = images.reshape(B, C, H, W)
        else:
            B, L, C, H, W = images.shape
            images = images.reshape(B*L, C, H, W)
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images = self.normalize(images)
        features = self.r3m_model(images)
        if len(images.shape) == 4:
            return features
        else:
            return features.reshape(B, L, -1)