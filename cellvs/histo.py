import cv2
import numpy as np
import torch
from PIL import Image

MODEL = "hf-hub:wisdomik/QuiltNet-B-32"
TEMPLATES = ["a histology image of {}", "an H&E stained section of {}", "a microscopy image of normal {} tissue"]
TISSUES = {
    "cerebelo": "cerebellum", "cortex cerebral": "cerebral cortex", "medula espinal": "spinal cord",
    "nervo": "peripheral nerve", "lingua": "tongue", "esofago": "esophagus", "estomago": "stomach",
    "intestino delgado": "small intestine", "colon": "colon", "figado": "liver", "pancreas": "pancreas",
    "rim": "kidney", "ureter": "ureter", "bexiga": "urinary bladder", "pulmao": "lung", "traqueia": "trachea",
    "musculo cardiaco": "heart muscle", "musculo esqueletico": "skeletal muscle", "pele": "skin", "baco": "spleen",
    "linfonodo": "lymph node", "tireoide": "thyroid", "arteria": "artery", "osso": "bone", "cartilagem": "cartilage",
    "tecido adiposo": "adipose tissue", "testiculo": "testis", "ducto deferente": "ductus deferens",
    "utero": "uterus", "ovario": "ovary",
}


def crop_circle(bgr):
    """largest bright blob = the eyepiece circle; returns the square inside it, in rgb"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(cv2.GaussianBlur(gray, (21, 21), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    ys, xs = np.where(lab == k)
    half = int(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi) / np.sqrt(2) * 0.95)
    y0, x0 = max(int(ys.mean()) - half, 0), max(int(xs.mean()) - half, 0)
    return cv2.cvtColor(bgr[y0 : y0 + 2 * half, x0 : x0 + 2 * half], cv2.COLOR_BGR2RGB)


def through_eyepiece(bgr):
    """phone photos through the eyepiece have dark corners, scanned or published micrographs do not"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = max(min(h, w) // 10, 1)
    corners = [gray[:k, :k], gray[:k, -k:], gray[-k:, :k], gray[-k:, -k:]]
    return float(np.mean([c.mean() for c in corners])) < 60


def read_photo(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return crop_circle(bgr) if through_eyepiece(bgr) else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _norm(e):
    return e / e.norm(dim=-1, keepdim=True)


class Model:
    def __init__(self):
        import open_clip

        self.net, _, self.pre = open_clip.create_model_and_transforms(MODEL)
        self.net.eval()
        tok = open_clip.get_tokenizer(MODEL)
        with torch.no_grad():
            t = [_norm(self.net.encode_text(tok([s.format(en) for s in TEMPLATES]))).mean(0) for en in TISSUES.values()]
        self.text = _norm(torch.stack(t))
        self.scale = self.net.logit_scale.exp().item()

        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()

    @torch.no_grad()
    def quilt_emb(self, rgbs):
        return _norm(self.net.encode_image(torch.stack([self.pre(Image.fromarray(x)) for x in rgbs]))).numpy()

    def zero_shot(self, rgbs=None, emb=None):
        e = torch.from_numpy(self.quilt_emb(rgbs) if emb is None else emb)
        return (self.scale * e @ self.text.T).softmax(-1).numpy()

    @torch.no_grad()
    def prompt_probs(self, emb, prompts):
        import open_clip

        t = _norm(self.net.encode_text(open_clip.get_tokenizer(MODEL)(prompts)))
        return (self.scale * torch.from_numpy(emb) @ t.T).softmax(-1).numpy()

    @torch.no_grad()
    def similarity_emb(self, rgbs):
        """dinov2 for photo-vs-photo: it spreads unrelated photos further apart than QuiltNet does"""
        x = np.stack([cv2.resize(r, (224, 224), interpolation=cv2.INTER_AREA) for r in rgbs]).astype(np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return _norm(self.dino(torch.from_numpy(x.transpose(0, 3, 1, 2).astype(np.float32)))).numpy()
