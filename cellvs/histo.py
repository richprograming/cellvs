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
# textbook features, written once for every tissue (not tuned on the evaluation images)
DESC = {
    "cerebelo": ["cerebellum with folia, a pale molecular layer, Purkinje cells and a dense blue granular layer",
                 "H&E of cerebellar cortex showing molecular layer, Purkinje cell layer and granule cell layer"],
    "cortex cerebral": ["cerebral cortex with neurons and glial cells in neuropil, pyramidal neurons",
                        "H&E of brain gray matter with scattered neurons"],
    "medula espinal": ["spinal cord cross section with butterfly shaped gray matter, large motor neurons and white matter",
                       "H&E of spinal cord with anterior horn motor neurons surrounded by white matter tracts"],
    "nervo": ["peripheral nerve with fascicles of wavy axons surrounded by perineurium",
              "H&E of a nerve bundle with wavy fibers and Schwann cell nuclei"],
    "lingua": ["tongue with stratified squamous epithelium, papillae and interlacing skeletal muscle bundles",
               "H&E of tongue showing filiform papillae over skeletal muscle fibers running in different directions"],
    "esofago": ["esophagus with thick nonkeratinized stratified squamous epithelium and muscularis",
                "H&E of esophageal mucosa with stratified squamous epithelium and submucosal glands"],
    "estomago": ["stomach with gastric pits and glands lined by parietal and chief cells",
                 "H&E of gastric mucosa with foveolae and fundic glands"],
    "intestino delgado": ["small intestine with finger-like villi and crypts, goblet cells",
                          "H&E of ileum or jejunum showing long villi lined by columnar epithelium"],
    "colon": ["colon with straight tubular crypts full of goblet cells and no villi",
              "H&E of large intestine mucosa with parallel crypts and muscularis"],
    "figado": ["liver with plates of polygonal hepatocytes, sinusoids, central veins and portal triads",
               "H&E of hepatic lobules with hepatocyte cords"],
    "pancreas": ["pancreas with lobules of serous acini and pale islets of Langerhans",
                 "H&E of exocrine pancreatic acinar tissue with an islet"],
    "rim": ["kidney cortex with round glomeruli and proximal and distal convoluted tubules",
            "H&E of renal parenchyma with glomeruli and tubules"],
    "ureter": ["ureter with star-shaped lumen lined by transitional epithelium and smooth muscle wall",
               "H&E of a ureter cross section with urothelium"],
    "bexiga": ["urinary bladder with transitional epithelium urothelium and thick smooth muscle detrusor",
               "H&E of bladder wall with urothelium and lamina propria"],
    "pulmao": ["lung with thin-walled alveoli, alveolar septa, bronchioles and blood vessels",
               "H&E of lung parenchyma with air spaces like lace"],
    "traqueia": ["trachea with ciliated pseudostratified epithelium, seromucous glands and a hyaline cartilage ring",
                 "H&E of tracheal wall with respiratory epithelium over cartilage"],
    "musculo cardiaco": ["cardiac muscle with branching striated fibers, central nuclei and intercalated discs",
                         "H&E of myocardium with branching cardiomyocytes"],
    "musculo esqueletico": ["skeletal muscle with long parallel striated fibers and peripheral nuclei",
                            "H&E of skeletal muscle fascicles in cross or longitudinal section"],
    "pele": ["skin with keratinized stratified squamous epidermis, dermis, hair follicles and glands",
             "H&E of skin showing epidermis and dermal collagen"],
    "baco": ["spleen with white pulp lymphoid follicles around central arteries and red pulp full of blood",
             "H&E of splenic red pulp and white pulp"],
    "linfonodo": ["lymph node with capsule, cortical lymphoid follicles with germinal centers, paracortex and medulla",
                  "H&E of lymph node densely packed with lymphocytes"],
    "tireoide": ["thyroid with round follicles lined by cuboidal cells and filled with pink colloid",
                 "H&E of thyroid follicles with colloid"],
    "arteria": ["artery cross section with tunica intima, thick muscular tunica media and adventitia",
                "H&E of an elastic or muscular artery wall"],
    "osso": ["bone with osteons, Haversian canals, lamellae and osteocytes in lacunae",
             "ground section or H&E of compact bone"],
    "cartilagem": ["hyaline cartilage with chondrocytes in lacunae within a smooth glassy matrix and perichondrium",
                   "H&E of cartilage with isogenous groups of chondrocytes"],
    "tecido adiposo": ["adipose tissue with large empty polygonal fat cells and thin rims of cytoplasm",
                       "H&E of white fat, chicken wire appearance"],
    "testiculo": ["testis with seminiferous tubules containing spermatogenic cells and interstitial Leydig cells",
                  "H&E of testicular seminiferous tubules"],
    "ducto deferente": ["ductus deferens with a small irregular lumen and a very thick three-layered smooth muscle wall",
                        "H&E of vas deferens cross section with pseudostratified epithelium"],
    "utero": ["uterus with endometrial glands in stroma and thick smooth muscle myometrium",
              "H&E of endometrium and myometrium"],
    "ovario": ["ovary with cortex containing ovarian follicles at different stages and oocytes in a cellular stroma",
               "H&E of ovarian follicles and corpus luteum"],
}
NAMES = list(TISSUES)


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


def prepare(bgr):
    """what the model sees: the inside of the eyepiece circle, or the whole image when there is none"""
    return crop_circle(bgr) if through_eyepiece(bgr) else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_photo(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return prepare(bgr)


def tiles(rgb):
    """whole image + its 4 quarters: the model sees 224 px, quarters keep some cell detail"""
    h, w = rgb.shape[0] // 2, rgb.shape[1] // 2
    return [rgb, rgb[:h, :w], rgb[:h, w:], rgb[h:, :w], rgb[h:, w:]]


def l2(e):
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


class Model:
    def __init__(self):
        import open_clip

        self.net, _, self.pre = open_clip.create_model_and_transforms(MODEL)
        self.net.eval()
        tok = open_clip.get_tokenizer(MODEL)
        with torch.no_grad():
            t = [l2(self.net.encode_text(tok([s.format(TISSUES[n]) for s in TEMPLATES] + DESC[n])).numpy()).mean(0)
                 for n in NAMES]
        self.text = l2(np.stack(t))
        self.scale = self.net.logit_scale.exp().item()

    @torch.no_grad()
    def emb(self, rgbs):
        out = []
        for x in rgbs:
            e = self.net.encode_image(torch.stack([self.pre(Image.fromarray(t)) for t in tiles(x)])).numpy()
            out.append(l2(l2(e).mean(0)))
        return np.stack(out).astype(np.float32)

    def zero_shot(self, emb, names=NAMES):
        """log-probabilities over `names`"""
        z = self.scale * emb @ self.text[[NAMES.index(n) for n in names]].T
        return z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)


def clf_logp(clf, emb):
    """log-probabilities of a saved logistic regression (coef, intercept)"""
    z = emb @ clf["coef"].T + clf["intercept"]
    z -= z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


class TissuePredictor:
    """zero-shot + the classifier from scripts/tissue_train.py when it exists"""

    def __init__(self, clf_path="models/tissue_clf.npz"):
        from pathlib import Path

        self.model = Model()
        self.clf = dict(np.load(clf_path)) if Path(clf_path).exists() else None
        self.names = [str(n) for n in self.clf["names"]] if self.clf else NAMES

    def probs(self, rgbs):
        emb = self.model.emb(rgbs)
        s = self.model.zero_shot(emb, self.names) + (clf_logp(self.clf, emb) if self.clf else 0)
        p = np.exp(s - s.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)
