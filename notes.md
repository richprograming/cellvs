# notes

Working log. Newest at the bottom. Fill this in as I go, including what didn't work.

## goal
Predict mechanism of action (MoA) of compounds from cell images (BBBC021), without training on MoA labels.
Compare ImageNet ResNet-50 vs DINOv2 embeddings, and see how much batch correction matters.

## evaluation
- NSC: 1-NN (cosine) on treatment-level embeddings, neighbor must be a different compound
- NSCB: same, and the neighbor must also come from a different batch (week)
- this is the usual protocol in the BBBC021 papers, so numbers are comparable-ish. Not exactly: they differ in
  how treatments are aggregated, so don't compare against published tables without checking that

## open questions
- aggregation per (compound, conc, week): does that match how others do it?
- DMSO centering is crude. Try TVN (typical variation normalization)
- fine-tune on the cell images or keep backbones frozen?
- 12 MoA classes, some with 2-3 compounds only -> high variance, report per-class too

## log
- [ ] get data, run 0_prepare_metadata.py
- [ ] resnet50 baseline, raw vs centered
- [ ] dinov2_vits14
- [ ] UMAP plots in notebooks/
