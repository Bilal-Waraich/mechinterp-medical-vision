# Spatial Flattening

## Background: How the Swin Transformer Represents an Image

UMedPT is built on a **Swin Transformer** (Liu et al., 2021). To understand spatial flattening you first need to understand how a Swin Transformer processes an image — it is fundamentally different from a CNN.

A CNN slides convolutional filters across the image and ultimately produces a single summary vector (after global average pooling) that represents the whole image. There is no notion of "where" information came from by the time you read off the final representation.

A Swin Transformer works differently. It divides the input image into a grid of non-overlapping patches and processes each patch as a **token** — a vector that retains its spatial identity throughout the network. This is directly analogous to how a language model processes a sentence: each word is a token, each patch is a token. Just as a language model produces one output vector per word position, the Swin Transformer produces one output vector per patch position.

After passing a 224×224 image through the encoder and hooking at the Stage 3→4 PatchMerging layer (pm_index=2), the output has shape:

```
[1, 7, 7, 1024]   ← (batch, spatial_height, spatial_width, d_model)
```

**One image produces 49 tokens**, arranged in a 7×7 grid. Each token is a 1024-dimensional vector encoding what the encoder has learned about a ~32×32 pixel region of the original image. The grid maps directly back onto the image:

```
Original 224×224 image divided into 7×7 regions (each ~32×32 px):

┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ (0,0)│ (0,1)│ (0,2)│ (0,3)│ (0,4)│ (0,5)│ (0,6)│  ← upper lung / clavicle
├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ (1,0)│ (1,1)│ (1,2)│ (1,3)│ (1,4)│ (1,5)│ (1,6)│
├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ (2,0)│ (2,1)│ (2,2)│ (2,3)│ (2,4)│ (2,5)│ (2,6)│  ← upper-mid lung field
├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ (3,0)│ (3,1)│ (3,2)│ (3,3)│ (3,4)│ (3,5)│ (3,6)│  ← cardiac / mid lung
├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ (4,0)│ (4,1)│ (4,2)│ (4,3)│ (4,4)│ (4,5)│ (4,6)│
├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ (5,0)│ (5,1)│ (5,2)│ (5,3)│ (5,4)│ (5,5)│ (5,6)│  ← lower lung / diaphragm
├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ (6,0)│ (6,1)│ (6,2)│ (6,3)│ (6,4)│ (6,5)│ (6,6)│  ← base / costophrenic angle
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

Each cell contains one 1024-dim vector. Together the 49 vectors describe the whole image at a semantic, region-level granularity.

---

## What Flattening Does

A Sparse Autoencoder operates on a **single vector** at a time. It has no architecture for processing grids or sequences — it takes in one `[1024]` vector and produces one set of sparse feature activations. It cannot consume the `[7, 7, 1024]` grid directly.

Flattening collapses the two spatial dimensions into the batch dimension:

```python
feat = captured["feat"]               # [B, 7, 7, 1024]
flat = feat.reshape(B * 7 * 7, 1024)  # [B*49, 1024]
```

For a batch of 32 images this produces `32 × 49 = 1,568` vectors. The SAE processes each of these 1,568 vectors completely independently — it sees no difference between "49 tokens from the same image" and "49 tokens from 49 different images". From the SAE's perspective they are simply 1,568 training examples.

**Crucially, the spatial correspondence is never lost.** Because the reshape is deterministic and ordered, token at flat index `i` always came from grid position `(i // 7, i % 7)` of image `i // 49`. The spatial structure is not destroyed — it is set aside during SAE processing and can be recovered exactly afterwards.

---

## Why We Do It

There are three distinct reasons, each important:

### 1. The SAE architecture requires it

An SAE is a two-layer MLP: encoder matrix `W_enc [d_model, d_sae]`, decoder matrix `W_dec [d_sae, d_model]`. It is defined for fixed-size vectors. There is no way to feed a 2D spatial grid into it without first reducing it to a flat sequence of vectors. Flattening is the simplest and most information-preserving way to do this — no spatial information is discarded, just reindexed.

### 2. Position-agnostic features are what we want

The goal is to find features that represent **disease concepts** — "consolidation", "pleural effusion", "TB infiltrate" — not features that represent "consolidation in the lower-left patch of a CheXpert image". A pleural effusion appears in the lower lung fields but its exact patch position varies between patients, between images, and between datasets.

By training the SAE on all 49 patches without any positional encoding, we force it to learn features that fire on the *content* of a patch regardless of where in the image that patch came from. A feature that fires on opacity-containing patches will fire whether that opacity is in position (3,2) or (5,4). This is the right inductive bias for disease feature discovery.

If we had instead kept positional information (e.g. concatenated patch coordinates to each token), the SAE could learn features like "upper-left patch of a TB image", which would be useless for cross-dataset universality.

### 3. It multiplies training data by 49×

SAEs are data-hungry. Training on too few tokens leads to dead features — latent dimensions that never activate and contribute nothing. With ~26,700 source images across four datasets, treating each image as a single vector would give the SAE only ~26,700 training examples — far too few for a d_sae=4096 dictionary.

Flattening gives **~1.3 million training tokens** from the same images:

| Dataset | Images | Tokens (×49) |
|---|---|---|
| CheXpert | 10,000 | 490,000 |
| RSNA | 10,000 | 490,000 |
| SIIM-COVID19 | ~6,000 | ~294,000 |
| TB Shenzhen | ~700 | ~34,300 |
| **Total** | **~26,700** | **~1,308,300** |

This is the same logic as in NLP SAEs: Gao et al. (2024) train on billions of tokens from text by treating each token position in each sequence independently. We apply the same principle to vision.

---

## Recovering Spatial Structure at Analysis Time

Because the spatial ordering is preserved, flattening can be reversed at any time:

```python
# Get the 49 SAE feature activations for one image
tokens = activations[image_idx * 49 : (image_idx + 1) * 49]  # [49, d_sae]

# Encode through SAE to get feature activations
feature_acts = sae.encode(tokens)                              # [49, d_sae]

# Recover the spatial map for feature f
heatmap = feature_acts[:, f].reshape(7, 7)                    # [7, 7]
```

This gives a **spatial activation map** — a heatmap showing which of the 49 image regions caused feature `f` to fire, and how strongly. The heatmap can then be upsampled from 7×7 back to 224×224 and overlaid on the original X-ray for visualisation.

This is also the basis of **bounding box validation**: SIIM-COVID19 and RSNA provide radiologist bounding boxes marking where opacities appear. For a feature identified as representing airspace opacity, we check whether its 7×7 heatmap spatially overlaps with those boxes. Agreement means the feature is not just statistically correlated with the disease label — it is firing at the correct anatomical location. This would be impossible if we had discarded spatial structure by averaging all 49 tokens into a single image vector.

---

## The Spatial Grids at Each Hook Stage

The hook stage (pm_index) controls how fine-grained the spatial grid is. Earlier stages produce finer grids with smaller receptive fields per token:

| pm_index | Stage | Grid | Tokens/image | d_model | Receptive field/patch |
|---|---|---|---|---|---|
| 0 | Stage 1→2 | 28×28 | 784 | 256 | ~8×8 px — local texture, edges |
| 1 | Stage 2→3 | 14×14 | 196 | 512 | ~16×16 px — mid-level structure |
| 2 | Stage 3→4 | **7×7** | **49** | **1024** | **~32×32 px — semantic regions** |
| 3 | Stage 4 | 7×7 | 49 | 1024 | largest receptive field, most context |

**pm_index=2** is used. At this stage each token's receptive field (~32×32 px) is large enough to contain a meaningful anatomical region — a costophrenic angle, a lobe of the lung, the cardiac silhouette — but small enough that 49 tokens provide genuine spatial coverage of the image. Earlier stages would give finer localisation but features at those stages represent texture and low-level structure rather than disease semantics. Later stages (Stage 4) have the same 7×7 grid but even larger receptive fields, making spatial localisation less precise.
