"""Every constant of iSEE, in one place, with the part of the paper it comes from.

"Sec." refers to the main paper, "Supp." to the supplementary material.
"""

# ---------------------------------------------------------------------------------------------
# Backbone f and feature MLP g_psi (Sec. 4.1 "Implementation details", Supp. "Encoder")
# ---------------------------------------------------------------------------------------------
BACKBONE_ARCH = "vit_small_patch14_reg4_dinov2"  # DINOv2 ViT-S/14 with 4 registers (timm name)
PATCH_SIZE = 14
BACKBONE_DIM = 384          # f(x) per patch
FEATURE_DIM = 64            # D: h_t = g_psi(f(x_t)), a two-layer MLP 384 -> 768 -> 64
FEATURE_HIDDEN = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------------------------
# Two streams (Sec. 3.1, Supp. "Two streams")
# ---------------------------------------------------------------------------------------------
SLOT_DIM = 64               # s^k = [a^k ; p^k]
APPEARANCE_DIM = 60         # a^k = s^k[0:60]
POSITION_DIM = 4            # p^k = s^k[60:64]
POSITION_CODE_DIM = 4       # c_n = [sin(pi/2 x), cos(pi/2 x), sin(pi/2 y), cos(pi/2 y)]
POSITION_CODE_FREQ = 0.5    # in units of pi: the single Fourier band pi/2 (injective on [-1, 1])
N_ITERS = 2                 # corrector iterations per frame ...
N_ITERS_FIRST = 3           # ... and on the first frame
APPEARANCE_MLP_HIDDEN = 256 # residual MLP after GRU_a (4 x slot width, as in SlotContrast)
ATTENTION_EPS = 1e-8        # A-bar = (A + eps) / sum_n (A + eps)
RMS_EPS = 1e-6              # RMS normalisation of the position stream

# Predictors: one transformer block with four heads per stream; the position predictor reads
# an 8-dimensional projection of a through a stop-gradient.
PREDICTOR_HEADS = 4
PREDICTOR_APPEARANCE_PROJ = 8

# Decoder (Supp. "Two streams", the displayed decoder equation)
DECODER_HIDDEN = (1024, 1024, 1024)
POSE_HEAD_HIDDEN = 64       # h(p) -> (mu, sigma)
SCALE_RANGE = (0.01, 4.0)   # sigma is a clamped exponential
DELTA = 5.0                 # u = (x - mu) / (delta * sigma), as in ISA
FOURIER_BANDS = 8           # gamma(u): u, sin(pi 2^j u), cos(pi 2^j u), j = 0..7
SCALE_EPS = 1e-5

# ---------------------------------------------------------------------------------------------
# Temporal evidence normalisation, TEN (Sec. 3.2, Supp. "Temporal evidence normalisation")
# ---------------------------------------------------------------------------------------------
TEN_TOP_M = 4               # e^k_t = mean of the m largest attention values
TEN_BLOCK = 3               # e_loc: largest mean over a 3 x 3 block of patches inside W^k
TEN_GUARD_KAPPA = 2.0       # onset guard: (e_loc / (kappa * ebar))^power < tau ...
TEN_GUARD_POWER = 8
TEN_GUARD_EMA = 0.5         # ... with ebar the exponential average of e_loc at this rate
TEN_SCENE_SIGMA = 4.0       # track_scene: distance scale (patches) of the other slots' weights ...
TEN_SCENE_MAX_AREA = 0.15   # ... and the largest share of the frame a slot may cover to be counted
# tau, rho, the release evidence and the window size are set per model in its config (`ten:`).

# ---------------------------------------------------------------------------------------------
# Walker (Sec. 3.3, Supp. "Walker")
# ---------------------------------------------------------------------------------------------
WALKER_EMBED_DIM = 32       # e_t: layer norm + 1x1 conv to 32 channels (+ ConvGRU residual)
WALKER_WINDOW = 11          # W(n): 11 x 11 patches
WALKER_STRIDE = 2           # the walker steps every second frame
PSTAR_ITERS = 200           # P*: 200 iterations of GRU_p from zero, driven by v_p(c_n)
WALKER_SCALE_INIT = 20.0    # s at initialisation
WALKER_STAY_INIT = 5.0      # beta at offset 0 at initialisation (every other offset 0)
WALKER_SIGMA_INIT = 0.05    # the training likelihood's width sigma at initialisation ...
WALKER_SIGMA_MIN = 0.02     # ... and its lower bound
