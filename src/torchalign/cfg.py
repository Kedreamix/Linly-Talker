from yacs.config import CfgNode as CN

_C = CN()

### INPUT ####
_C.INPUT = CN()
_C.INPUT.SIZE = [256, 256]
_C.INPUT.SCALE = 1.25
_C.INPUT.DATASET = 'WFLW'
_C.INPUT.BBOX = 'P1'
_C.INPUT.FLIP = True
_C.INPUT.FLIP_ORDER = [
    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
    14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0, 46, 45, 44,
    43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52, 53,
    54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61,
    60, 67, 66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91,
    90, 89, 88, 95, 94, 93, 97, 96
]

### BACKBONE ###
_C.BACKBONE = CN()
_C.BACKBONE.ARCH = 'hrnet18'

### HEATMAP ###
_C.HEATMAP = CN()
_C.HEATMAP.ARCH = 'HeatmapHead'
_C.HEATMAP.IN_CHANNEL = 270
_C.HEATMAP.PROJ_CHANNEL = 270
_C.HEATMAP.OUT_CHANNEL = 98 # number of landmarks
_C.HEATMAP.STRIDE = 4.0
_C.HEATMAP.ENCODER = 'Coordinate2BinaryHeatmap'
_C.HEATMAP.DECODER = 'BinaryHeatmap2Coordinate'
_C.HEATMAP.BLOCK = 'BinaryHeadBlock'
_C.HEATMAP.TOPK = 9

cfg = _C
