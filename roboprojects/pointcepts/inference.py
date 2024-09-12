import sys
sys.path.append('/home/alan/AlanLiang/Projects/3D_Perception/3D_semantic_segmentation/PSMamba')

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from ALPlugin.inference.inference_engines import INFERENCERS

def get_inferencer(config):
    args = default_argument_parser().parse_args()
    for key, value in config.items():
        setattr(args, key, value)
    cfg = default_config_parser(args.config_file, args.options)
    cfg = default_setup(cfg)
    inferencer = INFERENCERS.build(dict(type='SemSegInferencer', cfg=cfg))

    return inferencer