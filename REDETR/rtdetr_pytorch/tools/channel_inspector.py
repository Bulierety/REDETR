#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-DETR Channel Inspector - Minimal Version
"""

import torch
import torch.nn as nn
import argparse
import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_model(config):
    try:
        from src.zoo.rtdetr.rtdetr import RTDETR
        from src.nn.backbone.presnet import PResNet
        from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
        from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer as Decoder
        
        backbone = PResNet(**config.get('PResNet', {}))
        encoder = HybridEncoder(**config.get('HybridEncoder', {}))
        decoder = Decoder(**config.get('RTDETRTransformer', {}))
        model = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder, **config.get('RTDETR', {}))
        return model
    except ImportError as e:
        print(f"Import error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='RT-DETR Channel Inspector')
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--minimal', '-m', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    model = build_model(config)
    
    if model is None:
        return
    
    model.eval()
    
    # Minimal output only
    print("RT-DETR Channel Summary:")
    print("=" * 30)
    print(f"Backbone out: {model.backbone.out_channels}")
    print(f"Encoder in:  {model.encoder.in_channels}")
    print(f"Encoder out: {model.encoder.out_channels}")
    
    # Try to get decoder input channels
    if hasattr(model.decoder, 'feat_channels'):
        print(f"Decoder in:  {model.decoder.feat_channels}")
    elif hasattr(model.decoder, 'in_channels'):
        print(f"Decoder in:  {model.decoder.in_channels}")
    else:
        # Decoder input should be same as encoder output
        print(f"Decoder in:  {model.encoder.out_channels}")

if __name__ == '__main__':
    main()