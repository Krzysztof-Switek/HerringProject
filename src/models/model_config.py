MODEL_CONFIGS = {
    "resnet50": {
        "image_size": 224,
        "classifier": "fc",
        "grad_cam_target_layer": "layer4.2.conv3",
        "weights": "ResNet50_Weights.IMAGENET1K_V1"
    },
    "convnext_large": {
        "image_size": 384,
        "classifier": "classifier",
        "grad_cam_target_layer": "features.7.2.conv",
        "weights": "ConvNeXt_Large_Weights.IMAGENET1K_V1"
    },
    "vit_h_14": {
        "image_size": 384,
        "classifier": "heads",
        "grad_cam_target_layer": "encoder.layers.encoder_layer_31.ln_1",
        "weights": "ViT_H_14_Weights.IMAGENET1K_V1"
    },
    "efficientnet_v2_l": {
        "image_size": 480,
        "classifier": "classifier",
        "grad_cam_target_layer": "features.8.0",
        "weights": "EfficientNet_V2_L_Weights.IMAGENET1K_V1"
    },
    "regnet_y_32gf": {
        "image_size": 384,
        "classifier": "fc",
        "grad_cam_target_layer": "trunk_output.block4.block4-30.branch2.conv",
        "weights": "RegNet_Y_32GF_Weights.IMAGENET1K_V1"
    }
}
