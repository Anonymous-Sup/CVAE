from models.CVAE import VAE
from models.Classifier import Classifier, NormalizedClassifier

__factory = {
    'CVAE': VAE
}


def build_model(config, num_classes):
    print("Initializing model: {}".format(config.MODEL.NAME))
    if config.MODEL.NAME == 'CVAE':
        model = VAE(
            encoder_layer_sizes=config.MODEL.ENCODER_LAYER_SIZES,
            latent_size=config.MODEL.LATENT_SIZE,
            decoder_layer_sizes=config.MODEL.DECODER_LAYER_SIZES,
            conditional=config.MODEL.CONDITIONAL,
            num_labels=config.MODEL.NUM_LABELS)
    else:
        raise KeyError("Invalid model name, got '{}', but expected to be one of {}".format(config.MODEL.NAME, __factory.keys()))
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    
    # Build classifier
    if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
        classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_classes)
    else:
        classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_classes)
    
    return model, classifier


