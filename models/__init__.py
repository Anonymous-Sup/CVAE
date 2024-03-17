from models.CVAE import VAE
from models.Classifier import Classifier, NormalizedClassifier
from models.Flows import Flows, InvertibleMLPFlow
from models.NIPS import NIPS

__factory = {
    'CVAE': VAE
}


def build_model(config, num_classes):

    print("Initializing Flow model: {}".format(config.MODEL.FLOW_TYPE))
    if config.MODEL.FLOW_TYPE == 'invertmlp':
        flows_model = InvertibleMLPFlow(config.MODEL.LATENT_SIZE, config.MODEL.LATENT_SIZE, 1)
    else:
        flows_model = Flows(config.MODEL.LATENT_SIZE, flow_type=config.MODEL.FLOW_TYPE, K=10)

    print("Initializing vae model: {}".format(config.MODEL.VAE_TYPE))
    if config.MODEL.VAE_TYPE == 'cvae':
        vae_model = VAE(
            encoder_layer_sizes=config.MODEL.ENCODER_LAYER_SIZES,
            latent_size=config.MODEL.LATENT_SIZE,
            decoder_layer_sizes=config.MODEL.DECODER_LAYER_SIZES)
    else:
        raise KeyError("Invalid model name, got '{}', but expected to be one of {}".format(config.MODEL.NAME, __factory.keys()))

    model = NIPS(vae_model, flows_model, feature_dim=config.MODEL.FEATURE_DIM, hidden_dim=None, latent_size=config.MODEL.LATENT_SIZE)
    
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    
    # Build classifier
    if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
        classifier = Classifier(feature_dim=config.MODEL.LATENT_SIZE, num_classes=num_classes)
    else:
        classifier = NormalizedClassifier(feature_dim=config.MODEL.LATENT_SIZE, num_classes=num_classes)
    
    return model, classifier


