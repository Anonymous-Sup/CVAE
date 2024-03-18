from models.CVAE import VAE
from models.Classifier import Classifier, NormalizedClassifier
from models.Flows import Flows, InvertibleMLPFlow, YuKeMLPFLOW, YuKeMLPFLOW_onlyX, YuKeMLPFLOW_onlyX_seperateZ, YuKeMLPFLOW_onlyX_seperateZ_init
from models.NIPS import NIPS

__factory = {
    'CVAE': VAE
}

test_only_x_input = True

def build_model(config, num_classes):
    
    print("Initializing Flow model: {}".format(config.MODEL.FLOW_TYPE))
    if config.MODEL.FLOW_TYPE == 'invertmlp':
        flows_model = InvertibleMLPFlow(config.MODEL.LATENT_SIZE, config.MODEL.LATENT_SIZE, 1)
    
    elif config.MODEL.FLOW_TYPE == 'yuke_mlpflow':
        if test_only_x_input:
            # flows_model = YuKeMLPFLOW_onlyX(
            #     latent_size=config.MODEL.LATENT_SIZE,
            #     hidden_dim=64,
            #     output_dim=config.MODEL.LATENT_SIZE,
            #     num_layers=3
            # )
            flows_model = YuKeMLPFLOW_onlyX_seperateZ(
                latent_size=config.MODEL.LATENT_SIZE,
                hidden_dim=64,
                output_dim=1,
                num_layers=4
            )
        else:
            '''
             todo: if latent_size is not is there is a problm about hidden_dim
            '''
            flows_model = YuKeMLPFLOW(
                latent_size=config.MODEL.LATENT_SIZE,
                hidden_dim=64,
                output_dim=1,
                num_layers=4)
    else:
        flows_model = Flows(config.MODEL.LATENT_SIZE, flow_type=config.MODEL.FLOW_TYPE, K=10)

    print("Initializing vae model: {}".format(config.MODEL.VAE_TYPE))
    if config.MODEL.VAE_TYPE == 'cvae':
        vae_model = VAE(
            feature_dim=config.MODEL.FEATURE_DIM,
            hidden_dim=256,
            output_dim=config.MODEL.LATENT_SIZE, 
            n_layers=4
    )
    else:
        raise KeyError("Invalid model name, got '{}', but expected to be one of {}".format(config.MODEL.NAME, __factory.keys()))

    # 768 = latent_size * hiden_dim = 12*64
    # if latent_size is not 12, there is a problem
    # maybe 36*24 = 864
    # maybe 64*12 = 768
    model = NIPS(vae_model, flows_model, feature_dim=config.MODEL.FEATURE_DIM, hidden_dim=768, latent_size=config.MODEL.LATENT_SIZE)
    
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    
    # Build classifier
    if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
        classifier = Classifier(feature_dim=config.MODEL.LATENT_SIZE, num_classes=num_classes)
    else:
        classifier = NormalizedClassifier(feature_dim=config.MODEL.LATENT_SIZE, num_classes=num_classes)
    
    return model, classifier


