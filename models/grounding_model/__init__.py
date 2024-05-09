from .modal_encoder import CrossModalEncoder
from .query_decoder import QueryDecoder
from .modal_encoder_vivit import CrossModalEncoder as CMEViViT

    
def build_encoder(cfg, vivit = False):
    if vivit:
        return CMEViViT(cfg)
    return CrossModalEncoder(cfg)

def build_decoder(cfg):
    return QueryDecoder(cfg)