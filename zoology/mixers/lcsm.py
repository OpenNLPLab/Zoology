from torch import nn

try:
    from lcsm_pytorch import EosLayer
except:
    EosLayer = None
    
class Lcsm(nn.Module):
    def __init__(
        self,
        d_model=512,
        expand_dim=128,
        bias=False,
        c_type=0,  # compute type, 0: ssm, 1: linear layer
        e_type=0,
        o_type=0,
        o_learned=True,
        s_type=0,
        t_type=0,  # transform(act function) type
        ssm_dim=16,
        tau=16,
        use_tau=True,
        **kwargs,
    ):
        super().__init__()
        
        embed_dim = d_model
        self.mixer = EosLayer(
            embed_dim=embed_dim,
            expand_dim=expand_dim,
            bias=bias,
            c_type=c_type,
            e_type=e_type,
            o_type=o_type,
            o_learned=o_learned,
            s_type=s_type,
            t_type=t_type,
            ssm_dim=ssm_dim,
            tau=tau,
            use_tau=use_tau,
        )
        self.embed_dim = embed_dim
        self.expand_dim = expand_dim
        
    def forward(self, x):
        return self.mixer(x)
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.embed_dim * self.expand_dim