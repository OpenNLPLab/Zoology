from torch import nn

try:
    from mnet_pytorch import EOS
except:
    EOS = None
    
class Lcsm(nn.Module):
    def __init__(
        self,
        d_model=512,
        expand_dim=128,
        bias=False,
        c_type=0,  # compute type, 1: linear layer 2: ssm
        e_type=0,
        f_type=0,
        s_type=0,
        f_learned=True,
        ssm_dim=16,
        tau=16,
        t_type=0,  # transform type
        use_tau=True,
        **kwargs,
    ):
        super().__init__()
        
        embed_dim = d_model
        self.mixer = EOS(
            embed_dim=embed_dim,
            expand_dim=expand_dim,
            bias=bias,
            c_type=c_type,
            e_type=e_type,
            f_type=f_type,
            s_type=s_type,
            f_learned=f_learned,
            ssm_dim=ssm_dim,
            tau=tau,
            t_type=t_type,
            use_tau=use_tau,
        )
        self.embed_dim = embed_dim
        self.expand_dim = expand_dim
        
    def forward(self, x):
        return self.mixer(x)
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.embed_dim * self.expand_dim