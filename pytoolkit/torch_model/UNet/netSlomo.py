from .UNet_flow import UNet_flow

__all__ = ['backwarp', 'netSlomo']

def backwarp(trgtImg, Flow):
    # trgtImg: B[RGB]HW, Flow: B[UV]HW
    Flow = Flow.permute(0, 2, 3, 1) # -> BHW[UV]
    B, H, W, C = Flow.size()
    XX, YY = np.meshgrid(range(W), range(H))  # [0, W-1] x [0, H-1]
    Grid = np.array([XX, YY]).transpose((1, 2, 0)) # HW[XY]
    Grid = np.broadcast_to(Grid, (B, H, W, C)) #BHW[XY]
    Grid = torch.from_numpy(Grid).float().to(Flow.device)
    warpedGrid = Grid + Flow
    warpedXX, warpedYY = warpedGrid.split(1, dim=3)
    warpedXX = torch.clamp(warpedXX/(W-1)*2-1, -1, 1)
    warpedYY = torch.clamp(warpedYY/(H-1)*2-1, -1, 1)
    return FF.grid_sample(trgtImg, torch.cat((warpedXX, warpedYY), 3), mode='bilinear')

def approx_flow_interp(F01, F10, t):
    Ft1 = (1-t)*(1-t) * F01 - t*(1-t) * F10
    Ft0 = -(1-t)*t * F01 + t*t * F10
    return Ft1, Ft0

def final_image_interp(I0, I1, t, Ft0, Ft1, Vt0):
    It = (1-t) * Vt0 * backwarp(I0, Ft0) + t * (1-Vt0) * backwarp(I1, Ft1)
    return It.div( (1-t) * Vt0 + t * (1-Vt0) + 1e-8 )

class netSlomo(nn.Module):
    def __init__(self, maxFlow=30):
        super(netSlomo, self).__init__()
        self.maxFlow = maxFlow
        self.netA = torchmodel.UNet_flow(6, 2, ch=32)
        self.netB = torchmodel.UNet_flow(16, 5, ch=32)

    def forward_A(self, gif0, gif1):
        # enhance color, estimate flow
        concat = lambda xs: torch.cat(xs, dim=1)
        split = lambda x: torch.split(x, 3, dim=1)
        F01 = self.netA(concat([gif0, gif1])).tanh()
        F10 = self.netA(concat([gif1, gif0])).tanh()
        return F01, F10

    def forward_B(self, I0, I1, I0t, I1t, Ft0, Ft1):
        # enhance color, estimate flow
        concat = lambda xs: torch.cat(xs, dim=1)
        split = lambda x: torch.split(x, 2, dim=1)
        Ft0, Ft1, Vt0 = split(self.netB(concat([I0, I1, I0t, I1t, Ft0, Ft1])).tanh())
        Vt0 = (Vt0+1)/2.0 # in range [0, 1]
        return Ft0, Ft1, Vt0

    def forward(self, gif0, gif1, I0, I1, ts=[0.5]):
        # assertion on ts
        assert isinstance(ts, list), "ts must be a list of numbers"
        for t in ts:
            assert 0<=t<=1, "t must be in range [0, 1]"

        # apply netA
        A_F01, A_F10 = self.forward_A(gif0, gif1)

        # apply netB
        A_Ft1s, A_Ft0s, B_Vt0s, B_Its = [], [], [], []
        for t in ts:
            # interp flow, backwarp image
            A_Ft1, A_Ft0 = approx_flow_interp(A_F01, A_F10, t)
            A_I1t = backwarp(I1, A_Ft1*self.maxFlow)
            A_I0t = backwarp(I0, A_Ft0*self.maxFlow)
            # refine flow, estimate visibility map
            B_Ft0, B_Ft1, B_Vt0 = self.forward_B(I0, I1, A_I0t, A_I1t, A_Ft0, A_Ft1)
            # interp image at t
            B_It = final_image_interp(I0, I1, t, B_Ft0*self.maxFlow, B_Ft1*self.maxFlow, B_Vt0)
            # record
            A_Ft1s.append(A_Ft1.unsqueeze(dim=1))
            A_Ft0s.append(A_Ft0.unsqueeze(dim=1))
            B_Vt0s.append(B_Vt0.unsqueeze(dim=1))
            B_Its.append(B_It.unsqueeze(dim=1))
        A_Ft1s = torch.cat(A_Ft1s, dim=1)
        A_Ft0s = torch.cat(A_Ft0s, dim=1)
        B_Vt0s = torch.cat(B_Vt0s, dim=1)
        B_Its = torch.cat(B_Its, dim=1)
        return B_Its, A_F01, A_F10, A_Ft1s, A_Ft0s, B_Vt0s
