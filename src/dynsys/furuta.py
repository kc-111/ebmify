class FurutaDynamics(nn.Module):
    def __init__(self, Jr=0.001, Mp=0.02, Lp=0.1, gr=9.81):
        super().__init__()
        self.Jr = Jr # Rotor inertia
        self.Mp = Mp # Pendulum mass
        self.Lp = Lp # Pendulum length
        self.g  = gr

    def forward(self, t, state, u):
        """
        state: [phi (arm), theta (pend), d_phi, d_theta]
        u: Motor torque at the base arm (phi)
        """
        phi, theta, dphi, dtheta = state.unbind(dim=1)
        
        # Terms derived from Lagrangian mechanics
        # Note the coupling: dphi and dtheta appear in each other's acceleration
        Lp = self.Lp
        Mp = self.Mp
        
        # Mass matrix entries
        m11 = self.Jr + Mp * Lp**2 * torch.sin(theta)**2
        m12 = Mp * Lp**2 * torch.cos(theta)
        m21 = m12
        m22 = Mp * Lp**2
        
        # Non-linear force terms (Coriolis + Gravity)
        # These are the "ghost forces" that make it hard to control
        f1 = u.squeeze() - 2 * Mp * Lp**2 * torch.sin(theta) * torch.cos(theta) * dphi * dtheta
        f2 = Mp * self.g * Lp * torch.sin(theta) + Mp * Lp**2 * torch.sin(theta) * torch.cos(theta) * dphi**2
        
        # Inverse dynamics
        det = m11 * m22 - m12 * m21
        ddphi   = (f1 * m22 - f2 * m12) / det
        ddtheta = (m11 * f2 - m21 * f1) / det
        
        return torch.stack([dphi, dtheta, ddphi, ddtheta], dim=1)