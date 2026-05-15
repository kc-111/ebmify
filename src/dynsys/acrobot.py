class AcrobotDynamics(nn.Module):
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        super().__init__()
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g
        # Moments of inertia for thin rods
        self.I1 = (m1 * l1**2) / 12
        self.I2 = (m2 * l2**2) / 12

    def forward(self, t, state, u):
        """
        state: [theta1, theta2, d_theta1, d_theta2] (N, 4)
        u: Torque at joint 2 (N, 1)
        """
        th1, th2, dth1, dth2 = state.unbind(dim=1)
        
        # Mass Matrix M(q)
        m11 = self.I1 + self.I2 + self.m2 * self.l1**2 + \
              0.5 * self.m1 * self.l1**2 + self.m2 * self.l1 * self.l2 * torch.cos(th2)
        m12 = self.I2 + 0.5 * self.m2 * self.l1 * self.l2 * torch.cos(th2)
        m21 = m12
        m22 = self.I2
        
        # Coriolis/Centrifugal Vector C(q, dq)
        h = self.m2 * self.l1 * (self.l2 / 2.0) * torch.sin(th2)
        c1 = -h * dth2**2 - 2 * h * dth1 * dth2
        c2 = h * dth1**2
        
        # Gravity Vector G(q)
        g1 = (0.5 * self.m1 * self.l1 + self.m2 * self.l1) * self.g * torch.sin(th1) + \
             (0.5 * self.m2 * self.l2) * self.g * torch.sin(th1 + th2)
        g2 = (0.5 * self.m2 * self.l2) * self.g * torch.sin(th1 + th2)
        
        # Solve M(q) * ddq = Torque - C - G
        # Only joint 2 is actuated: Torque = [0, u]
        rhs1 = -c1 - g1
        rhs2 = u.squeeze() - c2 - g2
        
        # Cramer's rule or explicit inverse for 2x2
        det = m11 * m22 - m12 * m21
        ddth1 = (rhs1 * m22 - rhs2 * m12) / det
        ddth2 = (m11 * rhs2 - m21 * rhs1) / det
        
        return torch.stack([dth1, dth2, ddth1, ddth2], dim=1)