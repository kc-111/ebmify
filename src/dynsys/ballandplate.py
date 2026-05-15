class BallAndPlateDynamics(nn.Module):
    def __init__(self, m_ball: float = 0.05, R_ball: float = 0.01, 
                 M_plate: float = 1.0, L_plate: float = 0.5, g: float = 9.81):
        super().__init__()
        self.m, self.R = m_ball, R_ball
        self.M, self.L = M_plate, L_plate
        self.g = g
        
        # Moment of inertia of a solid sphere: (2/5) * m * r^2
        # For the ball-and-plate, we use an effective mass constant 'kb' 
        # that accounts for rotational kinetic energy (m + I/R^2)
        self.kb = self.m + (0.4 * self.m * self.R**2) / self.R**2
        
        # Inertia of the plate (treated as a thin square)
        self.Ip = (1/12) * self.M * (self.L**2 + self.L**2)

    def forward(self, t: float, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        state: [x, y, vx, vy, theta_x, theta_y, omega_x, omega_y] (N, 8)
               x, y: position of ball on plate
               theta_x, theta_y: tilt angles of the plate
        u: [tau_x, tau_y] (N, 2) torques applied to the plate axes
        """
        x, y, vx, vy, th_x, th_y, w_x, w_y = state.unbind(dim=1)
        tau_x, tau_y = u.unbind(dim=1)

        # 1. Ball Dynamics (Simplified assuming small angles)
        # Acceleration = (m*g*sin(theta)) / (m + I/R^2)
        # We use the plate tilt to accelerate the ball.
        ax = (self.m * self.g * torch.sin(th_x)) / self.kb
        ay = (self.m * self.g * torch.sin(th_y)) / self.kb

        # 2. Plate Dynamics
        # Tau = I * alpha -> alpha = Tau / I
        # Note: In a high-fidelity model, the ball's position exerts a 
        # counter-torque on the plate (m*g*x), but we'll stick to the core 
        # control challenge here.
        alpha_x = tau_x / self.Ip
        alpha_y = tau_y / self.Ip

        return torch.stack([vx, vy, ax, ay, w_x, w_y, alpha_x, alpha_y], dim=1)