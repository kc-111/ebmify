import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class QuadcopterDynamics(nn.Module):
    def __init__(self, m: float = 1.0, g: float = 9.81, L: float = 0.25, km: float = 0.02):
        """
        Initializes the quadcopter physical parameters.
        
        Args:
            m: Mass of the quadcopter in kg.
            g: Gravitational acceleration (9.81 m/s^2).
            L: Arm length (meters). Distance from center to any motor.
            km: Drag coefficient ratio. Relates motor thrust to the counter-torque 
                produced by the spinning propeller (Yaw control).
        """
        super().__init__()
        self.m = m
        self.g = g
        self.L = L
        self.km = km
        
        # Principal moments of inertia (Inertia Tensor). 
        # These constants define how hard it is to rotate the drone around each axis.
        self.register_buffer('I', torch.tensor([0.005, 0.005, 0.01]))

    def forward(self, t: float, y: torch.Tensor, u: torch.Tensor, wind: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The core ODE function: computes the rate of change for all 12 states.
        
        Args:
            t: Current time (s).
            y: State tensor of shape (N, 12).
            u: Motor thrusts (N, 4). Each column is the force (Newtons) of one motor.
            wind: External force vector (N, 3) applied in the world frame.
            
        Returns:
            dy_dt: The time derivative of the state (N, 12).
        """
        # 1. Kinematics: The velocity of position is simply the velocity state.
        v_world = y[:, 3:6]
        
        # 2. Extract Orientation
        phi, theta, psi = y[:, 6], y[:, 7], y[:, 8]  # Roll, Pitch, Yaw
        p, q, r = y[:, 9], y[:, 10], y[:, 11]        # Body-frame angular rates

        # 3. Force and Torque "Mixer" (X-Configuration)
        # We combine 4 motor thrusts into 1 total force and 3 directional torques.
        T = u.sum(dim=1)
        leff = self.L * 0.707106 # L * sin(45 degrees)
        tau_x = leff * (u[:, 0] + u[:, 3] - u[:, 1] - u[:, 2]) # Roll torque
        tau_y = leff * (u[:, 0] + u[:, 1] - u[:, 2] - u[:, 3]) # Pitch torque
        tau_z = self.km * (u[:, 0] - u[:, 1] + u[:, 2] - u[:, 3]) # Yaw torque

        # 4. Translational Dynamics (Newton's Second Law: F = ma)
        cp, sp = torch.cos(phi), torch.sin(phi)
        ct, st = torch.cos(theta), torch.sin(theta)
        cs, ss = torch.cos(psi), torch.sin(psi)

        # Acceleration involves rotating the body-thrust into the world frame
        ax = (T / self.m) * (cp * st * cs + sp * ss)
        ay = (T / self.m) * (cp * st * ss - sp * cs)
        az = (T / self.m) * (cp * ct) - self.g # Subtract gravity
        
        dv_world = torch.stack([ax, ay, az], dim=1)
        if wind is not None:
            dv_world += (wind / self.m)

        # 5. Rotational Kinematics (Euler Angle Rates)
        tt = torch.tan(theta)
        d_phi   = p + q * sp * tt + r * cp * tt
        d_theta = q * cp - r * sp
        d_psi   = (q * sp + r * cp) / ct
        d_angles = torch.stack([d_phi, d_theta, d_psi], dim=1)

        # 6. Rotational Dynamics (Euler's Equations for Rigid Bodies)
        # Calculates how the angular velocity (p,q,r) changes based on torques.
        dp = (tau_x + (self.I[1] - self.I[2]) * q * r) / self.I[0]
        dq = (tau_y + (self.I[2] - self.I[0]) * p * r) / self.I[1]
        dr = (tau_z + (self.I[0] - self.I[1]) * p * q) / self.I[2]
        d_omega = torch.stack([dp, dq, dr], dim=1)

        return torch.cat([v_world, dv_world, d_angles, d_omega], dim=1)

def quad_ode_fun(t: float, y: torch.Tensor, args: Dict) -> torch.Tensor:
    """
    Standard wrapper for the ODE solver.
    
    Args:
        t: Current time.
        y: State (N, 12).
        args: Dictionary containing 'model' (QuadcopterDynamics), 
              'u' (Control input tensor), and optional 'wind'.
    """
    return args['model'](t, y, args['u'], wind=args.get('wind'))

