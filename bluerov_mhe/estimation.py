#!/usr/bin/env python
# coding: utf-8

"""
Functions for setting up and solving the Moving Horizon Estimation (MHE) problem 
for the BlueROV2 system.
 """

import casadi as ca
from casadi import vertcat, MX, nlpsol, inf, Function, jacobian # Added Function, jacobian
from casadi.tools import struct_symMX, entry # Import struct tools
import numpy as np

# Import model functions
try:
    from .model import discrete_dynamics_rov, measurement_function_rov
except ImportError:
    # Fallback if running script directly
    from model import discrete_dynamics_rov, measurement_function_rov

def constraint_dynamics_mhe(x_plus, x, u, w, theta, params, Ts):
    """Defines the dynamic constraint for the MHE NLP."""
    # Note: theta might be None if ntheta=0 in params
    # discrete_dynamics_rov now handles theta=None internally
    return x_plus - discrete_dynamics_rov(x, u, w, params, Ts, theta=theta)

def constraint_measurement_mhe(y_meas, x, v, params):
    """Defines the measurement constraint for the MHE NLP."""
    return y_meas - measurement_function_rov(x, v, params)

def setup_mhe_problem(M, params):
    """
    Generates the CasADi NLP problem structure for the MHE.

    Args:
        M: Estimation horizon length.
        params: Dictionary of model and MHE parameters.

    Returns:
        A list containing: [solver, shooting_vars, parameter_vars]
    """
    eta = params.get('eta', 0.9) # Forgetting factor
    nx = params['nx']
    ny = params['ny']
    nw = params['nw']
    nv = params['nv']
    nu = params['nu']
    ntheta = params['ntheta']
    Ts = params['Ts']

    # --- Define symbolic variables for the optimization ---
    shooting_vars_list = [
        entry("x_est", repeat=M + 1, shape=(nx, 1)), # Estimated states over horizon (use tuple shape)
        # entry("y_est", repeat=M, shape=(ny, 1)),   # REMOVED - Estimated measurements
        entry("w_est", repeat=M, shape=(nw, 1)),     # Estimated process noise (use tuple shape)
        entry("v_est", repeat=M, shape=(nv, 1))      # Estimated measurement noise (use tuple shape)
    ]
    # Add theta_est only if parameters are being estimated
    if ntheta > 0:
        # Assuming theta is a vector, shape=(ntheta, 1)
        shooting_vars_list.insert(1, entry("theta_est", repeat=1, shape=(ntheta, 1))) 
        
    shooting_vars = struct_symMX(shooting_vars_list)

    # --- Define symbolic parameters for the optimization ---
    parameter_vars_list = [
        entry("y_meas", repeat=M, shape=(ny, 1)),    # Measured outputs over horizon (use tuple shape)
        entry("u", repeat=M, shape=(nu, 1)),         # Inputs over horizon (use tuple shape)
        entry("x_prior", shape=(nx, 1)),             # Prior state estimate (use tuple shape)
    ]
    # Add theta_prior only if parameters are being estimated
    if ntheta > 0:
         # Assuming theta is a vector, shape=(ntheta, 1)
         parameter_vars_list.append(entry("theta_prior", shape=(ntheta, 1))) 

    parameter_vars = struct_symMX(parameter_vars_list)
    
    # --- Define Weighting matrices ---
    # Process noise weighting (inverse covariance)
    Q_w = np.eye(nw)
    bound_w_pos = params.get('bound_w_pos', 0.01) 
    bound_w_vel = params.get('bound_w_vel', 0.05)
    Q_w[0:4, 0:4] /= (bound_w_pos * bound_w_pos)
    Q_w[4:8, 4:8] /= (bound_w_vel * bound_w_vel)
    
    # Measurement noise weighting (inverse covariance)
    Q_v = np.eye(nv)
    bound_v_pos = params.get('bound_v_pos', 0.1)  
    bound_v_psi = params.get('bound_v_psi', 0.02) 
    if nv == 4: # Assuming [x,y,z,psi] measurement noise
        Q_v[0:3, 0:3] /= (bound_v_pos * bound_v_pos)
        Q_v[3, 3] /= (bound_v_psi * bound_v_psi)
    else: # Default if different measurement setup
         Q_v /= (0.05 * 0.05) 
    
    # Prior weighting (inverse covariance)
    prior_weighting_val = params.get('prior_weighting', 1.0)
    P_x = np.eye(nx) * prior_weighting_val
    if ntheta > 0:
        P_theta = np.eye(ntheta) * prior_weighting_val
    
    # --- Build Objective Function ---
    objective = 0
    
    # Add prior terms (arrival cost)
    delta_x_prior = shooting_vars["x_est"][0] - parameter_vars["x_prior"]
    objective += pow(eta, M) * (delta_x_prior.T @ P_x @ delta_x_prior)
    if ntheta > 0:
        # Access the first element since repeat=1 for theta_est
        delta_theta_prior = shooting_vars["theta_est"][0] - parameter_vars["theta_prior"] 
        objective += pow(eta, M) * (delta_theta_prior.T @ P_theta @ delta_theta_prior)

    # Add stage costs for noise over the horizon
    for k in range(M):
        discount_factor = pow(eta, M - k - 1)
        objective += discount_factor * (shooting_vars["w_est"][k].T @ Q_w @ shooting_vars["w_est"][k])
        objective += discount_factor * (shooting_vars["v_est"][k].T @ Q_v @ shooting_vars["v_est"][k])
        
    # --- Build Constraints ---
    constraints = []
    for k in range(M):
        # Dynamics constraints
        # Access the first element since repeat=1 for theta_est
        current_theta = shooting_vars["theta_est"][0] if ntheta > 0 else None 
        constraints.append(constraint_dynamics_mhe(shooting_vars["x_est"][k+1], 
                                                   shooting_vars["x_est"][k], 
                                                   parameter_vars["u"][k], 
                                                   shooting_vars["w_est"][k], 
                                                   current_theta, params, Ts))
        
        # Measurement constraints
        constraints.append(constraint_measurement_mhe(parameter_vars["y_meas"][k], 
                                                      shooting_vars["x_est"][k], 
                                                      shooting_vars["v_est"][k], params))
        
    # --- Create NLP Solver ---
    nlp = {"x": shooting_vars, "p": parameter_vars, "f": objective, "g": vertcat(*constraints)}
    
    # Solver options (adjust as needed)
    opts = {
        "ipopt.print_level": params.get("ipopt_print_level", 0), 
        "print_time": params.get("print_time", False), 
        'ipopt.max_iter': params.get("ipopt_max_iter", 300), # Increased max iterations
        'ipopt.tol': params.get("ipopt_tol", 1e-6),
        'ipopt.acceptable_tol': params.get("ipopt_acceptable_tol", 1e-3), # Increased acceptable tolerance
        # 'jit': True, # JIT compilation can speed up repeated calls
        # 'compiler': 'shell', 
        # 'jit_options': {'verbose': False}
    } 
    solver = nlpsol("nlpsol", "ipopt", nlp, opts)
    
    return [solver, shooting_vars, parameter_vars]


# --- EKF Functions ---

def generate_EKF_covariances_rov(params):
    """Generates covariance matrices Q, R, P for the EKF."""
    nx = params['nx']
    ntheta = params['ntheta']
    nw = params['nw'] # Process noise dimension
    nv = params['nv'] # Measurement noise dimension

    # Process noise covariance Q (diagonal, based on bounds)
    # Tune these values based on expected noise/uncertainty
    q_pos = params.get('bound_w_pos', 0.01)**2 
    q_vel = params.get('bound_w_vel', 0.05)**2
    q_theta = 0.01**2 # Tuning parameter for parameter uncertainty increase
    
    Q_diag = np.concatenate([
        q_pos * np.ones(4), # x, y, z, psi state noise variance
        q_vel * np.ones(4), # u, v, w, r state noise variance
        q_theta * np.ones(ntheta) # Parameter "process noise" variance
    ])
    Q_ekf = np.diag(Q_diag)

    # Measurement noise covariance R (diagonal, based on bounds)
    r_pos = params.get('bound_v_pos', 0.1)**2
    r_psi = params.get('bound_v_psi', 0.02)**2
    if nv == 4:
        R_diag = np.array([r_pos, r_pos, r_pos, r_psi])
    else:
        R_diag = (0.1**2) * np.ones(nv) # Default if ny != 4
    R_ekf = np.diag(R_diag)

    # Initial state/parameter covariance P
    p_state = 0.1**2 # Initial state uncertainty variance
    p_theta = 1.0**2 # Initial parameter uncertainty variance (larger)
    P_diag = np.concatenate([
        p_state * np.ones(nx),
        p_theta * np.ones(ntheta)
    ])
    P_ekf = np.diag(P_diag)
    
    return Q_ekf, R_ekf, P_ekf


def extended_discrete_dynamics_rov(x_tilde, u, w, params):
    """
    Extended dynamics for EKF (state + parameters).
    Assumes parameters are constant (zero derivative).
    w affects only the state part.
    """
    nx = params['nx']
    ntheta = params['ntheta']
    Ts = params['Ts']
    
    state = x_tilde[:nx]
    theta = x_tilde[nx:] if ntheta > 0 else None
    
    # Process noise for state part (ensure correct dimension)
    w_state = w[:nx] if w.shape[0] >= nx else np.zeros(nx) 
    
    state_plus = discrete_dynamics_rov(state, u, w_state, params, Ts, theta=theta)
    
    # Parameters are assumed constant
    theta_plus = theta if ntheta > 0 else [] 
    
    return vertcat(state_plus, theta_plus)


def extended_measurement_function_rov(x_tilde, v, params):
    """Extended measurement function for EKF."""
    nx = params['nx']
    # Measurement depends only on the state part
    return measurement_function_rov(x_tilde[:nx], v, params)


def get_Jacobian_f_rov(params):
    """Computes the CasADi function for the Jacobian of the extended dynamics."""
    nx = params['nx']
    nu = params['nu']
    ntheta = params['ntheta']
    nw = params['nw'] # Process noise dim used in extended_discrete_dynamics_rov

    x_tilde_sym = ca.MX.sym('x_tilde', nx + ntheta)
    u_sym = ca.MX.sym('u', nu)
    # Jacobian is wrt x_tilde, assuming w=0 for linearization
    w_zero = ca.MX.zeros(nw) 
    
    f_ext_expr = extended_discrete_dynamics_rov(x_tilde_sym, u_sym, w_zero, params)
    jac_f_expr = jacobian(f_ext_expr, x_tilde_sym)
    
    return Function('Jacobian_f', [x_tilde_sym, u_sym], [jac_f_expr])


def get_Jacobian_h_rov(params):
    """Computes the CasADi function for the Jacobian of the extended measurement function."""
    nx = params['nx']
    ntheta = params['ntheta']
    nv = params['nv']

    x_tilde_sym = ca.MX.sym('x_tilde', nx + ntheta)
    # Jacobian is wrt x_tilde, assuming v=0 for linearization
    v_zero = ca.MX.zeros(nv) 
    
    h_ext_expr = extended_measurement_function_rov(x_tilde_sym, v_zero, params)
    jac_h_expr = jacobian(h_ext_expr, x_tilde_sym)
    
    return Function('Jacobian_h', [x_tilde_sym], [jac_h_expr])


def EKF_update_rov(x_tilde_dm, u, P, y, Q, R, Jacobian_f, Jacobian_h, params):
    """Performs the EKF prediction and update step."""
    nx = params['nx']
    ntheta = params['ntheta']
    nv = params['nv']
    nw = params['nw'] 

    # Ensure inputs are CasADi DM types
    x_tilde_dm = ca.DM(x_tilde_dm)
    u_dm = ca.DM(u)
    P_dm = ca.DM(P)
    y_dm = ca.DM(y)
    Q_dm = ca.DM(Q) # Size (nx+ntheta, nx+ntheta)
    R_dm = ca.DM(R) # Size (ny, ny)

    # Evaluate Jacobians
    A = Jacobian_f(x_tilde_dm, u_dm)
    H = Jacobian_h(x_tilde_dm)

    # Prediction step
    zero_w = np.zeros(nw) # Use zero noise for prediction
    x_pred = extended_discrete_dynamics_rov(x_tilde_dm, u_dm, zero_w, params)
    P_pred = A @ P_dm @ A.T + Q_dm 

    # Measurement update step
    zero_v = np.zeros(nv) # Use zero noise for measurement prediction
    y_pred = extended_measurement_function_rov(x_pred, zero_v, params)
    S = H @ P_pred @ H.T + R_dm
    # Use pseudo-inverse for robustness if S is ill-conditioned
    K = P_pred @ H.T @ ca.pinv(S) 
    x_update = x_pred + K @ (y_dm - y_pred)
    P_update = (ca.DM.eye(nx + ntheta) - K @ H) @ P_pred

    return np.array(x_update).flatten(), np.array(P_update) # Return numpy arrays

def generate_mhe_bounds(shooting_vars, params):
    """Generates lower and upper bounds for the MHE decision variables."""
    lbx = shooting_vars(-inf)
    ubx = shooting_vars(inf)

    nx = params['nx']
    nw = params['nw']
    nv = params['nv']
    ntheta = params['ntheta']

    # Bounds on process noise (w_est) - Slightly increased
    bound_w_pos = params.get('bound_w_pos', 0.01) * 1.5 
    bound_w_vel = params.get('bound_w_vel', 0.05) * 1.5
    w_bounds = np.concatenate([bound_w_pos * np.ones(4), bound_w_vel * np.ones(4)]) # Assuming state order
    lbx["w_est"] = -w_bounds
    ubx["w_est"] = w_bounds
    
    # Bounds on measurement noise (v_est) - Slightly increased
    bound_v_pos = params.get('bound_v_pos', 0.1) * 1.5
    bound_v_psi = params.get('bound_v_psi', 0.02) * 1.5
    if nv == 4: # Assuming [x,y,z,psi] measurement noise
        v_bounds = np.array([bound_v_pos, bound_v_pos, bound_v_pos, bound_v_psi])
    else: # Default if different measurement setup
        v_bounds = 0.5 * 1.5 * np.ones(nv) 
    lbx["v_est"] = -v_bounds
    ubx["v_est"] = v_bounds
    
    # Optional: Add bounds on estimated states or parameters if needed
    # Only set theta bounds if ntheta > 0 AND theta_est exists in the structure
    if ntheta > 0 and "theta_est" in shooting_vars.keys():
        # Example: Bounds for drag coefficients (should be negative)
        # Ensure bounds match the number of parameters (ntheta)
        lbx["theta_est"] = -100.0 * np.ones(ntheta) # Lower bound for drag params
        ubx["theta_est"] = -0.01 * np.ones(ntheta)  # Upper bound (negative, away from zero)

    # Optional: Bounds on states (e.g., velocity limits)
    # ubx["x_est", :, 4:8] = 5.0 # Limit body velocities to 5 m/s or rad/s
    # lbx["x_est", :, 4:8] = -5.0 
    
    return lbx, ubx

# Example usage (for testing)
if __name__ == '__main__':
    # Assumes model.py is in the same directory
    from model import get_model_parameters

    params = get_model_parameters()
    params['Ts'] = 0.05
    params['eta'] = 0.95
    params['prior_weighting'] = 1.0
    # params['ntheta'] = 8 # Uncomment to test parameter estimation setup
    
    M = 10 # Example horizon length

    print("Setting up MHE problem...")
    solver, shooting, parameters = setup_mhe_problem(M, params)
    print("MHE problem setup complete.")
    print("Solver type:", solver.info()['nlpsol'])
    
    print("\nGenerating bounds...")
    lbx, ubx = generate_mhe_bounds(shooting, params)
    print("Bounds generated.")
    # print("Lower bounds (partial):", lbx[:10]) # Print first few bounds
    # print("Upper bounds (partial):", ubx[:10])

    print("\nShooting variables structure:")
    shooting.print_summary()
    
    print("\nParameter variables structure:")
    parameters.print_summary()


def setup_mhe_standard_rov(M, fixed_theta, params):
    """
    Generates the CasADi NLP problem structure for standard MHE (fixed parameters).

    Args:
        M: Estimation horizon length.
        fixed_theta: The fixed parameter vector to use in the dynamics.
        params: Dictionary of model and MHE parameters.

    Returns:
        A list containing: [solver, shooting_vars, parameter_vars]
        Returns None for items if setup fails (e.g., fixed_theta format wrong).
    """
    eta = params.get('eta', 0.9) # Forgetting factor
    nx = params['nx']
    ny = params['ny']
    nw = params['nw']
    nv = params['nv']
    nu = params['nu']
    # ntheta = params['ntheta'] # Not used here
    Ts = params['Ts']

    # Ensure fixed_theta is a CasADi DM
    try:
        fixed_theta_dm = ca.DM(fixed_theta)
        if fixed_theta_dm.shape[0] != params['ntheta'] or fixed_theta_dm.shape[1] != 1:
             raise ValueError(f"fixed_theta shape mismatch: expected ({params['ntheta']}, 1), got {fixed_theta_dm.shape}")
    except Exception as e:
        print(f"Error converting fixed_theta to DM: {e}")
        return None, None, None

    # --- Define symbolic variables (excluding theta_est) ---
    # --- Define symbolic variables (excluding theta_est) ---
    shooting_vars_list = [
        entry("x_est", repeat=M + 1, shape=(nx, 1)),
        # entry("y_est", repeat=M, shape=(ny, 1)),   # REMOVED - Estimated measurements
        entry("w_est", repeat=M, shape=(nw, 1)),     
        entry("v_est", repeat=M, shape=(nv, 1))      
    ]
    shooting_vars = struct_symMX(shooting_vars_list)

    # --- Define symbolic parameters (excluding theta_prior) ---
    parameter_vars_list = [
        entry("y_meas", repeat=M, shape=(ny, 1)),    
        entry("u", repeat=M, shape=(nu, 1)),         
        entry("x_prior", shape=(nx, 1)),             
    ]
    parameter_vars = struct_symMX(parameter_vars_list)
    
    # --- Define Weighting matrices (same as parametric MHE) ---
    Q_w = np.eye(nw)
    bound_w_pos = params.get('bound_w_pos', 0.01) 
    bound_w_vel = params.get('bound_w_vel', 0.05)
    Q_w[0:4, 0:4] /= (bound_w_pos * bound_w_pos)
    Q_w[4:8, 4:8] /= (bound_w_vel * bound_w_vel)
    
    Q_v = np.eye(nv)
    bound_v_pos = params.get('bound_v_pos', 0.1)  
    bound_v_psi = params.get('bound_v_psi', 0.02) 
    if nv == 4: 
        Q_v[0:3, 0:3] /= (bound_v_pos * bound_v_pos)
        Q_v[3, 3] /= (bound_v_psi * bound_v_psi)
    else: 
         Q_v /= (0.05 * 0.05) 
    
    prior_weighting_val = params.get('prior_weighting', 1.0)
    P_x = np.eye(nx) * prior_weighting_val
    
    # --- Build Objective Function (excluding theta prior term) ---
    objective = 0
    
    # Add prior term for state
    delta_x_prior = shooting_vars["x_est"][0] - parameter_vars["x_prior"]
    objective += pow(eta, M) * (delta_x_prior.T @ P_x @ delta_x_prior)

    # Add stage costs for noise
    for k in range(M):
        discount_factor = pow(eta, M - k - 1)
        objective += discount_factor * (shooting_vars["w_est"][k].T @ Q_w @ shooting_vars["w_est"][k])
        objective += discount_factor * (shooting_vars["v_est"][k].T @ Q_v @ shooting_vars["v_est"][k])
        
    # --- Build Constraints (using fixed_theta_dm) ---
    constraints = []
    for k in range(M):
        # Dynamics constraints - pass fixed_theta_dm
        constraints.append(constraint_dynamics_mhe(shooting_vars["x_est"][k+1], 
                                                   shooting_vars["x_est"][k], 
                                                   parameter_vars["u"][k], 
                                                   shooting_vars["w_est"][k], 
                                                   fixed_theta_dm, params, Ts)) # Pass fixed theta here
        
        # Measurement constraints
        constraints.append(constraint_measurement_mhe(parameter_vars["y_meas"][k], 
                                                      shooting_vars["x_est"][k], 
                                                      shooting_vars["v_est"][k], params))
        
    # --- Create NLP Solver ---
    nlp = {"x": shooting_vars, "p": parameter_vars, "f": objective, "g": vertcat(*constraints)}
    
    # Use same solver options as parametric MHE
    opts = {
        "ipopt.print_level": params.get("ipopt_print_level", 0), 
        "print_time": params.get("print_time", False), 
        'ipopt.max_iter': params.get("ipopt_max_iter", 300), 
        'ipopt.tol': params.get("ipopt_tol", 1e-6),
        'ipopt.acceptable_tol': params.get("ipopt_acceptable_tol", 1e-3), 
    } 
    solver = nlpsol("nlpsol", "ipopt", nlp, opts)
    
    return [solver, shooting_vars, parameter_vars]
