#!/usr/bin/env python
# coding: utf-8

"""
Main script to run and compare different MHE strategies and EKF for BlueROV2.
Based on the structure of the original parametric-mhe car example.
MODIFIED TO ONLY RUN AND PLOT EKF.
"""

import numpy as np
import casadi as ca
from casadi import vertcat, horzcat, DM, Function, jacobian, inv # Added DM, Function, jacobian, inv
import time
import os
import matplotlib.pyplot as plt # Import plt for placeholder plot

# Import local modules
from model import get_model_parameters, discrete_dynamics_rov, measurement_function_rov
from simulation import generate_process_noise, generate_measurement_noise, generate_inputs_rov, simulate_system_rov
from estimation import (setup_mhe_problem, generate_mhe_bounds,
                        setup_mhe_standard_rov, generate_EKF_covariances_rov,
                        get_Jacobian_f_rov, get_Jacobian_h_rov, EKF_update_rov)
# from plotting import plot_estimation_results, plot_comparison_results_rov # Commented out comparison plot

# --- Helper Functions ---

# Re-add initialize_state_trajectory function (was in simulation.py originally in run_mhe)
def initialize_state_trajectory(x0, u_seq, theta_init, params, M):
    """Generates an initial state trajectory guess for warm-starting MHE."""
    nx = params['nx']
    nw = params['nw']
    Ts = params['Ts']
    ntheta = params['ntheta']

    # Create a simple simulator function for initialization
    state_ca = ca.MX.sym('state', nx)
    action_ca = ca.MX.sym('action', params['nu'])
    disturbance_ca = ca.MX.sym('disturbance', nw)

    if ntheta > 0:
        theta_ca = ca.MX.sym('theta', ntheta)
        # Use the correct signature for discrete_dynamics_rov
        f_discrete_expr = discrete_dynamics_rov(state_ca, action_ca, disturbance_ca, params, Ts, theta=theta_ca)
        f_init_sim = ca.Function('f_init_sim', [state_ca, action_ca, disturbance_ca, theta_ca], [f_discrete_expr])
    else:
        f_discrete_expr = discrete_dynamics_rov(state_ca, action_ca, disturbance_ca, params, Ts)
        f_init_sim = ca.Function('f_init_sim', [state_ca, action_ca, disturbance_ca], [f_discrete_expr])

    # Simulate forward using zero noise
    initial_states = ca.DM.zeros(nx, M + 1)
    initial_states[:, 0] = x0
    current_state = DM(x0) # Ensure current_state starts as DM

    for k_sim in range(M):
        action = u_seq[:, k_sim] if k_sim < u_seq.shape[1] else u_seq[:, -1] # Use last input if needed
        if ntheta > 0:
            current_state = f_init_sim(current_state, action, np.zeros(nw), theta_init)
        else:
            current_state = f_init_sim(current_state, action, np.zeros(nw))
        initial_states[:, k_sim+1] = current_state

    return initial_states


# Renamed back to initialize_x_rov to match car example usage pattern
# This function simulates the *next* state based on the *last* state of the provided trajectory
# and is used to update the warm-start trajectory for the *next* MHE step.
def initialize_x_rov(initial_states_dm, action, theta, params):
    """
    Simulates the last state of a trajectory one step forward using the provided action and theta.
    Used to update the warm-start trajectory for the *next* MHE step.

    Args:
        initial_states_dm: CasADi DM matrix of state trajectory (nx, N) from previous step's solution/guess.
        action: Action vector for the *current* step (k) (nu,)
        theta: Parameter vector from the *previous* step (k-1) (ntheta,)
        params: Model parameters dictionary

    Returns:
        Shifted state trajectory including the new predicted state (CasADi DM). Shape (nx, N).
    """
    nx = params['nx']
    nu = params['nu']
    nw = params['nw']
    ntheta = params['ntheta']
    Ts = params['Ts']

    # Create CasADi function for one-step simulation if not already done
    state_ca = ca.MX.sym('state', nx)
    action_ca = ca.MX.sym('action', nu)
    disturbance_ca = ca.MX.sym('disturbance', nw) # Use nw here

    if ntheta > 0:
        theta_ca = ca.MX.sym('theta', ntheta)
        f_discrete_expr = discrete_dynamics_rov(state_ca, action_ca, disturbance_ca, params, Ts, theta=theta_ca)
        f_sim_step = ca.Function('f_sim_step', [state_ca, action_ca, disturbance_ca, theta_ca], [f_discrete_expr])
        # Simulate next state using zero noise and the *last* state of the input trajectory
        x_plus = f_sim_step(initial_states_dm[:, -1], action, np.zeros(nw), theta)
    else:
        f_discrete_expr = discrete_dynamics_rov(state_ca, action_ca, disturbance_ca, params, Ts)
        f_sim_step = ca.Function('f_sim_step', [state_ca, action_ca, disturbance_ca], [f_discrete_expr])
        # Simulate next state using zero noise
        x_plus = f_sim_step(initial_states_dm[:, -1], action, np.zeros(nw))

    # Shift trajectory (drop oldest state) and append new predicted state
    initial_states_shifted = horzcat(initial_states_dm[:, 1:], x_plus)
    return initial_states_shifted

# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    params = get_model_parameters() # ntheta is now 8
    model_name = "BlueROV2"

    # Simulation settings
    nsim = 400       # Number of simulation steps
    Ts = 0.05        # Sampling time (ensure consistency)
    params['Ts'] = Ts

    # MHE settings (still needed for setup, but loop is commented)
    M = 15           # Estimation horizon length
    params['eta'] = 0.95 # Forgetting factor
    params['prior_weighting'] = 1.0 # Weighting for prior info
    bool_start_with_fie = True # Use FIE for initialization phase

    # Initial conditions
    x0_true = np.zeros(params['nx'])
    x0_mhe = np.zeros(params['nx']) # Initial state guess for MHE/EKF

    # Define true parameters (from model dict)
    theta_true = np.array([
        params['X_u'], params['Y_v'], params['Z_w'], params['N_r'],
        params['X_uc'], params['Y_vc'], params['Z_wc'], params['N_rc']
    ])
    # Define initial parameter guess (e.g., 50% of true values)
    theta0_mhe = theta_true * 0.5

    # --- Simulation ---
    print(f"--- Starting {model_name} Simulation ---")
    np.random.seed(42) # for reproducibility
    w_true = generate_process_noise(params, nsim)
    v_true = generate_measurement_noise(params, nsim)
    u_traj = generate_inputs_rov(nsim, params)

    x_true, y_meas = simulate_system_rov(x0_true, u_traj, w_true, v_true, theta_true, params, nsim)
    print("Simulation complete.")

    # --- Estimator Setup ---
    nx = params['nx']
    ntheta = params['ntheta']
    nv = params['nv']
    nw = params['nw']

    # History storage for all methods (keep structure for EKF)
    results = {
        # 'mhe_theta_0': {'x': np.zeros((nx, nsim)), 'theta': np.zeros((ntheta, nsim)), 'time': np.zeros(nsim-1)},
        # 'mhe_theta_M': {'x': np.zeros((nx, nsim)), 'theta': np.zeros((ntheta, nsim)), 'time': np.zeros(nsim-1)},
        # 'mhe_standard': {'x': np.zeros((nx, nsim)), 'time': np.zeros(nsim-1)},
        'ekf': {'x': np.zeros((nx, nsim + 1)), 'theta': np.zeros((ntheta, nsim + 1)), 'time': np.zeros(nsim)} # Adjusted size for k=0 to nsim
    }

    # Initialize histories at k=0
    # results['mhe_theta_0']['x'][:, 0] = x0_mhe
    # results['mhe_theta_0']['theta'][:, 0] = theta0_mhe
    # results['mhe_theta_M']['x'][:, 0] = x0_mhe
    # results['mhe_theta_M']['theta'][:, 0] = theta0_mhe
    # results['mhe_standard']['x'][:, 0] = x0_mhe
    results['ekf']['x'][:, 0] = x0_mhe # Use x0_mhe as initial EKF guess
    results['ekf']['theta'][:, 0] = theta0_mhe # Use theta0_mhe as initial EKF guess

    # Generate MHE/FIE solvers (Commented out - Not needed for EKF only)
    # print(f"--- Setting up MHE/FIE Solvers (Horizon 1 to {M}) ---")
    # mhe_solvers = []
    # mhe_shoot_vars = []
    # mhe_param_vars = []
    # mhe_lbx = []
    # mhe_ubx = []
    # mhe_standard_solvers = []
    # mhe_standard_shoot_vars = []
    # mhe_standard_param_vars = []
    # mhe_standard_lbx = []
    # mhe_standard_ubx = []
    #
    # for k_horizon in range(1, M + 1):
    #     # Parametric MHE setup
    #     solver, shooting, parameters = setup_mhe_problem(k_horizon, params)
    #     lbx, ubx = generate_mhe_bounds(shooting, params)
    #     mhe_solvers.append(solver)
    #     mhe_shoot_vars.append(shooting)
    #     mhe_param_vars.append(parameters)
    #     mhe_lbx.append(lbx)
    #     mhe_ubx.append(ubx)
    #
    #     # Standard MHE setup (using fixed theta0_mhe)
    #     solver_std, shooting_std, params_std = setup_mhe_standard_rov(k_horizon, theta0_mhe, params)
    #     if solver_std: # Check if implemented
    #          lbx_std, ubx_std = generate_mhe_bounds(shooting_std, params) # This assumes bounds structure is compatible
    #          mhe_standard_solvers.append(solver_std)
    #          mhe_standard_shoot_vars.append(shooting_std)
    #          mhe_standard_param_vars.append(params_std)
    #          mhe_standard_lbx.append(lbx_std)
    #          mhe_standard_ubx.append(ubx_std)
    #     else: # Append placeholders if not implemented
    #          mhe_standard_solvers.append(None)
    #          mhe_standard_shoot_vars.append(None)
    #          mhe_standard_param_vars.append(None)
    #          mhe_standard_lbx.append(None)
    #          mhe_standard_ubx.append(None)


    # EKF Setup
    print("--- Setting up EKF ---")
    Q_ekf, R_ekf, P_ekf = generate_EKF_covariances_rov(params)
    Jacobian_f = get_Jacobian_f_rov(params)
    Jacobian_h = get_Jacobian_h_rov(params)
    current_P_ekf = P_ekf # Initial covariance

    # Initial warm-start states (Commented out - Not needed for EKF only)
    # zero_u_traj_M = np.zeros((params['nu'], M))
    # initial_states_theta_0_dm = initialize_state_trajectory(x0_mhe, zero_u_traj_M, theta0_mhe, params, M)
    # initial_states_theta_M_dm = initial_states_theta_0_dm
    # initial_states_standard_dm = initial_states_theta_0_dm

    # --- Estimation Loop ---
    print("--- Starting Estimation Loop (EKF Only) ---")
    total_start_time = time.time()

    for k in range(nsim): # Loop from 0 to nsim-1 for EKF update
        loop_start_time = time.time()
        # current_M = min(k + 1, M) # Effective horizon length for FIE phase (Not needed)
        # mhe_idx = current_M - 1 # Index for solver lists (0 to M-1) (Not needed)

        # --- EKF Update ---
        ekf_start_time = time.time()
        # Use state/theta from previous step k
        x_tilde_ekf = np.concatenate([results['ekf']['x'][:, k], results['ekf']['theta'][:, k]])
        # Use input u(k) and measurement y(k)
        x_tilde_plus, current_P_ekf = EKF_update_rov(
            x_tilde_ekf, u_traj[:, k], current_P_ekf, y_meas[:, k], Q_ekf, R_ekf, Jacobian_f, Jacobian_h, params
        )
        # Store result for step k+1
        results['ekf']['x'][:, k+1] = x_tilde_plus[:nx]
        results['ekf']['theta'][:, k+1] = x_tilde_plus[nx:]
        results['ekf']['time'][k] = time.time() - ekf_start_time

        # --- MHE/FIE Updates (Commented Out) ---
        # if bool_start_with_fie or k >= M:
            # ... (MHE code commented out) ...
        # else: # FIE phase skipped, just copy initial state/theta
             # ... (Copying commented out) ...

        if (k+1) % 50 == 0: # Adjust print frequency
            print(f"Step {k+1}/{nsim} complete. Total time: {time.time() - loop_start_time:.4f}s")

    total_end_time = time.time()
    print(f"--- Estimation Loop Finished ---")
    print(f"Total time: {total_end_time - total_start_time:.2f}s")

    # --- Plotting ---
    print("--- Plotting EKF Results ---")
    time_vec = np.arange(nsim + 1) * Ts

    # Plot EKF States vs True States
    fig_ekf_states, axs_ekf_states = plt.subplots(nx, 1, figsize=(10, 2*nx), sharex=True)
    state_labels = ['x', 'y', 'z', 'psi', 'u', 'v', 'w', 'r']
    for i in range(nx):
        axs_ekf_states[i].plot(time_vec, x_true[i, :], 'k-', label='True')
        axs_ekf_states[i].plot(time_vec, results['ekf']['x'][i, :], 'b--', label='EKF Estimate')
        axs_ekf_states[i].set_ylabel(f'{state_labels[i]}')
        axs_ekf_states[i].grid(True)
        if i == 0:
            axs_ekf_states[i].legend()
    axs_ekf_states[-1].set_xlabel('Time (s)')
    fig_ekf_states.suptitle('EKF State Estimation vs True States')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout

    # Plot EKF Parameters vs True Parameters
    if ntheta > 0:
        fig_ekf_params, axs_ekf_params = plt.subplots(ntheta, 1, figsize=(10, 2*ntheta), sharex=True)
        param_labels = ['X_u', 'Y_v', 'Z_w', 'N_r', 'X_uc', 'Y_vc', 'Z_wc', 'N_rc'] # Assuming this order
        if not isinstance(axs_ekf_params, np.ndarray): # Handle case where ntheta=1
             axs_ekf_params = [axs_ekf_params]
        for i in range(ntheta):
            axs_ekf_params[i].plot(time_vec, np.ones(nsim + 1) * theta_true[i], 'k-', label='True')
            axs_ekf_params[i].plot(time_vec, results['ekf']['theta'][i, :], 'g--', label='EKF Estimate')
            axs_ekf_params[i].set_ylabel(f'{param_labels[i]}')
            axs_ekf_params[i].grid(True)
            if i == 0:
                axs_ekf_params[i].legend()
        axs_ekf_params[-1].set_xlabel('Time (s)')
        fig_ekf_params.suptitle('EKF Parameter Estimation vs True Parameters')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout

    # Show plots
    plt.show()

    # Call the comparison plotting function (Commented Out)
    # plot_comparison_results_rov(x_true, y_meas, theta_true, theta0_mhe, results, params, nsim, M)

    print("--- Script Finished ---")
