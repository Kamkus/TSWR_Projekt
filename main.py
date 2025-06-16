import mujoco
import mujoco.viewer
import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
import do_mpc


from MPC.mpc import MPC

dt: float = 0.02


def get_end_effector_position_mujoco(model, data):
    body_id = model.body("attachment").id
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def main() -> None:
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)


    # Ustawianie pozycji początkowej
    data.qpos[0] = 1.63e-08  # joint1
    data.qpos[1] = -0.171    # joint2
    data.qpos[2] = 4.6e-08   # joint3
    data.qpos[3] = -2.05     # joint4
    data.qpos[4] = -0.000321 # joint5
    data.qpos[5] = 1.85      # joint6
    data.qpos[6] = -0.785    # joint7
    
    data.ctrl[0] = 0        # actuator1
    data.ctrl[1] = -0.176   # actuator2
    data.ctrl[2] = 0        # actuator3
    data.ctrl[3] = -2.05    # actuator4
    data.ctrl[4] = 0        # actuator5
    data.ctrl[5] = 1.85     # actuator6
    data.ctrl[6] = -0.785   # actuator7
    
    mujoco.mj_forward(model, data)

    mpc = MPC(data)
    contoller = mpc.controller
    x0 = np.zeros((14, 1))

    graphics = do_mpc.graphics.Graphics(mpc.controller.data)

    joint_states = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    ee_pos_mujoco = []      
    ee_rot_mujoco = []

    step_counter = 0

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False
    ) as viewer:
        # mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        x0[:7] = data.qpos[:7].reshape(-1, 1)
        x0[7:] = data.qvel[:7].reshape(-1, 1)
        contoller.x0 = x0
        contoller.u0 = data.ctrl[:7].reshape(-1, 1)
        contoller.set_initial_guess()

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)

            q_current = data.qpos[:7]
            qd_current = data.qvel[:7]
            x0[:7] = q_current.reshape(-1, 1)[:7]
            x0[7:] = qd_current.reshape(-1, 1)[:7]

            for i in range(7):
                joint_states[i].append(q_current[i])

            ee_pos_real, ee_rot_real = get_end_effector_position_mujoco(model, data)
            ee_rot_mujoco.append(ee_rot_real.copy())
            ee_pos_mujoco.append(ee_pos_real.copy())

            u0 = contoller.make_step(x0)

            data.ctrl[:7] = u0.flatten()

            viewer.sync()
            time_until_next_step = model.opt.timestep*10 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            step_counter += 1
    time_steps = np.arange(len(joint_states[0])) * model.opt.timestep
    plt.figure()
    
    for i in range(7):
        plt.plot(time_steps, joint_states[i], label=f'Joint {i}')
    plt.xlabel('Czas [s]')
    plt.ylabel('Pozycja [rad]')
    plt.title(f'Joint {i}')
    plt.legend()    

    fig, ax = plt.subplots(3, 1, figsize=(17, 15))
    
    graphics.add_line(var_type='_tvp', var_name='target_pos', axis=ax[0])
    ax[0].set_title('Docelowe pozycje końcówki (target_pos)')
    ax[0].set_ylabel('Pozycja X')
    ax[0].grid(True)
    ax[0].legend(['Target X', 'Target Y', 'Target Z'])
    
    graphics.add_line(var_type='_aux', var_name='ee_pos', axis=ax[1])
    ax[1].set_title('Rzeczywiste pozycje końcówki (MPC)')
    ax[1].set_ylabel('Pozycja')
    ax[1].grid(True)
    ax[1].legend(['EE X', 'EE Y', 'EE Z'])


    ee_pos_mujoco = np.array(ee_pos_mujoco)
    ee_rot_mujoco = np.array(ee_rot_mujoco)

    ax[2].plot(time_steps, ee_pos_mujoco[:, 0], 'b-', label='MuJoCo X', linewidth=2)
    ax[2].plot(time_steps, ee_pos_mujoco[:, 1], 'g-', label='MuJoCo Y', linewidth=2)
    ax[2].plot(time_steps, ee_pos_mujoco[:, 2], 'r-', label='MuJoCo Z', linewidth=2)
    ax[2].set_title('Rzeczywiste pozycje końcówki (MuJoCo)')
    ax[2].set_ylabel('Pozycja')
    ax[2].grid(True)
    ax[2].legend()


    fig, axcomp = plt.subplots(1, 1, figsize=(17, 15))

    graphics.add_line(var_type='_aux', var_name='ee_pos', axis=axcomp)
    axcomp.plot(time_steps, ee_pos_mujoco[:, 0], linewidth=2)
    axcomp.plot(time_steps, ee_pos_mujoco[:, 1], linewidth=2)
    axcomp.plot(time_steps, ee_pos_mujoco[:, 2], linewidth=2)
    axcomp.set_title('Rzeczywiste pozycje końcówki (MPC i MuJoCo)')
    axcomp.set_ylabel('Pozycja [m]')
    axcomp.grid(True)
    axcomp.legend(['MPC X', 'MPC Y', 'MPC Z', 'MuJoCo X', 'MuJoCo Y', 'MuJoCo Z'])

    plt.show()
    


if __name__ == "__main__":
    main()