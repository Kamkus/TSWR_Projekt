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

    mpc = MPC(data)
    contoller = mpc.controller

    key_name = "home"
    key_id = model.key(key_name).id

    # site_name = "attachment_site"
    # site_id = model.site(site_name).id

    x0 = np.zeros((14, 1))

    graphics = do_mpc.graphics.Graphics(mpc.controller.data)

    joint_states = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    ee_pos_mujoco = []      # Rzeczywista pozycja końcówki z MuJoCo
    ee_rot_mujoco = []      # Rzeczywista orientacja końcówki z MuJoCo
    step_counter = 0
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        x0[:7] = data.qpos[:7].reshape(-1, 1)
        x0[7:] = data.qvel[:7].reshape(-1, 1)
        contoller.x0 = x0
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

    fig, ax = plt.subplots(6, 1, figsize=(17, 15))
    
    graphics.add_line(var_type='_tvp', var_name='target_pos', axis=ax[0])
    ax[0].set_title('Docelowe pozycje końcówki (target_pos)')
    ax[0].set_ylabel('Pozycja X [m]')
    ax[0].grid(True)
    ax[0].legend(['Target X', 'Target Y', 'Target Z'])
    
    # Wykres rzeczywistej pozycji końcówki (ee_pos)
    graphics.add_line(var_type='_aux', var_name='ee_pos', axis=ax[1])
    ax[1].set_title('Rzeczywiste pozycje końcówki (MPC)')
    ax[1].set_ylabel('Pozycja [m]')
    ax[1].grid(True)
    ax[1].legend(['EE X', 'EE Y', 'EE Z'])


    ee_pos_mujoco = np.array(ee_pos_mujoco)
    ee_rot_mujoco = np.array(ee_rot_mujoco)

    ax[2].plot(time_steps, ee_pos_mujoco[:, 0], 'b-', label='MuJoCo X', linewidth=2)
    ax[2].plot(time_steps, ee_pos_mujoco[:, 1], 'g-', label='MuJoCo Y', linewidth=2)
    ax[2].plot(time_steps, ee_pos_mujoco[:, 2], 'r-', label='MuJoCo Z', linewidth=2)
    ax[2].set_title('Rzeczywiste pozycje końcówki (MuJoCo)')
    ax[2].set_ylabel('Pozycja [m]')
    ax[2].grid(True)
    ax[2].legend()

    ax[3].plot(time_steps, ee_rot_mujoco[:, 0], 'b-', label='MuJoCo Rot W', linewidth=2)
    ax[3].plot(time_steps, ee_rot_mujoco[:, 1], 'g-', label='MuJoCo X', linewidth=2)
    ax[3].plot(time_steps, ee_rot_mujoco[:, 2], 'r-', label='MuJoCo Y', linewidth=2)
    ax[3].plot(time_steps, ee_rot_mujoco[:, 3], 'y-', label='MuJoCo Z', linewidth=2)
    ax[3].set_title('Rzeczywista rotacja końcówki (MuJoCo)')
    ax[3].grid(True)
    ax[3].legend()

    graphics.add_line(var_type='_aux', var_name='ee_rot', axis=ax[4])
    ax[4].set_title('Rzeczywista rotacja końcówki (MPC)')
    ax[4].grid(True)
    ax[4].legend(['ROT W', 'ROT X', 'ROT Y', 'ROT Z'])

    graphics.add_line(var_type='_tvp', var_name='target_rot', axis=ax[5])
    ax[5].set_title('Docelowe rotacja końcówki')
    ax[5].set_ylabel('Pozycja X [m]')
    ax[5].grid(True)
    ax[5].legend(['Target W', 'Target X', 'Target Y', 'Target Z'])


    plt.show()
    


if __name__ == "__main__":
    main()