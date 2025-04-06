import mujoco
import mujoco.viewer
import mujoco
import numpy as np
import time

dt: float = 0.002

def main() -> None:
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)

    key_name = "home"
    key_id = model.key(key_name).id

    site_name = "attachment_site"
    site_id = model.site(site_name).id
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            calculated_center = np.array((np.array(data.xpos[11]) + np.array(data.xpos[10]))/2)
            print(data.site(site_id).xpos, calculated_center)
            # print("XPOS: ", data.xpos[11], data.xpos[10])



            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()