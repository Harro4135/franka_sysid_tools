from setuptools import find_packages, setup


package_name = "franka_sysid_tools"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/franka_sysid_topic_map.yaml"]),
    ],
    install_requires=["setuptools", "numpy", "casadi", "mujoco"],
    zip_safe=True,
    maintainer="Harro4135",
    maintainer_email="oliver.harrison@mail.utoronto.ca",
    description="MoveIt 2 tools for collecting Franka telemetry for Isaac Sim System Identification.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "franka_sysid_collect = franka_sysid_tools.franka_sysid_collect:main",
            "franka_sysid_collect_v2 = franka_sysid_tools.franka_sysid_collect_v2:main",
            "franka_sysid_collect_v3 = franka_sysid_tools.franka_sysid_collect_v3:main",
            "franka_sysid_optimize_v3_offline = franka_sysid_tools.franka_sysid_optimize_v3_offline:main",
            "franka_sysid_sim_mujoco = franka_sysid_tools.franka_sysid_sim_mujoco:main",
            "so101_sysid_collect_v2 = franka_sysid_tools.so101_sysid_collect_v2:main",
        ],
    },
)
