# V550 Ackermann URDF variants

This directory contains the two V550 Ackermann URDF variants used by
this project. The chassis uses front-wheel steering and rear-wheel drive.

- `rl_training.urdf`: lightweight geometry used by the original reinforcement-learning training scene.
- `industrial_navigation.urdf`: detailed visual model used by the dynamic industrial navigation scene.

The ROS packages, launch commands, and runtime assets use the
`v550_ackermann_*` naming scheme.
