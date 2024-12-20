# IGVC-Robot-Barbie-Jeep
This project focuses on developing an autonomous vehicle for the Intelligent Ground Vehicle Competition (IGVC). The vehicle integrates lane-keeping and object-avoidance algorithms, coupled with precise motor control, to navigate complex environments.

### Features:
- **Lane Detection**: Real-time lane detection algorithms for staying within designated paths, even when only one lane is visible.
- **Object Avoidance**: Dynamic algorithms to detect and avoid obstacles in the vehicle's path.
- **Motor Control**: Efficient communication between a NVIDIA Jetson Nano and Arduino ESP32 to control motors via a Sabretooth motor driver.
- **Wi-Fi Communication**: Commands are transmitted over Wi-Fi, ensuring seamless interaction between hardware components.
- **Hybrid Approach**: Combines Python-based software for vision processing on the Jetson Nano with motor control logic executed on the ESP32.
