# Z-BOT Software Architecture

This repository contains the core software architecture for **Z-BOT**, a spider-inspired climbing robot designed for scaffolding net environments.

## Project Goals
- Validate scaffolding net climbing mechanism
- Tele-operated, sequence-based climbing (no full autonomy yet)
- Modular software architecture ready for FSM and ROS2 integration

## Architecture Overview

zbot_software/
│
├── main.py               # PROGRAMI BAŞLATAN DOSYA
│
├── config/
│   ├── servos.py         # Servo index ve isimler
│   └── poses.py          # Tüm pose tanımları (tuning burada)
│
├── core/
│   ├── serial_comm.py    # Arduino ile konuşma
│   ├── pose_engine.py    # Pose uygulatma
│   └── climber.py        # Climbing sequence (beyin)
│
├── teleop/
│   └── keyboard.py       # Klavyeden komut alma
│
└── utils/
    └── logger.py         # Log / print işleri


## Current Status
- Servo control: implemented on Arduino
- Software architecture: initialized
- Climbing logic: sequence-based (FSM-ready)
- Tuning: pending mechanical integration

## Next Steps
- Keyboard / joystick tele-operation
- FSM integration
- Safety & recovery logic
- Thermal sensing module

## Authors
Z-BOT Team - Zijing College – Tsinghua University
