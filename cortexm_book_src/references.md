## 0.1 Arm Documents

| Document | Use it for |
|---|---|
| **Arm Cortex-M4 Devices Generic User Guide** (DUI0553) | Programmer's model, registers, NVIC, SysTick, MPU, FPU |
| **Armv7-M Architecture Reference Manual** (DDI0403) | Authoritative instruction and architecture definition |
| **Armv6-M and Armv8-M Reference Manuals** | M0/M0+ and M23/M33 details |
| **Cortex-M Technical Reference Manuals** | Core implementation and options |
| **Arm Procedure Call Standard (AAPCS)** | Register use and calling convention |
| **CoreSight and ADI specifications** | Debug architecture and SWD protocol |
| **CMSIS documentation** (arm-software.github.io/CMSIS_6) | Core API, DSP, RTOS, SVD, Pack |

## 0.2 Vendor Documents (STM32 Example)

| Document | Use it for |
|---|---|
| Reference manual RM0390 (F446) | Peripherals, memory map, clock tree |
| Datasheet | Pinout, electrical limits, alternate functions |
| Programming manual PM0214 | Cortex-M4 as implemented by ST |
| Errata sheet | Silicon bugs and workarounds |
| AN2606 | System bootloader |

## 0.3 Books

- Joseph Yiu, *The Definitive Guide to ARM Cortex-M3 and Cortex-M4 Processors*.
- Joseph Yiu, *The Definitive Guide to Arm Cortex-M0 and Cortex-M0+ Processors*.
- Jonathan Valvano, *Embedded Systems: Real-Time Interfacing to Arm Cortex-M Microcontrollers*.
- Elecia White, *Making Embedded Systems*.

## 0.4 Tools

- GNU Arm Embedded Toolchain (`arm-none-eabi-gcc`, `gdb`, `objdump`, `size`).
- OpenOCD, pyOCD, ST-LINK, J-Link.
- PulseView or Saleae for logic analysis.
- Compiler Explorer (godbolt.org) with `ARM GCC` to study generated code.

> [tip] How this book relates to the others | Books 1 and 2 (STM32 with C and Embedded Rust with STM32) use the facts in this book. When something puzzles you there, such as a fault, a clock problem or a linker issue, come back here.
