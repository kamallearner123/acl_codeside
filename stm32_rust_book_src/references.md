## 0.1 Documents and Books

| Resource | Use it for |
|---|---|
| **The Embedded Rust Book** (docs.rust-embedded.org/book) | Official introduction: no_std, PAC/HAL layers, interrupts, concurrency |
| **The Embedonomicon** | How startup, linker scripts and panic handlers work underneath |
| **The Rust Programming Language** | Ownership, borrowing, lifetimes, traits - the language itself |
| **Reference Manual RM0390** | Peripheral registers for STM32F446 |
| **Datasheet STM32F446xC/E** | Pinout, alternate functions, limits |
| **UM1724 Nucleo-64 manual** | Board schematic, ST-LINK, VCP wiring |
| **PM0214 Cortex-M4 programming manual** | NVIC, fault registers, instruction set |

## 0.2 Crates Used in This Course

| Crate | Purpose |
|---|---|
| `cortex-m`, `cortex-m-rt` | Core access, startup and vector table |
| `stm32f4xx-hal`, `stm32f4` | HAL and PAC for STM32F4 |
| `embedded-hal` 1.0, `embedded-hal-async` | Portable peripheral traits |
| `embedded-hal-bus`, `embedded-hal-mock` | Bus sharing and host test mocks |
| `heapless` | Fixed-capacity Vec, String, queues |
| `defmt`, `defmt-rtt`, `panic-probe` | Logging and panic reporting |
| `critical-section` | Safe sharing with interrupts |
| `bxcan` | CAN controller driver |
| `rtic`, `embassy-stm32`, `embassy-executor` | Concurrency frameworks |
| `embedded-graphics`, `embedded-sdmmc` | Displays and FAT storage |

## 0.3 Tools

- `rustup`, `cargo`, `clippy`, `rustfmt`.
- `probe-rs` for flashing, debugging and RTT.
- `flip-link` for stack overflow protection.
- `cargo-binutils` (`cargo size`, `cargo objdump`) and `cargo-bloat`.
- OpenOCD with `arm-none-eabi-gdb` as the classic alternative.
- PulseView/sigrok for logic analyzer captures.

> [warn] Versions change | HAL APIs differ between releases. Pin versions in `Cargo.toml`, read the `examples/` directory of the exact version, and check docs.rs before copying code from any article - including this one.
