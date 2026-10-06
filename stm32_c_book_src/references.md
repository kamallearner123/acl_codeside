## 0.1 Reference Documents

Always read the **right document for the question**. Download them from st.com by searching the document number.

| Document | Use it for |
|---|---|
| **Datasheet** (e.g. STM32F446xC/E) | Pinout, alternate functions, electrical limits, memory sizes, timings |
| **Reference Manual RM0390** | Every peripheral and register in detail: GPIO, TIM, ADC, USART, I2C, SPI, bxCAN, DMA |
| **Programming Manual PM0214** | Cortex-M4 instruction set, NVIC, SysTick, fault status registers |
| **Nucleo User Manual UM1724** | Board schematic, solder bridges, ST-LINK, Arduino/Morpho pin mapping |
| **UM1725 - HAL and LL driver description** | Every HAL function, parameter and return value |
| **UM1718 - STM32CubeMX user manual** | Pinout, clock tree, code generation options |
| **AN2606 - Boot modes** | System bootloader and boot pin behaviour |
| **AN4838 / AN4839** | MPU and fault handling on Cortex-M |

## 0.2 How to Read a Reference Manual

1. Start with the **block diagram** of the peripheral to learn its clock and signal path.
2. Read the **functional description** for the mode you use (e.g. "PWM mode" in the timer chapter).
3. Check the **register map** only after you understand the behaviour; use the bit descriptions for exact fields.
4. Verify the **clock source** of the peripheral (APB1 vs APB2) in the RCC chapter - most "wrong frequency" bugs are here.
5. Compare with the **HAL source** in `Drivers/STM32F4xx_HAL_Driver` - it is working reference code.

## 0.3 Tools and Further Reading

- STM32CubeIDE, STM32CubeMX and STM32CubeProgrammer from st.com.
- OpenOCD and the GNU Arm Embedded Toolchain documentation (GDB, `addr2line`).
- PulseView / sigrok decoder list for I2C, SPI, UART and CAN.
- ARM *Cortex-M4 Devices Generic User Guide* for the architecture view.
- Joseph Yiu, *The Definitive Guide to ARM Cortex-M3 and Cortex-M4 Processors* - the standard architecture text.
- Your own lab notes: keep a log of every bug, its root cause and how you found it.
