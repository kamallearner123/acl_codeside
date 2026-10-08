(() => {
  'use strict';

  const node = (label, kind, about, facts = []) => ({ label, kind, about, facts });
  const edge = (data, why, description, title, route = null) => ({
    data, why, description, title, ...(route || {})
  });
  const flow = (group, title, summary, type, nodes, edges, core) => ({
    group, title, summary, type, nodes, edges, core
  });

  const architectureFlow = flow('ARCHITECTURE', 'Cortex-M4 STM32 block diagram',
    'Follow instruction, data, address, control, DMA, interrupt, clock, and reset paths through an STM32L476RG-style MCU.',
    'ARCHITECTURE · BUS MATRIX',
    [
      node('Cortex-M4F core', 'BUS MASTER', 'The Cortex-M4F executes instructions and initiates instruction fetches, data accesses, and system/peripheral accesses using distinct core interfaces.', [
        ['I-CODE', 'Instruction fetch interface to the code region.'],
        ['D-CODE', 'Data access interface to the code region.'],
        ['SYSTEM', 'Access path for SRAM, peripherals, and the system address space.']
      ]),
      node('NVIC', 'INTERRUPT CONTROL', 'The Nested Vectored Interrupt Controller manages interrupt enable, pending state, priority, and delivery of exceptions to the processor.', [
        ['IN', 'Peripheral interrupt requests feed the NVIC.'],
        ['OUT', 'An eligible exception is signaled to the Cortex-M core.']
      ]),
      node('Debug / trace', 'SWD · JTAG · DWT', 'Debug and trace facilities provide debug access and optional instrumentation. Available trace features and pins depend on the MCU/package and board.'),
      node('RCC + reset', 'CLOCK CONTROL', 'RCC selects/configures system and bus clocks, gates peripheral clocks, and works with reset/power-control circuitry. Clock lines are timing/control, not data buses.'),
      node('DMA1 / DMA2', 'AHB MASTER', 'DMA controllers can initiate configured data transfers between supported peripherals and memories without a CPU load/store for every item.'),
      node('AHB bus matrix', 'INTERCONNECT', 'The bus matrix arbitrates among bus masters and routes transactions to memory/peripheral targets. Address selects a target; control qualifies the transfer; read/write data carries the value.'),
      node('Flash interface', 'CODE / DATA', 'Flash stores program instructions, constants, and the initial image for writable initialized data. The core can fetch instructions through I-CODE.'),
      node('SRAM1', 'DATA MEMORY', 'SRAM1 holds writable data, stack, heap, and buffers according to the linker/application layout. CPU and DMA accesses share the memory system.'),
      node('SRAM2', 'DATA MEMORY', 'SRAM2 is another on-chip SRAM region with its own address range and device-specific properties. Consult the reference manual for exact size and retention.'),
      node('AHB2 peripherals', 'GPIO · EXTI', 'AHB-connected peripherals include GPIO and related I/O functions. GPIO registers are memory-mapped; pins require correct mode and board wiring.'),
      node('AHB / APB bridge', 'BUS BRIDGE', 'The bridge connects the AHB interconnect to APB peripheral buses. APB transactions have protocol/timing rules distinct from AHB transfers.'),
      node('APB1 peripherals', 'TIM · USART · I2C · CAN', 'APB1 connects selected lower-speed peripherals such as timers, serial interfaces, and bxCAN on supported STM32L4 parts. Verify the exact instance map for the device.'),
      node('APB2 peripherals', 'TIM · SPI · ADC', 'APB2 connects selected peripherals such as timers, serial interfaces, and ADC on supported STM32L4 parts. Exact mapping is device-specific.'),
      node('GPIO pins', 'PACKAGE PADS', 'Peripheral and GPIO signals reach package pads through pin alternate-function selection. The board schematic determines external connections.')
    ],
    [
      edge('Instruction address + fetch control', 'The Cortex-M4 has separate instruction and data paths into the memory system. This is a logical on-chip interface, not an external address pin bus.', 'The core issues an instruction fetch on its I-CODE interface toward the interconnect.'),
      edge('Flash address → instruction data', 'The matrix routes the selected address to the Flash interface; fetched instruction data returns to the core.', 'The instruction fetch reaches Flash and returns instruction data.'),
      edge('Data address + read/write control', 'The D-CODE interface accesses the code region; the SYSTEM interface covers SRAM and peripheral/system accesses. Which interface serves an address depends on the memory map.', 'The core issues a data or system transaction with address and access-control information.'),
      edge('Read data ← / write data →', 'Read data returns to the requesting master; write data travels toward the selected target. The matrix arbitrates concurrent CPU and DMA requests.', 'A memory transaction transfers its data between the core and SRAM through the matrix.'),
      edge('Peripheral address + control / data', 'Address decoding and the AHB/APB bridge route accesses to peripheral registers; address/control and data are distinct parts of a transaction.', 'The matrix routes a memory-mapped peripheral access through the appropriate AHB/APB path.'),
      edge('DMA request → transfer → memory/peripheral', 'DMA is another bus master. Source/destination, width, direction, request mapping, and arbitration are configured; CPU involvement is still needed for setup and completion.', 'DMA initiates a configured transfer across the interconnect.'),
      edge('IRQ request → NVIC → exception', 'Peripheral interrupt requests do not use the data bus as payload. NVIC priority/enable and core masking determine when the CPU takes the exception.', 'A peripheral event requests service through the NVIC to the Cortex-M core.'),
      edge('Clock / reset distribution', 'RCC supplies timing and reset control to core, bus domains, and peripherals; these are not address/data transfers.', 'Configured clocks and reset signals establish when the MCU blocks can operate.')
    ],
    'A bus transaction is more than a wire: the master presents an address and control, the interconnect selects a target, and data is transferred in the appropriate direction. Interrupt and clock/reset paths are separate from payload data buses.'
  );
  architectureFlow.architecture = {
    width: 1480,
    height: 900,
    wires: [
      { edge: 0, kind: 'instruction', label: 'I-CODE · instruction address + fetch control →', labelX: 350, labelY: 145, points: [[320, 158], [430, 158], [430, 330], [565, 330]] },
      { edge: 1, kind: 'address', label: 'Address + READ control → Flash', labelX: 805, labelY: 105, points: [[795, 345], [845, 345], [845, 128], [900, 128]] },
      { edge: 1, kind: 'instruction', label: '← instruction data', labelX: 810, labelY: 153, points: [[900, 154], [825, 154], [825, 363], [795, 363]] },
      { edge: 1, kind: 'instruction', label: '← fetched instruction to core', labelX: 365, labelY: 170, points: [[565, 352], [490, 352], [490, 171], [320, 171]] },
      { edge: 2, kind: 'address', label: 'D-CODE · data address / access control →', labelX: 350, labelY: 191, points: [[320, 205], [448, 205], [448, 374], [565, 374]] },
      { edge: 2, kind: 'address', label: 'SYSTEM · SRAM / peripheral address + control →', labelX: 350, labelY: 238, points: [[320, 252], [466, 252], [466, 420], [565, 420]] },
      { edge: 3, kind: 'address', label: 'Address + control → SRAM1', labelX: 800, labelY: 252, points: [[795, 380], [850, 380], [850, 253], [900, 253]] },
      { edge: 3, kind: 'data', label: '← read data', labelX: 800, labelY: 279, points: [[900, 282], [835, 282], [835, 400], [795, 400]] },
      { edge: 3, kind: 'data', label: 'write data →', labelX: 800, labelY: 296, points: [[795, 415], [825, 415], [825, 297], [900, 297]] },
      { edge: 3, kind: 'address', label: 'Address + control → SRAM2', labelX: 800, labelY: 414, points: [[795, 410], [845, 410], [845, 400], [900, 400]] },
      { edge: 3, kind: 'data', label: '← read data / write data →', labelX: 800, labelY: 434, points: [[900, 430], [825, 430], [825, 455], [795, 455]] },
      { edge: 4, kind: 'address', label: 'AHB2 address + control → GPIO / EXTI', labelX: 795, labelY: 523, points: [[795, 442], [820, 442], [820, 560], [900, 560]] },
      { edge: 4, kind: 'address', label: 'APB address + control → bridge', labelX: 800, labelY: 677, points: [[795, 466], [840, 466], [840, 740], [900, 740]] },
      { edge: 4, kind: 'address', label: 'APB address + control → APB1', labelX: 1140, labelY: 423, points: [[1130, 715], [1150, 715], [1150, 438], [1195, 438]] },
      { edge: 4, kind: 'data', label: 'APB1 read data ←', labelX: 1140, labelY: 457, points: [[1195, 462], [1160, 462], [1160, 730], [1130, 730]] },
      { edge: 4, kind: 'data', label: 'APB1 write data →', labelX: 1140, labelY: 473, points: [[1130, 745], [1147, 745], [1147, 477], [1195, 477]] },
      { edge: 4, kind: 'address', label: 'APB address + control → APB2', labelX: 1140, labelY: 568, points: [[1130, 755], [1172, 755], [1172, 583], [1195, 583]] },
      { edge: 4, kind: 'data', label: 'APB2 read data ←', labelX: 1140, labelY: 607, points: [[1195, 608], [1182, 608], [1182, 770], [1130, 770]] },
      { edge: 4, kind: 'data', label: 'APB2 write data →', labelX: 1140, labelY: 623, points: [[1130, 780], [1170, 780], [1170, 625], [1195, 625]] },
      { edge: 4, kind: 'data', label: 'GPIO signal → package pin', labelX: 1140, labelY: 753, points: [[1130, 580], [1164, 580], [1164, 735], [1195, 735]] },
      { edge: 5, kind: 'dma', label: 'DMA AHB master · address / control →', labelX: 380, labelY: 582, points: [[555, 665], [545, 665], [545, 475], [565, 475]] },
      { edge: 5, kind: 'dma', label: 'DMA read / write data ↔ memory or peripheral', labelX: 385, labelY: 602, points: [[555, 680], [530, 680], [530, 490], [585, 490]] },
      { edge: 5, kind: 'dma', label: 'write data → SRAM / peripheral', labelX: 385, labelY: 619, points: [[585, 485], [520, 485], [520, 700], [555, 700]] },
      { edge: 6, kind: 'event', label: 'AHB peripheral IRQ', labelX: 835, labelY: 637, points: [[900, 612], [875, 612], [875, 830], [850, 830]] },
      { edge: 6, kind: 'event', label: 'APB1 IRQ', labelX: 1100, labelY: 482, points: [[1195, 480], [1180, 480], [1180, 820], [850, 820]] },
      { edge: 6, kind: 'event', label: 'APB2 IRQ', labelX: 1100, labelY: 628, points: [[1195, 625], [1190, 625], [1190, 810], [850, 810]] },
      { edge: 6, kind: 'event', label: 'IRQ request → NVIC → core exception', labelX: 445, labelY: 843, points: [[850, 830], [340, 830], [340, 386], [320, 386]] },
      { edge: 7, kind: 'clock', label: 'RCC clock enables / clock domains · not payload data', labelX: 585, labelY: 83, points: [[555, 123], [860, 123], [860, 505], [900, 505]] },
      { edge: 7, kind: 'clock', label: 'SYSCLK → core', labelX: 338, labelY: 91, points: [[380, 96], [350, 96], [350, 108], [190, 108], [190, 120]] },
      { edge: 7, kind: 'clock', label: 'APB clocks', labelX: 1100, labelY: 340, points: [[555, 145], [1165, 145], [1165, 448], [1195, 448]] },
      { edge: 7, kind: 'clock', label: 'APB2 clock', labelX: 1100, labelY: 356, points: [[555, 160], [1180, 160], [1180, 593], [1195, 593]] },
      { edge: 7, kind: 'reset', label: 'Reset control', labelX: 326, labelY: 95, points: [[380, 112], [360, 112], [360, 294], [190, 294], [190, 285]] }
    ],
    components: [
      { node: 0, id: 'core', x: 60, y: 120, width: 260, height: 165, title: 'Cortex-M4F core', subtitle: 'CPU · registers · ALU · FPU', inner: ['PC · SP · LR · R0–R12 · xPSR', 'I-CODE   D-CODE   SYSTEM'] },
      { node: 1, id: 'nvic', x: 60, y: 340, width: 260, height: 92, title: 'NVIC', subtitle: 'interrupt enable · pending · priority' },
      { node: 2, id: 'debug', x: 60, y: 465, width: 260, height: 78, title: 'Debug / trace', subtitle: 'SWD · optional JTAG / trace*' },
      { node: 3, id: 'rcc', x: 380, y: 75, width: 175, height: 105, title: 'RCC + reset', subtitle: 'clock source · prescaler · enable' },
      { node: 4, id: 'dma', x: 380, y: 620, width: 175, height: 100, title: 'DMA1 / DMA2', subtitle: 'additional bus master' },
      { node: 5, id: 'matrix', x: 565, y: 305, width: 230, height: 185, title: 'AHB bus matrix', subtitle: 'arbitrate · decode · route', inner: ['address + control → target', 'read data ←  /  write data →'] },
      { node: 6, id: 'flash', x: 900, y: 78, width: 230, height: 100, title: 'Flash interface', subtitle: 'program · constants · .data image' },
      { node: 7, id: 'sram1', x: 900, y: 205, width: 230, height: 100, title: 'SRAM1', subtitle: 'code data · stack · heap · buffers' },
      { node: 8, id: 'sram2', x: 900, y: 350, width: 230, height: 100, title: 'SRAM2', subtitle: 'additional on-chip SRAM region' },
      { node: 9, id: 'ahb2', x: 900, y: 500, width: 230, height: 120, title: 'AHB2 peripherals', subtitle: 'GPIO · EXTI · I/O control', inner: ['memory-mapped registers'] },
      { node: 10, id: 'bridge', x: 900, y: 690, width: 230, height: 100, title: 'AHB / APB bridge', subtitle: 'bus protocol + clock domain' },
      { node: 11, id: 'apb1', x: 1195, y: 400, width: 220, height: 100, title: 'APB1 peripherals', subtitle: 'TIM · USART · I2C · bxCAN' },
      { node: 12, id: 'apb2', x: 1195, y: 545, width: 220, height: 100, title: 'APB2 peripherals', subtitle: 'TIM · SPI · ADC' },
      { node: 13, id: 'pins', x: 1195, y: 700, width: 220, height: 70, title: 'GPIO package pins', subtitle: 'alternate function → board circuit' }
    ]
  };
  architectureFlow.edgeNodes = [
    [0, 5],
    [5, 6],
    [0, 5],
    [5, 7, 8],
    [5, 9, 10, 11, 12, 13],
    [4, 5, 7, 9],
    [9, 11, 12, 1, 0],
    [3, 0, 5, 9, 10, 11, 12]
  ];
  [
    'instruction',
    'instruction',
    'address',
    'data',
    'address',
    'dma',
    'event',
    'clock'
  ].forEach((kind, index) => {
    architectureFlow.edges[index].kind = kind;
  });

  const flows = [
    flow('FOUNDATIONS', 'Power-on and reset',
      'See how reset selects the initial stack pointer and reset handler before C runtime setup reaches main().',
      'RESET · STARTUP',
      [
        node('Power + reset', 'RESET SOURCE', 'Power-on reset or another reset source holds the MCU in its defined reset state.'),
        node('Vector table', 'FLASH', 'The vector table starts at the reset vector-table address selected by the device boot configuration.'),
        node('Initial MSP', 'STACK POINTER', 'The first vector-table word supplies the initial Main Stack Pointer value. The core loads it before executing the reset handler.'),
        node('Reset_Handler', 'CPU ENTRY', 'The reset handler is the second vector-table entry. Cortex-M exception entry begins at this handler after reset.'),
        node('C runtime setup', 'STARTUP CODE', 'Startup code copies initial values for writable data into SRAM and clears zero-initialized data. It uses the stack already established by reset.'),
        node('SystemInit', 'SYSTEM SETUP', 'Device startup code may configure clocks and low-level system state before the C library initializer. Exact order depends on the startup project.'),
        node('main()', 'APPLICATION', 'The C runtime calls main() after the startup and runtime initialization steps complete.')
      ],
      [
        edge('Power-good / reset release', 'The core cannot fetch application instructions until reset is released.', 'The MCU exits its reset state and starts the Cortex-M reset sequence.'),
        edge('Vector-table address', 'The reset sequence uses the device boot mapping and vector-table location.', 'The core reads the first two vector-table entries from the active reset vector table.'),
        edge('Initial stack address', 'The stack is ready before the first handler instruction executes.', 'The first vector-table word is loaded into MSP; the next word identifies Reset_Handler.'),
        edge('Program counter = Reset_Handler', 'The processor begins executing the reset handler.', 'The reset handler branches into the toolchain startup code.'),
        edge('Copy .data; clear .bss', 'Initialized writable objects need their initial values in SRAM; zero-initialized objects must start at zero.', 'Startup prepares the C memory image in SRAM and then performs device-specific low-level initialization.'),
        edge('Runtime initialization complete', 'The application should start only after its required runtime state has been prepared.', 'The runtime enters the application at main().')
      ],
      'Reset loads MSP and the reset-handler address from the vector table. Startup prepares C data in SRAM; it does not copy the whole program out of Flash.'
    ),
    flow('FOUNDATIONS', 'Flash and SRAM layout',
      'Separate code and constants in Flash from writable program data, stack, and heap in SRAM.',
      'MEMORY · DATA',
      [
        node('Flash', 'NONVOLATILE', 'Flash stores the program image and retains its contents without power. Most instructions execute from Flash.'),
        node('.text / .rodata', 'LINKER SECTIONS', '.text contains code; .rodata contains read-only constants. Their exact placement is defined by the linker script.'),
        node('Initial .data image', 'FLASH LOAD IMAGE', 'The initial values for writable initialized globals are stored in the program image in Flash.'),
        node('Startup copy', 'RESET HANDLER', 'Startup copies the .data load image from Flash to its SRAM run address before application code uses it.'),
        node('SRAM .data / .bss', 'WRITABLE DATA', '.data holds initialized writable objects; .bss holds zero-initialized or uninitialized static-storage objects.'),
        node('Stack / heap', 'RUNTIME MEMORY', 'The stack supports calls and automatic objects. A heap may be used if the runtime and application allocate dynamically.')
      ],
      [
        edge('Instruction fetch', 'The CPU can normally fetch instructions directly from Flash.', 'The program image provides instruction bytes and read-only constants.'),
        edge('Initial values', 'Writable global values need a RAM copy even though their initial image is stored in Flash.', 'The Flash load image supplies initial values for the .data section.'),
        edge('Startup copy', 'The application expects initialized writable data to reside at its SRAM run address.', 'Reset startup copies the .data load image into SRAM.'),
        edge('Zero initialization', 'C requires zero-initialized static-storage objects to begin as zero.', 'Startup clears the .bss range in SRAM before main().'),
        edge('Runtime reads / writes', 'Function calls and local state use the stack; heap use depends on the program.', 'The running application accesses its SRAM sections and runtime memory.')
      ],
      'The entire firmware image is not copied into SRAM at boot. Code usually executes from Flash; startup copies .data and clears .bss.'
    ),
    flow('FOUNDATIONS', 'CPU instruction execution',
      'Trace one C expression through instruction fetch, operand reads, arithmetic, and a result write.',
      'CPU · INSTRUCTIONS',
      [
        node('Flash instruction', 'FETCH', 'The CPU fetches an instruction at the address in the Program Counter (PC).'),
        node('Instruction data', 'FETCHED OPCODE', 'The instruction bits return from Flash before the core decodes the operation.'),
        node('Decode', 'CORTEX-M4', 'The core decodes the instruction and determines its operands and operation.'),
        node('Read operands', 'REGISTERS / SRAM', 'Operands may already be in registers or may require load instructions from memory.'),
        node('ALU operation', 'EXECUTE', 'The ALU performs integer arithmetic such as an addition and updates condition flags when requested.'),
        node('Write result', 'REGISTER / SRAM', 'The result is written to a destination register or stored to memory by a later instruction.')
      ],
      [
        edge('PC → instruction address', 'The PC identifies the next instruction stream location.', 'The core fetches an instruction from the program image.'),
        edge('Opcode + operands', 'The fetched instruction must be decoded before its operation can execute.', 'The core identifies the instruction and its operand requirements.'),
        edge('a and b', 'The addition needs both input values; a compiler may choose registers or memory loads.', 'The core obtains the operands for the expression.'),
        edge('a + b', 'The ALU computes the sum as part of executing the instruction sequence.', 'The arithmetic result is produced by the CPU core.'),
        edge('x', 'The compiler decides whether the result stays in a register or is written to memory.', 'The result is committed to its selected destination.')
      ],
      'A C statement is compiled into one or more machine instructions. The CPU executes instructions; it does not execute C syntax directly.'
    ),
    flow('FOUNDATIONS', 'GPIO output: register to pin',
      'Connect a C register write to the output driver and the external LED or load.',
      'DATA · GPIO',
      [
        node('C statement', 'APPLICATION', 'Example: set a GPIO output bit using a device-header register definition.'),
        node('CPU load / store', 'CORTEX-M4', 'The compiler emits instructions that read or write a memory-mapped peripheral address.'),
        node('Bus interconnect', 'AHB / PERIPHERAL PATH', 'The bus fabric routes the transaction to the addressed peripheral. Exact paths depend on the bus matrix and register region.'),
        node('GPIO ODR / BSRR', 'PERIPHERAL REGISTER', 'GPIO output data can be changed through ODR or atomically set/reset through BSRR.'),
        node('GPIO output logic', 'PAD CONTROL', 'GPIO mode and output configuration determine the electrical behavior of the output driver.'),
        node('Pin → LED', 'BOARD CIRCUIT', 'The MCU pad drives the connected board circuit, subject to pin configuration and electrical limits.')
      ],
      [
        edge('GPIOA->BSRR = (1U << 5)', 'Software describes the intended peripheral-register action.', 'The source code compiles into a sequence that targets a GPIO register.'),
        edge('Address + write value', 'A peripheral register access is a memory-mapped I/O transaction.', 'The CPU issues the register write across the interconnect.'),
        edge('GPIO register address', 'Address decoding directs the transaction to the GPIO block.', 'The peripheral bus path delivers the write to the GPIO register.'),
        edge('Set pin 5', 'The selected register bit updates GPIO output state.', 'GPIO output logic applies the configured output value to the pin.'),
        edge('Logic level at pad', 'External behavior depends on alternate-function/mode, load, board wiring, and voltage limits.', 'The pin changes its electrical output; a connected LED responds according to its circuit.')
      ],
      'GPIO is a memory-mapped peripheral, not ordinary SRAM. A register write changes the GPIO hardware state; board wiring determines what the pin drives.'
    ),
    flow('FOUNDATIONS', 'GPIO input: pin to program',
      'Follow a button level through the input circuit and IDR bit into a CPU decision.',
      'DATA · GPIO',
      [
        node('Button / signal', 'EXTERNAL INPUT', 'A switch or external circuit applies a logic level to the configured input pin.'),
        node('GPIO pad', 'MCU PIN', 'The pin must be configured for the required input or alternate-function mode, with appropriate electrical setup.'),
        node('Input circuit', 'SYNCHRONIZER', 'The input path conditions the external signal for the digital GPIO logic. Signal timing and filtering are device/configuration dependent.'),
        node('GPIOx_IDR', 'INPUT REGISTER', 'The Input Data Register exposes sampled pin state as readable bit fields.'),
        node('CPU load', 'MEMORY-MAPPED READ', 'A CPU load reads the GPIO register through the peripheral address path.'),
        node('C condition', 'APPLICATION', 'Software tests the relevant IDR bit and chooses what to do next.')
      ],
      [
        edge('Low / high voltage', 'The external circuit and pull configuration establish the input level.', 'The button or signal changes the electrical state presented to the pin.'),
        edge('Digital input level', 'GPIO mode and input electrical characteristics govern how the level is interpreted.', 'The pin input path presents a digital state to the GPIO block.'),
        edge('IDR bit sample', 'The peripheral makes the pin state visible through its input register.', 'GPIO updates the state that software can read from IDR.'),
        edge('GPIOx_IDR address', 'The CPU must perform a peripheral-register read to sample the state.', 'A bus transaction returns the input register value to the core.'),
        edge('IDR[pin] = 0 or 1', 'The software condition uses the selected bit, not an abstract button object.', 'The C code evaluates the sampled pin state.')
      ],
      'A GPIO input read is a register read of the sampled pin state. Debouncing is a separate hardware or software design decision.'
    ),
    flow('FOUNDATIONS', 'RCC and clock flow',
      'Separate the clock tree from data paths and see why a peripheral clock must be enabled.',
      'CLOCK · RCC',
      [
        node('Clock source', 'HSI / HSE / MSI', 'The STM32L476 offers internal and external clock sources. Available sources and startup constraints are device-specific.'),
        node('RCC configuration', 'CLOCK CONTROL', 'RCC selects and prescales system/bus clocks and controls peripheral clock enables.'),
        node('SYSCLK → buses', 'CLOCK TREE', 'Clock selection and bus prescalers determine clocks delivered to core and bus domains.'),
        node('Peripheral enable', 'RCC ENABLE BIT', 'The relevant RCC enable bit supplies a clock to a peripheral bus interface/block.'),
        node('GPIO / TIM / USART', 'CLOCKED PERIPHERAL', 'Peripheral operation depends on the correct clock domain and its own configuration.')
      ],
      [
        edge('Oscillating clock', 'A valid source must be ready before it is selected as the active system clock.', 'The chosen internal or external source becomes available.'),
        edge('Source + prescalers', 'RCC controls source selection and clock division; it does not create data transfers.', 'RCC configures the clock tree for the core and bus domains.'),
        edge('Bus clock', 'The selected peripheral sits on a clocked bus/domain.', 'The configured clock reaches the associated peripheral bus domain.'),
        edge('Peripheral clock enable', 'A peripheral may not respond as expected if its clock is gated off.', 'Software enables its RCC clock and then configures the peripheral.')
      ],
      'Clock is a timing signal, not payload data. RCC configuration and peripheral clock enables are prerequisites for many peripheral operations.'
    ),
    flow('PERIPHERALS', 'Timer: clock to event',
      'See a timer count clock ticks, match a programmed limit, then either be polled or request an interrupt.',
      'CLOCK · TIMER',
      [
        node('Timer input clock', 'APB / TIMER CLOCK', 'The timer receives a clock derived from its bus clocking rules and RCC configuration.'),
        node('Prescaler', 'PSC', 'The prescaler divides the timer input clock before it advances the counter.'),
        node('Counter', 'TIMx_CNT', 'The counter increments, decrements, or counts center-aligned according to timer configuration.'),
        node('ARR / compare', 'TIMx_ARR / CCRx', 'Auto-reload and capture/compare values define update or compare conditions.'),
        node('Timer flag / event', 'STATUS / OUTPUT', 'A timer event updates status and may affect output-compare/PWM pins depending on configuration.'),
        node('CPU polling or NVIC', 'SOFTWARE SERVICE', 'Software can poll a flag or enable a timer interrupt routed through the NVIC.')
      ],
      [
        edge('Timer clock ticks', 'The counter advances only when its clocking and enable configuration allow it.', 'The configured timer clock enters the prescaler.'),
        edge('Divided ticks', 'The prescaler sets the count rate relative to its input clock.', 'The prescaler generates counter clock events.'),
        edge('CNT increments', 'Counter behavior is determined by mode, direction, and reload configuration.', 'The counter advances toward its configured update or compare condition.'),
        edge('ARR / CCR match', 'A programmed boundary or compare value defines the event condition.', 'The timer detects an update/compare event and sets status or changes an output.'),
        edge('UIF / CC flag; optional IRQ', 'Polling reads status; interrupt mode additionally requires interrupt enables and NVIC configuration.', 'The application observes the timer event using its selected service method.')
      ],
      'A timer event does not automatically mean an ISR runs. Polling, interrupt enable, NVIC routing, and timer status are distinct parts of the flow.'
    ),
    flow('PERIPHERALS', 'Interrupt: event to callback',
      'Make the hardware request, NVIC arbitration, exception entry, ISR, HAL handler, and callback explicit.',
      'EVENT · NVIC',
      [
        node('Peripheral status', 'STATUS FLAG', 'The peripheral records the configured condition in its status state.'),
        node('Peripheral event', 'GPIO / TIMER / USART', 'A configured hardware condition sets peripheral status and may request an interrupt.'),
        node('IRQ request', 'PERIPHERAL OUTPUT', 'The peripheral asserts its interrupt request only when relevant status and enable conditions are met.'),
        node('NVIC', 'PRIORITY / PENDING', 'The NVIC tracks pending/enabled interrupt lines and applies priority and masking rules.'),
        node('Exception entry', 'CORTEX-M4', 'The core accepts an eligible exception, stacks architectural state, and fetches the handler address.'),
        node('Vector → ISR', 'HANDLER', 'The vector table maps the exception/IRQ number to its handler function.'),
        node('HAL / driver handler', 'SOFTWARE LAYER', 'A framework handler may inspect and clear peripheral state before dispatching a user callback.'),
        node('Application callback', 'USER CODE', 'A callback is framework behavior, not a direct hardware call; its name and dispatch path depend on the driver.')
      ],
      [
        edge('Status condition', 'A hardware event first changes state in the peripheral.', 'The peripheral detects a configured event and may set a status flag.'),
        edge('IRQ line', 'The peripheral must have its relevant interrupt source enabled.', 'The peripheral asserts an interrupt request to its interrupt controller.'),
        edge('Pending + priority', 'NVIC enable, priority, and core masking determine when an IRQ can be taken.', 'The request becomes pending and competes according to NVIC/core rules.'),
        edge('Exception frame', 'The Cortex-M exception model saves state and changes control flow to the exception handler.', 'The core accepts the IRQ and performs exception entry.'),
        edge('Handler address', 'The vector table provides the handler entry for the IRQ number.', 'Execution branches to the configured ISR.'),
        edge('Driver dispatch', 'HAL or a driver may translate the IRQ into a higher-level event.', 'The ISR/handler services the peripheral and may invoke a callback.'),
        edge('Application event', 'Application code runs only if the software handler explicitly dispatches it.', 'The callback handles the event according to the framework contract.')
      ],
      'Hardware raises an interrupt request; it does not call C callbacks. The NVIC and CPU exception mechanism enter an ISR, and software may dispatch a callback.'
    ),
    flow('PERIPHERALS', 'DMA: hardware data movement',
      'Compare a CPU-serviced transfer with a configured DMA transfer and its completion notification.',
      'DATA · DMA',
      [
        node('Peripheral / memory', 'TRANSFER ENDPOINT', 'A peripheral data register or memory location is one endpoint of a configured transfer.'),
        node('DMA request', 'TRIGGER', 'A peripheral request or software trigger signals that a transfer can proceed.'),
        node('DMA channel', 'TRANSFER ENGINE', 'DMA uses programmed source, destination, count, direction, width, and mode settings.'),
        node('SRAM / peripheral', 'DESTINATION', 'The destination receives each transfer unit; buffer ownership and cache rules matter on applicable cores.'),
        node('DMA status / IRQ', 'COMPLETION', 'Half-transfer, transfer-complete, or error status may be reported to software.'),
        node('CPU service', 'APPLICATION', 'The CPU configures DMA and handles completion or errors; DMA is not a second general-purpose CPU.')
      ],
      [
        edge('Peripheral request', 'A request or trigger coordinates transfers with the peripheral data-ready/space condition.', 'The transfer source requests service from the DMA controller.'),
        edge('Configured transfer', 'DMA moves data according to programmed parameters and arbitration.', 'The DMA engine performs a transfer unit without a CPU load/store for every item.'),
        edge('Data item / block', 'The selected direction and increment settings determine where the item is written.', 'The transfer reaches the programmed destination.'),
        edge('TC / HT / error flag', 'Software needs status or an interrupt to know when its buffer can be consumed or refilled.', 'DMA records progress or completion and may request an IRQ.'),
        edge('Completion service', 'CPU time may be reduced for repeated data movement, but setup, arbitration, and completion handling still cost time.', 'The CPU services completion, error, or next-buffer work.')
      ],
      'DMA is a hardware data-movement engine. It reduces per-item CPU servicing; it does not eliminate configuration, arbitration, memory bandwidth, or completion work.'
    ),
    flow('SERIAL I/O', 'UART transmit',
      'Follow a byte from software through the UART data path and out as an asynchronous serial frame.',
      'DATA · UART TX',
      [
        node('Application byte', 'SOFTWARE', 'The application provides a byte or buffer, often through a polling, interrupt, or DMA API.'),
        node('UART TDR', 'TRANSMIT DATA', 'Software or DMA writes data to the transmit data register when the peripheral can accept it.'),
        node('Transmit shift register', 'SERIALIZER', 'The UART shifts frame bits at the configured baud rate and frame format.'),
        node('TX pin', 'ALTERNATE FUNCTION', 'The pin must be configured for the selected USART/UART transmit alternate function.'),
        node('External receiver', 'UART LINK', 'The receiver samples the line according to baud, data bits, parity, and stop-bit settings.')
      ],
      [
        edge("'A' = 0x41", 'The application chooses the byte; the transfer API determines how it reaches the peripheral.', 'Software offers one byte for transmission.'),
        edge('Byte written to TDR', 'The transmit register must be ready; polling, TX interrupt, or DMA can manage this handoff.', 'The byte enters the UART transmit path.'),
        edge('Start + LSB-first data + parity? + stop', 'UART frame layout follows configured word length, parity, and stop bits.', 'The UART serializes the byte onto the transmit signal at the selected baud rate.'),
        edge('Logic waveform', 'The pin alternate function and board wiring connect the peripheral signal to the external device.', 'The transmit waveform leaves the MCU pin and is sampled by the receiver.')
      ],
      'The UART peripheral generates the frame timing and start/data/stop signaling. TXE/TXFNF and TC indicate different states; they are not interchangeable.'
    ),
    flow('SERIAL I/O', 'UART receive with interrupt',
      'Show how a received character sets peripheral state and can request CPU service through the NVIC.',
      'DATA · UART RX · IRQ',
      [
        node('External transmitter', 'UART LINK', 'An external sender drives a serial frame using agreed baud and format settings.'),
        node('RX pin', 'ALTERNATE FUNCTION', 'The pin must be configured for the matching UART/USART receive alternate function.'),
        node('UART receiver', 'SAMPLER', 'The peripheral detects frame timing and samples bits according to its configuration.'),
        node('Receive data / status', 'RDR + FLAGS', 'Received data is available to software; status flags report data-ready and possible errors.'),
        node('NVIC → CPU', 'INTERRUPT PATH', 'If the relevant interrupt is enabled and unmasked, the peripheral request can be serviced by the core.'),
        node('ISR / driver', 'SOFTWARE SERVICE', 'The handler reads data and clears/services status according to the reference manual and driver.'),
        node('Application buffer', 'USER DATA', 'Software stores or queues the byte for later application processing.')
      ],
      [
        edge('Serial frame', 'The sender and receiver must agree on baud and frame format.', 'The external device transmits start, data, optional parity, and stop bits.'),
        edge('RX waveform', 'The pin mux routes the physical signal into the selected receiver.', 'The receive pin presents the signal to the UART peripheral.'),
        edge('Sampled byte', 'UART hardware reconstructs the configured frame and checks status.', 'The receiver makes the byte and status available in peripheral state.'),
        edge('RX-ready / error request', 'An enabled interrupt source can request service; data overrun is possible if service is too late.', 'The UART asserts an interrupt request for software service.'),
        edge('Pending USART IRQ', 'NVIC enable/priority and CPU masking affect when the handler executes.', 'The CPU enters the USART interrupt handler when the request is eligible.'),
        edge('Read RDR / clear condition', 'The exact flag-clear sequence is device-specific; follow the STM32L4 reference manual.', 'The driver reads the received data and places it in software-managed storage.')
      ],
      'An RX interrupt signals a service condition; it does not automatically create a safe message buffer or prevent overrun.'
    ),
    flow('SERIAL I/O', 'UART receive with DMA',
      'Follow incoming serial bytes into a DMA-managed SRAM buffer and then to CPU completion service.',
      'DATA · UART RX · DMA',
      [
        node('External transmitter', 'UART LINK', 'A sender transmits a stream using the configured UART framing and baud rate.'),
        node('UART RDR', 'RECEIVE DATA', 'The UART receiver places each completed frame into its receive data path and raises the DMA request when configured.'),
        node('DMA channel', 'TRANSFER ENGINE', 'The configured DMA channel moves receive data to the selected SRAM buffer and tracks transfer count.'),
        node('SRAM buffer', 'APPLICATION MEMORY', 'The destination buffer stores received bytes. Size, wrap, ownership, and processing policy must be designed by software.'),
        node('DMA event', 'HT / TC / ERROR', 'Half-transfer, transfer-complete, or error can notify software; circular reception often uses periodic/idle events.'),
        node('CPU processing', 'APPLICATION', 'The CPU processes completed portions and handles framing/packet boundaries.'),
        node('Parsed packet', 'APPLICATION RESULT', 'The application validates and interprets a byte range using the protocol framing rules.')
      ],
      [
        edge('UART serial frame', 'The stream arrives according to UART framing and baud configuration.', 'The peripheral receives an external serial frame.'),
        edge('RDR data + DMA request', 'UART-to-DMA mapping and request enable must be configured for this device.', 'The UART presents received data and signals a DMA transfer request.'),
        edge('Byte → buffer slot', 'DMA count, destination increment, transfer width, and mode determine buffer behavior.', 'DMA transfers the receive data into the next configured SRAM location.'),
        edge('Buffer progress', 'Software must know which bytes are valid and avoid reading data while it is being overwritten.', 'DMA advances through the buffer and records half/complete/error conditions.'),
        edge('Completion / idle event', 'DMA events indicate progress; UART idle detection or protocol framing may be needed for variable-length data.', 'An interrupt or polling check makes the CPU aware that data is ready.'),
        edge('Byte range / packet', 'The application owns parsing and buffer lifecycle after data becomes available.', 'The CPU processes the valid received data range.')
      ],
      'DMA fills memory; it does not identify packets or add a C string terminator. Treat receive data as bytes with explicit lengths.'
    ),
    flow('SERIAL I/O', 'SPI full-duplex transfer',
      'Visualise clocked exchange: MOSI and MISO shift simultaneously under an SCK signal and chip select.',
      'DATA · SPI',
      [
        node('CPU / DMA', 'DATA SOURCE', 'Software or DMA supplies transmit data and chooses how received data is collected.'),
        node('SPI data path', 'TX / RX REGISTERS', 'The SPI peripheral loads a shift register, generates clocks as master, and captures incoming bits.'),
        node('SCK + CS', 'CONTROL SIGNALS', 'The master configures clock polarity/phase and selects a slave with chip select.'),
        node('MOSI ↔ MISO', 'SERIAL SIGNALS', 'MOSI carries master-to-slave bits; MISO carries slave-to-master bits during the same clock pulses.'),
        node('External device', 'SPI SLAVE', 'The selected device interprets the transaction and returns bits according to its command protocol.')
      ],
      [
        edge('TX word / buffer', 'Each full-duplex frame has both a transmit and receive side, even if one side is ignored.', 'CPU or DMA supplies the next transmit word.'),
        edge('TX shift register', 'The peripheral uses its configured frame size, bit order, and clock mode.', 'SPI prepares outgoing bits and clocks the transfer.'),
        edge('SCK edges + chip select', 'Clock polarity/phase and slave selection define when data is sampled and shifted.', 'SCK and CS frame the transaction for the selected device.'),
        edge('MOSI out + MISO in', 'Data shifts in both directions concurrently; receive data must be drained to avoid overrun.', 'The external device exchanges bits with the STM32 on each clock cycle.')
      ],
      'SPI is usually full duplex: every clock shifts a bit out and a bit in. The device-specific command protocol gives those bits meaning.'
    ),
    flow('SERIAL I/O', 'I2C register read',
      'Step through a BME280-style register read and see address, ACK/NACK, repeated START, and STOP boundaries.',
      'DATA · I2C',
      [
        node('STM32 I2C master', 'BUS CONTROLLER', 'The master generates START/STOP conditions, clocks, address, and data according to I2C timing configuration.'),
        node('SDA + SCL', 'OPEN-DRAIN BUS', 'SDA and SCL are shared open-drain lines that require pull-ups and compatible voltage levels.'),
        node('BME280 address', '7-BIT TARGET', 'The sensor responds at a 7-bit address commonly 0x76 or 0x77 depending on its SDO pin.'),
        node('Register address', 'WRITE PHASE', 'The master selects a sensor register; exact protocol depends on device and access type.'),
        node('Repeated START', 'DIRECTION CHANGE', 'A combined transaction may issue repeated START to switch from register-address write to data read.'),
        node('Register data', 'READ PHASE', 'The sensor returns one or more bytes. The master ACKs intermediate bytes and NACKs the final byte before STOP.'),
        node('Transaction complete', 'STOP / RELEASE', 'The master issues STOP to finish the combined read and release the shared bus.')
      ],
      [
        edge('START + address + W', 'The master begins a transaction and addresses the target for a write phase.', 'The I2C controller emits START and the sensor address with the write direction.'),
        edge('ACK or NACK', 'ACK means a receiver accepted the byte; NACK means it did not acknowledge.', 'The addressed device responds in the acknowledge bit time.'),
        edge('Register index', 'The register pointer is device-protocol data, not part of generic I2C itself.', 'The master sends the target register address.'),
        edge('Repeated START + address + R', 'Many register reads use a repeated START to keep the combined transaction active.', 'The master changes direction and requests register data.'),
        edge('Data byte(s), ACK then final NACK', 'The master controls acknowledgement while reading; the final NACK signals that it will stop reading.', 'The sensor returns the register contents to the master.'),
        edge('STOP', 'A STOP releases the bus transaction; exact timing still follows the I2C specification.', 'The master ends the read transaction.')
      ],
      'I2C is an electrical shared bus plus a protocol. ACK/NACK, repeated START, pull-ups, and the sensor register map are separate concepts.'
    ),
    flow('SENSING & NETWORKING', 'ADC: analog to digital',
      'Watch an analog input become a sampled digital conversion result for CPU or DMA service.',
      'DATA · ADC',
      [
        node('Analog source', 'SENSOR / VOLTAGE', 'A sensor circuit provides an analog voltage within the MCU input range and reference constraints.'),
        node('ADC input', 'PIN + SAMPLE TIME', 'The selected pin/channel and sample time determine how the input is acquired by the ADC.'),
        node('Sample-and-hold', 'ANALOG FRONT END', 'The ADC samples the input and holds it long enough for conversion; source impedance affects settling.'),
        node('ADC conversion', 'SUCCESSIVE APPROXIMATION', 'The ADC converts the sampled voltage into a configured resolution digital code.'),
        node('Data register', 'ADC RESULT', 'The conversion result and status become available to software or the DMA request path.'),
        node('CPU or DMA → SRAM', 'RESULT HANDOFF', 'Software can read each result or DMA can store repeated conversions into a memory buffer.')
      ],
      [
        edge('Analog voltage', 'Input voltage must remain within allowed analog input/reference limits.', 'The sensor presents a changing voltage at the selected ADC channel.'),
        edge('Selected channel', 'Channel selection and sample time configure what voltage the ADC acquires.', 'The analog front end samples and holds the selected input.'),
        edge('Held voltage', 'The ADC converts the acquired sample according to resolution and reference.', 'The conversion engine produces a digital result.'),
        edge('Digital code + EOC', 'End-of-conversion status or request indicates a result is ready.', 'The ADC records the conversion value and may request DMA or interrupt service.'),
        edge('ADC DR → software/buffer', 'CPU reads or DMA writes depend on the chosen acquisition method.', 'The result is consumed by software or stored in SRAM for later processing.')
      ],
      'ADC measures voltage relative to its configured reference. Conversion code, sample time, source impedance, and reference quality all affect accuracy.'
    ),
    flow('SENSING & NETWORKING', 'CAN: application to bus',
      'Keep the STM32 bxCAN controller and external CAN transceiver as separate blocks.',
      'DATA · bxCAN',
      [
        node('Application', 'MESSAGE PRODUCER', 'The application chooses an identifier, payload, and transmission policy.'),
        node('bxCAN mailboxes', 'INTEGRATED CONTROLLER', 'The STM32L476 bxCAN peripheral handles frame formatting, arbitration, acknowledgement, and error state.'),
        node('CAN TX / RX pins', 'MCU SIGNALS', 'Alternate-function pins carry logic-level transmit and receive signals between the MCU and transceiver.'),
        node('CAN transceiver', 'PHYSICAL LAYER', 'An external transceiver converts MCU logic signals to differential CANH/CANL signaling and receives the bus state.'),
        node('CANH / CANL bus', 'DIFFERENTIAL PAIR', 'The physical bus uses differential signaling, termination, and topology appropriate to the network.'),
        node('Other ECU', 'BUS PARTICIPANT', 'Other nodes observe the frame; acceptance filters and identifier arbitration influence which frames are received or transmitted.')
      ],
      [
        edge('ID + DLC + payload', 'The application constructs a protocol-level message for the bus.', 'Software submits a CAN frame for transmission.'),
        edge('TX mailbox / request', 'The bxCAN peripheral is the controller; do not add a second separate controller block.', 'The integrated controller queues and prepares the frame.'),
        edge('TX logic signal', 'The MCU pins connect the controller logic to an external physical-layer transceiver.', 'bxCAN drives the selected transmit pin toward the transceiver.'),
        edge('Differential frame', 'The transceiver handles electrical signaling; the CAN controller handles protocol framing.', 'The external transceiver drives CANH/CANL onto the bus.'),
        edge('Arbitrated CAN frame', 'All network nodes observe bus traffic; identifier priority resolves simultaneous starts bit-by-bit.', 'The frame is visible to other ECUs on the shared bus.')
      ],
      'bxCAN is integrated in the STM32L476. A separate CAN transceiver is required for the differential physical bus; the MCU pins do not drive CANH/CANL directly.'
    ),
    flow('SENSING & NETWORKING', 'CAN receive interrupt',
      'Trace a received CAN frame through the transceiver, bxCAN FIFO/filtering, IRQ, and application service.',
      'DATA · bxCAN · IRQ',
      [
        node('CANH / CANL', 'PHYSICAL BUS', 'A differential frame arrives on the shared bus from another network node.'),
        node('CAN transceiver', 'PHYSICAL LAYER', 'The transceiver converts bus levels into MCU logic-level receive signaling.'),
        node('bxCAN receiver', 'INTEGRATED CONTROLLER', 'The STM32 bxCAN peripheral decodes the frame and checks protocol/error conditions.'),
        node('Filter + RX FIFO', 'FRAME QUEUE', 'Acceptance filters decide which messages enter a receive FIFO; FIFO capacity is finite.'),
        node('CAN IRQ → NVIC', 'INTERRUPT PATH', 'Configured FIFO/status conditions can request a CAN interrupt.'),
        node('ISR / HAL dispatch', 'SOFTWARE SERVICE', 'The interrupt handler drains or acknowledges FIFO/status according to the driver and reference manual.'),
        node('Application message', 'USER PROCESSING', 'Application code handles identifier, DLC, payload, and any higher-level protocol.')
      ],
      [
        edge('Differential frame', 'The bus carries a physical CAN frame; the MCU cannot connect directly to CANH/CANL.', 'The external bus signal reaches the transceiver.'),
        edge('RX logic signal', 'The transceiver converts electrical bus signaling to the controller receive pin level.', 'The transceiver presents the received signal to bxCAN.'),
        edge('Decoded frame', 'Controller hardware checks framing and updates receive/error state.', 'bxCAN reconstructs the frame and applies configured acceptance filtering.'),
        edge('Accepted frame → FIFO', 'Filters and FIFO overrun policy determine whether a message is queued.', 'The accepted frame is stored in the selected receive FIFO.'),
        edge('FIFO pending / status', 'Interrupt enables and NVIC configuration are required for CPU notification.', 'bxCAN requests service for the configured receive condition.'),
        edge('IRQ handler reads FIFO', 'Software must drain frames before FIFO overrun and follow register/driver semantics.', 'The handler retrieves the message and dispatches it to the application.')
      ],
      'CAN acceptance filters and finite FIFOs matter. The ISR should do bounded service and pass message data to application work safely.'
    ),
    flow('SENSING & NETWORKING', 'ADC + DMA sensor pipeline',
      'Model repeated ADC sampling into a circular SRAM buffer and block-wise CPU processing.',
      'DATA · ADC · DMA',
      [
        node('Sensor signal', 'ANALOG SOURCE', 'An analog sensor and front-end circuit provide the signal to be sampled.'),
        node('ADC trigger', 'TIMER / SOFTWARE', 'A timer or software trigger starts conversions at a deliberate sample rate.'),
        node('ADC conversion', 'DIGITAL SAMPLE', 'The ADC creates one digital sample for each completed conversion.'),
        node('DMA channel', 'DATA MOVEMENT', 'DMA transfers each result to the next configured buffer position.'),
        node('Circular SRAM buffer', 'SAMPLE BLOCKS', 'A circular buffer contains recent samples; half/full events define regions software can process.'),
        node('CPU algorithm', 'BLOCK PROCESSING', 'The CPU processes a block or half-buffer, rather than servicing every sample individually.'),
        node('Application output', 'DECISION / TELEMETRY', 'Processed results can drive control decisions, diagnostics, or communication.')
      ],
      [
        edge('Analog signal', 'Sensor bandwidth and analog front-end conditioning constrain useful sample rate.', 'The physical signal reaches the measurement input.'),
        edge('Sample trigger', 'A stable timer trigger can produce more regular sampling than a software loop.', 'The configured trigger starts an ADC conversion.'),
        edge('ADC code', 'Each conversion produces a discrete digital sample.', 'The ADC completes a sample and raises its DMA request.'),
        edge('Sample → buffer slot', 'DMA address increment and circular mode define the buffer traversal.', 'DMA stores the sample into the next SRAM slot.'),
        edge('Half / full buffer event', 'Software must finish before DMA reuses a region; use explicit ownership and timing.', 'DMA progress makes a block ready for processing.'),
        edge('Sample block', 'Block processing amortizes per-sample CPU overhead but does not remove deadlines.', 'The CPU processes the ready region and updates application state.')
      ],
      'DMA can capture regular sample blocks efficiently. Correctness still depends on sample timing, buffer ownership, and completing work before DMA wraps.'
    ),
    flow('FAULTS & POWER', 'Watchdog: timeout to reset',
      'Contrast a healthy refresh path with a stalled application that stops refreshing the watchdog.',
      'CONTROL · WATCHDOG',
      [
        node('Application health', 'SUPERVISOR', 'A supervisor decides whether critical tasks have completed correctly before a refresh is permitted.'),
        node('IWDG counter', 'INDEPENDENT WATCHDOG', 'The independent watchdog counts down from its configured reload value using its clock source.'),
        node('Refresh key', 'SERVICE ACTION', 'Software refreshes the watchdog only after it has evidence that the system is healthy.'),
        node('Timeout', 'NO REFRESH', 'If the counter expires, the watchdog requests a system reset according to configuration.'),
        node('Reset startup', 'RECOVERY', 'The MCU re-enters reset/startup; persistent fault evidence may be captured if designed appropriately.')
      ],
      [
        edge('Tasks complete / hang', 'Refreshing from an unconditional loop can hide a failed task; refresh policy should represent system health.', 'The application supervisor checks whether required work completed.'),
        edge('Refresh or no refresh', 'Refresh reloads the watchdog counter; a stalled system allows it to continue counting down.', 'Healthy software refreshes the IWDG; a hung path does not.'),
        edge('Counter reaches zero', 'Timeout behavior and option-byte configuration are device-specific.', 'The watchdog asserts its reset behavior after expiration.'),
        edge('Reset cause + boot', 'Reset-cause flags and retained diagnostic storage can help distinguish watchdog resets.', 'The MCU restarts and runs its startup sequence.')
      ],
      'A watchdog detects failure to refresh within a window; it cannot prove the software is correct. Refresh only after meaningful health checks.'
    ),
    flow('FAULTS & POWER', 'Fault: exception to diagnosis',
      'Map a processor fault condition to exception handling and a deliberate diagnose, record, or recovery strategy.',
      'CONTROL · FAULT',
      [
        node('Fault condition', 'CPU / MEMORY', 'Examples include an invalid access, bus error, or invalid instruction state. Exact fault classification depends on the event and enabled fault handlers.'),
        node('Configurable fault', 'MEMMANAGE / BUS / USAGE', 'MemManage, BusFault, and UsageFault can be enabled/configured; otherwise some conditions escalate to HardFault.'),
        node('HardFault', 'ESCALATION / HANDLER', 'HardFault handles severe faults and escalated configurable faults.'),
        node('Stacked context', 'EXCEPTION FRAME', 'Exception entry stacks core state when possible; fault status registers provide diagnostic clues.'),
        node('Fault handler', 'MINIMAL ISR', 'The handler should capture useful state safely and avoid complex operations that depend on corrupted context.'),
        node('Record / recover / reset', 'SYSTEM POLICY', 'The system can record a crash, enter a safe state, attempt recovery, or reset based on safety requirements.')
      ],
      [
        edge('Faulting instruction / access', 'The architecture classifies the event according to the core and memory-system condition.', 'The core detects a fault condition during execution.'),
        edge('Configurable fault request', 'Enable bits and fault type determine whether the configurable handler can run or escalates.', 'The core raises the corresponding exception or escalates it to HardFault.'),
        edge('Exception entry', 'The processor attempts to preserve execution context and enters the selected handler.', 'The core stacks context and branches to a fault vector when possible.'),
        edge('CFSR / HFSR / address registers', 'Status registers must be interpreted according to the Cortex-M4 and STM32L4 documentation.', 'The handler reads fault status and captured context for diagnosis.'),
        edge('Safe policy action', 'Recovery is application-specific; a fault handler should not assume returning is safe.', 'The system records evidence and selects a safe recovery or reset action.')
      ],
      'Fault names identify exception classes, not generic error callbacks. Enable configurable faults deliberately and decode status registers using the core/device manuals.'
    ),
    flow('FAULTS & POWER', 'Low-power entry and wake-up',
      'Distinguish entering a selected low-power mode from the event that can wake the core.',
      'CONTROL · POWER',
      [
        node('Application prepares', 'QUIESCE', 'Software saves required state and places peripherals in a mode appropriate for the selected low-power state.'),
        node('Select power mode', 'SLEEP / STOP / STANDBY', 'These modes differ in clock behavior, retained state, wake sources, and return path.'),
        node('WFI / WFE', 'CORE INSTRUCTION', 'The core executes the selected wait instruction after pending events and interrupt conditions are considered.'),
        node('Low-power state', 'POWER CONTROL', 'The selected mode controls which clocks/domains stop and what state is retained.'),
        node('Wake source', 'IRQ / EVENT / PIN', 'Only supported, enabled wake sources can leave the selected mode; exact options depend on the mode.'),
        node('Resume or reset', 'RETURN PATH', 'Some modes resume after sleep/stop; standby-style wake can behave as a reset/reboot path.'),
        node('Application continues', 'POST-WAKE', 'Software follows the mode-specific resume or startup path and restores any required state.')
      ],
      [
        edge('Save / quiesce', 'Entering low power safely requires software and peripheral preparation.', 'The application prepares state and configures wake sources.'),
        edge('Mode + wake configuration', 'Sleep, Stop, and Standby are not interchangeable power states.', 'Software selects a mode and its power-control configuration.'),
        edge('WFI / WFE', 'The instruction requests waiting; the configured power mode defines hardware effects.', 'The core enters the chosen wait/power state when conditions permit.'),
        edge('Reduced activity', 'Retained state and running clocks depend on the selected low-power mode.', 'The device remains in the configured low-power state.'),
        edge('Enabled wake event', 'Wake capability varies by mode and source; check the STM32L4 reference manual.', 'A valid interrupt, event, or pin condition requests wake-up.'),
        edge('Continue or restart', 'The CPU may resume execution or follow reset startup based on the mode.', 'The application restores required clocks/state and continues or restarts.')
      ],
      'Always name the mode. Sleep, Stop, and Standby have different clock, retention, wake, and reset semantics.'
    ),
    flow('SYSTEM VIEW', 'Complete data journey',
      'Bring the course together: sensor sampling, DMA buffering, CPU processing, GPIO output, UART telemetry, CAN reception, timer events, and RCC clocks.',
      'SYSTEM · DATA JOURNEY',
      [
        node('Sensor', 'ANALOG INPUT', 'A sensor produces a voltage that the board analog input can safely accept.'),
        node('ADC', 'CONVERSION', 'A configured ADC channel samples and digitizes the sensor voltage.'),
        node('DMA', 'TRANSFER ENGINE', 'DMA moves conversion results to memory under configured trigger/count/mode rules.'),
        node('SRAM buffer', 'SAMPLES', 'SRAM holds the sample block while DMA fills one region and the CPU handles another.'),
        node('Cortex-M4 CPU', 'PROCESS / CONTROL', 'The CPU executes the application algorithm and handles selected peripheral events.'),
        node('Algorithm', 'APPLICATION LOGIC', 'Software interprets samples, applies thresholds/filters, and chooses an action.'),
        node('GPIO → LED', 'OUTPUT ACTION', 'A memory-mapped GPIO write changes an output pin and drives the board circuit.'),
        node('UART → PC', 'TELEMETRY', 'The UART sends a configured serial frame for diagnostics or host communication.'),
        node('CAN transceiver', 'PHYSICAL INTERFACE', 'The external transceiver connects the bxCAN controller logic to the differential bus.'),
        node('bxCAN RX FIFO', 'NETWORK INPUT', 'The integrated controller accepts filtered frames into its finite receive FIFO.'),
        node('NVIC / timer', 'EVENT ROUTING', 'Configured timer/CAN events can request core service through the NVIC.'),
        node('RCC clocks', 'TIMING FOUNDATION', 'RCC clock configuration supplies the core, buses, and enabled peripheral clock domains.')
      ],
      [
        edge('Sensor voltage', 'The board circuit brings a physical quantity into an analog input range.', 'The sensor signal reaches the ADC input.'),
        edge('Digital conversion', 'The ADC samples on a trigger and produces a result code.', 'The ADC converts the analog input to a digital sample.'),
        edge('DMA request', 'DMA uses configured source/destination/count and trigger mapping.', 'The DMA engine transfers the result into SRAM.'),
        edge('Sample block', 'The buffer separates acquisition timing from application processing.', 'SRAM provides the block of samples to be consumed by the CPU.'),
        edge('Instructions + data', 'The Cortex-M4 executes compiled instructions and reads/writes memory-mapped peripheral state.', 'The CPU processes the samples and responds to configured events.'),
        edge('Decision', 'Application policy converts measurements into system behavior.', 'The algorithm selects an output, telemetry, or control action.'),
        edge('GPIO register write', 'GPIO output logic applies the configured pin state to its electrical pad.', 'The chosen output action appears on the LED or connected circuit.'),
        edge('UART frame', 'UART shifts a configured start/data/parity/stop frame on its TX pin.', 'Telemetry leaves the MCU for a connected host.'),
        edge('bxCAN TX/RX signals', 'The transceiver is a separate external physical-layer device.', 'Differential signaling connects the MCU CAN peripheral to the bus.'),
        edge('Accepted CAN message', 'bxCAN filters and finite FIFOs govern message acceptance and queuing.', 'A received frame becomes pending for software service.'),
        edge('IRQ request', 'Peripheral status, interrupt enables, NVIC state, and CPU masking all affect service.', 'A timer or CAN event can cause exception handling on the core.'),
        edge('Clock tree', 'Clock configuration is a prerequisite and timing basis; it is not the payload-data path.', 'RCC supplies configured clocks to the core, buses, and peripheral domains.', null, { from: 11, to: 4 })
      ],
      'One reliable mental model: C code configures hardware; CPU instructions and bus transactions access registers; peripherals create data/events; DMA and interrupts move work between hardware, memory, and software.'
    )
  ];

  const architectureContextByFlow = {
    'Power-on and reset': { label: 'POWER · RESET · STARTUP', nodes: [0, 1, 3, 5, 6, 7], edges: [0, 1, 3, 7] },
    'Flash and SRAM layout': { label: 'FLASH · SRAM · MEMORY ACCESS', nodes: [0, 5, 6, 7, 8], edges: [1, 2, 3] },
    'CPU instruction execution': { label: 'CPU · INSTRUCTION / DATA PATHS', nodes: [0, 5, 6, 7, 8], edges: [0, 1, 2, 3] },
    'GPIO output: register to pin': { label: 'SYSTEM BUS · AHB2 · GPIO PINS', nodes: [0, 3, 5, 9, 13], edges: [2, 4, 7] },
    'GPIO input: pin to program': { label: 'GPIO PINS · AHB2 · CPU', nodes: [0, 5, 9, 13], edges: [2, 4] },
    'RCC and clock flow': { label: 'RCC · CLOCK DOMAINS', nodes: [0, 3, 5, 9, 10, 11, 12], edges: [7] },
    'Timer: clock to event': { label: 'RCC · TIMER BUS · NVIC', nodes: [0, 1, 3, 5, 10, 11, 12], edges: [4, 6, 7] },
    'Interrupt: event to callback': { label: 'PERIPHERAL IRQ · NVIC · CPU', nodes: [0, 1, 5, 9, 11, 12], edges: [4, 6] },
    'DMA: hardware data movement': { label: 'DMA MASTER · MATRIX · MEMORY', nodes: [0, 4, 5, 7, 8, 9, 11], edges: [3, 4, 5] },
    'UART transmit': { label: 'CPU · APB · USART · TX PIN', nodes: [0, 5, 10, 11, 13], edges: [2, 4] },
    'UART receive with interrupt': { label: 'USART · NVIC · CPU', nodes: [0, 1, 5, 10, 11], edges: [4, 6] },
    'UART receive with DMA': { label: 'USART · DMA · SRAM', nodes: [0, 4, 5, 7, 10, 11], edges: [3, 4, 5] },
    'SPI full-duplex transfer': { label: 'CPU / DMA · APB · SPI PINS', nodes: [0, 4, 5, 10, 12, 13], edges: [2, 4, 5] },
    'I2C register read': { label: 'CPU / DMA · APB · I2C PINS', nodes: [0, 4, 5, 10, 11, 13], edges: [2, 4, 5] },
    'ADC: analog to digital': { label: 'ADC · DMA OPTION · SRAM', nodes: [0, 4, 5, 7, 10, 12, 13], edges: [3, 4, 5] },
    'CAN: application to bus': { label: 'CPU · APB · bxCAN · TRANSCEIVER', nodes: [0, 5, 10, 11, 13], edges: [2, 4] },
    'CAN receive interrupt': { label: 'bxCAN FIFO · NVIC · CPU', nodes: [0, 1, 5, 10, 11], edges: [4, 6] },
    'ADC + DMA sensor pipeline': { label: 'ADC · DMA · SRAM · CPU', nodes: [0, 4, 5, 7, 10, 12], edges: [3, 4, 5, 6] },
    'Watchdog: timeout to reset': { label: 'WATCHDOG · IRQ / RESET · RCC', nodes: [0, 1, 3, 5, 11], edges: [4, 6, 7] },
    'Fault: exception to diagnosis': { label: 'CPU · SYSTEM BUS · NVIC', nodes: [0, 1, 5, 6, 7], edges: [2, 3, 6] },
    'Low-power entry and wake-up': { label: 'RCC · POWER / CLOCK · WAKE IRQ', nodes: [0, 1, 3, 5, 9, 11, 12], edges: [6, 7] },
    'Complete data journey': { label: 'FULL MCU DATA JOURNEY', nodes: [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], edges: [1, 2, 3, 4, 5, 6, 7] }
  };

  const svgNS = 'http://www.w3.org/2000/svg';
  const nav = document.getElementById('flow-navigation');
  const architectureSvg = document.getElementById('architecture-diagram');
  const architectureTitle = document.getElementById('architecture-title');
  const architectureDescription = document.getElementById('architecture-description');
  const svg = document.getElementById('flow-diagram');
  const titleNode = document.getElementById('diagram-title');
  const descNode = document.getElementById('diagram-description');
  let selectedFlow = 0;
  let stepIndex = 0;
  let mode = 'manual';
  let timer = null;
  let isPlaying = false;
  let selectedNode = null;
  let selectedArchitectureNode = null;
  let showFlowPrompt = false;

  const byId = (id) => document.getElementById(id);
  const flowCountLabel = byId('flow-count');
  const speedSelect = byId('speed-select');
  const progressBar = document.querySelector('.step-progress');
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  function el(name, attributes = {}, parent = null) {
    const item = document.createElementNS(svgNS, name);
    Object.entries(attributes).forEach(([key, value]) => item.setAttribute(key, String(value)));
    if (parent) parent.appendChild(item);
    return item;
  }

  function text(parent, value, attributes = {}) {
    const item = el('text', attributes, parent);
    item.textContent = value;
    return item;
  }

  function signalColor(type) {
    if (type.includes('INTERRUPT') || type.includes('IRQ')) return '#ffb454';
    if (type.includes('DMA')) return '#55d69e';
    if (type.includes('CLOCK') || type.includes('RCC')) return '#b39aff';
    if (type.includes('UART')) return '#f5d76e';
    if (type.includes('CAN')) return '#f18ee0';
    if (type.includes('FAULT') || type.includes('RESET') || type.includes('WATCHDOG')) return '#ff777e';
    if (type.includes('CPU') || type.includes('INSTRUCTION')) return '#60a5fa';
    return '#39d9d2';
  }

  function makePositions(count) {
    const centers = [170, 500, 830];
    return Array.from({ length: count }, (_, index) => {
      const row = Math.floor(index / centers.length);
      const column = index % centers.length;
      const col = row % 2 === 0 ? column : centers.length - 1 - column;
      return { x: centers[col], y: 98 + row * 145 };
    });
  }

  function edgePath(from, to) {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const length = Math.hypot(dx, dy) || 1;
    const ux = dx / length;
    const uy = dy / length;
    const halfWidth = 95;
    const halfHeight = 34;
    const startScale = Math.min(halfWidth / Math.max(Math.abs(ux), 0.001), halfHeight / Math.max(Math.abs(uy), 0.001));
    const endScale = Math.min(halfWidth / Math.max(Math.abs(ux), 0.001), halfHeight / Math.max(Math.abs(uy), 0.001));
    const sx = from.x + ux * startScale;
    const sy = from.y + uy * startScale;
    const ex = to.x - ux * endScale;
    const ey = to.y - uy * endScale;
    const curve = Math.min(42, Math.abs(ex - sx) * 0.16);
    const bend = dy === 0 ? 0 : Math.sign(dy) * curve;
    return `M ${sx} ${sy} C ${sx + ux * curve} ${sy + bend}, ${ex - ux * curve} ${ey - bend}, ${ex} ${ey}`;
  }

  function renderNavigation() {
    nav.replaceChildren();
    let lastGroup = '';
    flows.forEach((item, index) => {
      if (item.group !== lastGroup) {
        const label = document.createElement('div');
        label.className = 'flow-group-label';
        label.textContent = item.group;
        nav.appendChild(label);
        lastGroup = item.group;
      }
      const entry = document.createElement('div');
      entry.className = 'flow-nav-item';
      const number = document.createElement('button');
      number.className = 'flow-number';
      number.type = 'button';
      number.dataset.flowIndex = String(index);
      number.textContent = String(index + 2).padStart(2, '0');
      number.setAttribute('aria-label', `Show prompt for flow ${number.textContent}: ${item.title}`);
      number.setAttribute('aria-controls', 'flow-prompt-panel');
      number.setAttribute('aria-expanded', String(showFlowPrompt && index === selectedFlow));
      number.setAttribute('aria-current', index === selectedFlow ? 'page' : 'false');
      number.addEventListener('click', () => selectFlow(index, true));

      const button = document.createElement('button');
      button.className = 'flow-link';
      button.type = 'button';
      button.dataset.flowIndex = String(index);
      button.setAttribute('aria-current', index === selectedFlow ? 'page' : 'false');
      const label = document.createElement('span');
      label.className = 'flow-link-title';
      label.textContent = item.title;
      button.append(label);
      button.addEventListener('click', () => selectFlow(index, false));
      entry.append(number, button);
      nav.appendChild(entry);
    });
    flowCountLabel.textContent = String(flows.length + 1).padStart(2, '0');
  }

  function selectFlow(index, revealPrompt) {
    stopPlayback();
    selectedFlow = index;
    stepIndex = 0;
    selectedNode = null;
    selectedArchitectureNode = null;
    showFlowPrompt = revealPrompt;
    render();
    closeNavigation();
    if (revealPrompt) {
      byId('flow-prompt-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function renderFlowPrompt(item) {
    const panel = byId('flow-prompt-panel');
    panel.hidden = !showFlowPrompt;
    if (!showFlowPrompt) return;

    const number = String(selectedFlow + 2).padStart(2, '0');
    byId('flow-prompt-number').textContent = `FLOW PROMPT · ${number} / ${String(flows.length + 1).padStart(2, '0')}`;
    byId('flow-prompt-title').textContent = item.title;
    byId('flow-prompt-text').textContent = `Follow this prompt: ${item.summary}`;
    byId('flow-prompt-focus').textContent = item.core;
  }

  function renderDiagram(item) {
    svg.classList.remove('architecture-diagram');
    svg.classList.toggle('is-playing', isPlaying);
    const positions = makePositions(item.nodes.length);
    const rowCount = Math.ceil(item.nodes.length / 3);
    const height = Math.max(410, 180 + rowCount * 145);
    const activeSignalColor = signalColor(item.type);
    svg.setAttribute('viewBox', `0 0 1000 ${height}`);
    svg.style.setProperty('--signal-color', activeSignalColor);
    titleNode.textContent = `${item.title} flow diagram`;
    descNode.textContent = `${item.nodes.map((part) => part.label).join(' to ')}. Select a component for details.`;
    svg.replaceChildren(titleNode, descNode);
    const defs = el('defs', {}, svg);
    const marker = el('marker', {
      id: 'arrow-muted', viewBox: '0 0 10 10', refX: '8.5', refY: '5',
      markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse'
    }, defs);
    el('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#8ea1b9' }, marker);
    const currentMarker = el('marker', {
      id: 'arrow-current', viewBox: '0 0 10 10', refX: '8.5', refY: '5',
      markerWidth: '8', markerHeight: '8', orient: 'auto-start-reverse'
    }, defs);
    el('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: activeSignalColor }, currentMarker);
    const completeMarker = el('marker', {
      id: 'arrow-complete', viewBox: '0 0 10 10', refX: '8.5', refY: '5',
      markerWidth: '8', markerHeight: '8', orient: 'auto-start-reverse'
    }, defs);
    el('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#55d69e' }, completeMarker);

    const edgesGroup = el('g', { 'aria-hidden': 'true' }, svg);
    item.edges.forEach((step, index) => {
      const pathData = edgePath(
        positions[step.from ?? index],
        positions[step.to ?? index + 1]
      );
      const isCurrent = stepIndex < item.edges.length && index === stepIndex;
      const path = el('path', {
        d: pathData,
        class: `diagram-edge${index < stepIndex ? ' is-complete' : ''}${isCurrent ? ' is-current' : ''}`,
        'marker-end': isCurrent ? 'url(#arrow-current)' : index < stepIndex ? 'url(#arrow-complete)' : 'url(#arrow-muted)'
      }, edgesGroup);
      if (isCurrent && isPlaying && !prefersReducedMotion.matches) {
        const particle = el('circle', { r: '4.5', fill: activeSignalColor, opacity: '.95' }, edgesGroup);
        const motion = el('animateMotion', {
          dur: `${Math.max(0.45, Number(speedSelect.value) / 1700)}s`,
          repeatCount: 'indefinite',
          path: pathData
        }, particle);
        motion.setAttribute('begin', '0s');
      }
      if (index < stepIndex) path.setAttribute('stroke-linecap', 'round');
    });

    const nodeGroup = el('g', {}, svg);
    item.nodes.forEach((part, index) => {
      const pos = positions[index];
      const group = el('g', {
        class: `diagram-node${index < stepIndex ? ' is-complete' : ''}${index === Math.min(stepIndex, item.nodes.length - 1) ? ' is-active' : ''}${selectedNode === index ? ' is-selected' : ''}`,
        transform: `translate(${pos.x - 105} ${pos.y - 36})`,
        tabindex: '0',
        role: 'button',
        'aria-label': `${part.label}: ${part.about}`
      }, nodeGroup);
      group.style.setProperty('--signal-color', activeSignalColor);
      el('rect', { width: '210', height: '72', rx: '11' }, group);
      el('circle', { class: 'node-port', cx: '9', cy: '36', r: '3.5' }, group);
      el('circle', { class: 'node-port', cx: '201', cy: '36', r: '3.5' }, group);
      text(group, part.kind, { x: '105', y: '25', class: 'node-kind' });
      text(group, part.label.length > 25 ? `${part.label.slice(0, 23)}…` : part.label, { x: '105', y: '49', class: 'node-title' });
      group.addEventListener('click', () => inspectNode(index));
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          inspectNode(index);
        }
      });
    });
  }

  function renderArchitectureDiagram(item) {
    const diagram = item.architecture;
    const selectedFlowContext = architectureContextByFlow[flows[selectedFlow].title];
    const activeSignalColor = signalColor(flows[selectedFlow].type);
    architectureSvg.setAttribute('viewBox', `0 0 ${diagram.width} ${diagram.height}`);
    architectureSvg.style.setProperty('--signal-color', activeSignalColor);
    architectureTitle.textContent = 'STM32L476RG Cortex-M4 bus and component block diagram';
    architectureDescription.textContent = `Logical MCU block diagram for ${flows[selectedFlow].title}. Highlighted paths show its relation to the on-chip architecture.`;
    architectureSvg.replaceChildren(architectureTitle, architectureDescription);
    byId('architecture-context-label').textContent = `PATH: ${selectedFlowContext.label}`;

    const defs = el('defs', {}, architectureSvg);
    const markerColors = {
      instruction: '#60a5fa',
      address: '#ffb454',
      data: '#39d9d2',
      dma: '#55d69e',
      event: '#ffb454',
      clock: '#b39aff',
      reset: '#ff777e'
    };
    Object.entries(markerColors).forEach(([kind, color]) => {
      const marker = el('marker', {
        id: `arrow-${kind}`, viewBox: '0 0 10 10', refX: '8.5', refY: '5',
        markerWidth: '8', markerHeight: '8', orient: 'auto-start-reverse'
      }, defs);
      el('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: color }, marker);
    });

    const enclosure = el('g', { 'aria-hidden': 'true' }, architectureSvg);
    el('rect', { x: 25, y: 28, width: 1430, height: 842, rx: 16, class: 'mcu-boundary' }, enclosure);
    text(enclosure, 'STM32L476RG · CORTEX-M4F SYSTEM', { x: 52, y: 58, class: 'mcu-boundary-label' });
    text(enclosure, 'LOGICAL ON-CHIP PATHS · NOT EXTERNAL ADDRESS/DATA PINS', { x: 852, y: 58, class: 'bus-caption' });

    const wiresGroup = el('g', { 'aria-hidden': 'true' }, architectureSvg);
    diagram.wires.forEach((wire) => {
      const points = wire.points.map(([x, y]) => `${x},${y}`).join(' ');
      const active = selectedFlowContext.edges.includes(wire.edge);
      const path = el('polyline', {
        points,
        class: `bus-wire ${wire.kind}${active ? ' is-context' : ''}`,
        'marker-end': `url(#arrow-${wire.kind})`,
        'data-edge': wire.edge
      }, wiresGroup);
      text(wiresGroup, wire.label, {
        x: wire.labelX,
        y: wire.labelY,
        class: 'bus-label',
        'text-anchor': wire.anchor || 'start'
      });
    });

    const componentsGroup = el('g', {}, architectureSvg);
    diagram.components.forEach((component) => {
      const part = item.nodes[component.node];
      const current = selectedFlowContext.nodes.includes(component.node);
      const selected = selectedArchitectureNode === component.node;
      const group = el('g', {
        class: `arch-block${current ? ' is-context' : ''}${selected ? ' is-selected' : ''}`,
        transform: `translate(${component.x} ${component.y})`,
        tabindex: '0',
        role: 'button',
        'aria-label': `${part.label}: ${part.about}`
      }, componentsGroup);
      group.style.setProperty('--signal-color', activeSignalColor);
      el('rect', { width: component.width, height: component.height, rx: '12' }, group);
      text(group, component.title, {
        x: component.width / 2,
        y: component.inner ? 28 : component.height / 2 - 5,
        class: 'arch-block-title'
      });
      text(group, component.subtitle, {
        x: component.width / 2,
        y: component.inner ? 46 : component.height / 2 + 15,
        class: 'arch-block-subtitle'
      });
      if (component.inner) {
        const startY = component.height - component.inner.length * 25 - 7;
        component.inner.forEach((label, index) => {
          const y = startY + index * 25;
          el('rect', {
            x: '10', y: y - 14, width: component.width - 20, height: '20', rx: '5',
            class: 'arch-inner-block'
          }, group);
          text(group, label, {
            x: component.width / 2,
            y,
            class: 'arch-inner-label'
          });
        });
      }
      group.addEventListener('click', () => inspectArchitectureNode(component.node));
      group.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          inspectArchitectureNode(component.node);
        }
      });
    });

    const noteGroup = el('g', { 'aria-hidden': 'true' }, architectureSvg);
    text(noteGroup, 'ADDRESS selects a target  ·  CONTROL qualifies the access  ·  DATA carries the value', {
      x: 52, y: 855, class: 'arch-note'
    });
    text(noteGroup, '* Trace pins/features depend on MCU/package and board; pin mapping is not shown to scale.', {
      x: 880, y: 855, class: 'arch-note'
    });
  }

  function inspectNode(index) {
    const item = flows[selectedFlow];
    const part = item.nodes[index];
    selectedNode = index;
    selectedArchitectureNode = null;
    updateInspector(part);
    renderArchitectureDiagram(architectureFlow);
    renderDiagram(item);
  }

  function inspectArchitectureNode(index) {
    selectedArchitectureNode = index;
    selectedNode = null;
    updateInspector(architectureFlow.nodes[index]);
    renderArchitectureDiagram(architectureFlow);
  }

  function updateInspector(part) {
    byId('inspector-heading').textContent = part.label;
    byId('inspector-description').textContent = part.about;
    const facts = byId('inspector-facts');
    facts.replaceChildren();
    part.facts.forEach(([label, value]) => {
      const row = document.createElement('div');
      row.className = 'inspector-fact';
      const factLabel = document.createElement('span');
      factLabel.textContent = label;
      const factValue = document.createElement('strong');
      factValue.textContent = value;
      row.append(factLabel, factValue);
      facts.appendChild(row);
    });
  }

  function updateStep(item) {
    const finished = stepIndex >= item.edges.length;
    const displayedEdge = stepIndex > 0 ? item.edges[stepIndex - 1] : null;
    const currentSource = displayedEdge ? item.nodes[item.edges.indexOf(displayedEdge)] : item.nodes[0];
    const currentDestination = displayedEdge
      ? item.nodes[item.edges.indexOf(displayedEdge) + 1]
      : item.nodes[0];
    const visibleStep = finished ? item.edges.length : stepIndex;

    byId('flow-number').textContent = `FLOW ${String(selectedFlow + 2).padStart(2, '0')} / ${flows.length + 1}`;
    byId('flow-title').textContent = item.title;
    byId('flow-summary').textContent = item.summary;
    byId('flow-type').textContent = item.type;
    byId('step-count').textContent = finished
      ? `STEP ${item.edges.length} / ${item.edges.length} · COMPLETE`
      : `STEP ${visibleStep} / ${item.edges.length}`;
    byId('step-mode-label').textContent = mode.toUpperCase();
    byId('step-title').textContent = finished
      ? 'Flow complete'
      : displayedEdge ? displayedEdge.title || `${currentSource.label} → ${currentDestination.label}` : 'Start the flow';
    byId('step-description').textContent = finished
      ? item.core
      : displayedEdge ? displayedEdge.description : 'Press Next to follow the signal one step at a time, or choose Auto play.';
    byId('step-source').textContent = displayedEdge ? currentSource.label : '—';
    byId('step-destination').textContent = displayedEdge ? currentDestination.label : '—';
    byId('step-data').textContent = displayedEdge ? displayedEdge.data : finished ? 'End of flow' : 'Ready when you are';
    byId('step-why').textContent = displayedEdge
      ? displayedEdge.why
      : finished ? 'Review the path, then choose another flow or reset to step through it again.' : item.core;
    byId('core-idea').textContent = item.core;
    byId('path-status').textContent = isPlaying
      ? 'Animation running'
      : finished ? 'Flow complete' : stepIndex ? 'Step paused' : 'Ready';
    byId('previous-button').disabled = stepIndex === 0;
    byId('next-button').disabled = finished;
    byId('play-button').disabled = mode !== 'auto' || isPlaying || finished;
    byId('pause-button').disabled = !isPlaying;
    const percent = item.edges.length ? Math.round((stepIndex / item.edges.length) * 100) : 100;
    byId('step-progress-fill').style.width = `${percent}%`;
    progressBar.setAttribute('aria-valuenow', String(percent));
    byId('diagram-description').textContent = `${item.nodes.map((part) => part.label).join(' to ')}. ${finished ? 'Flow complete.' : `Step ${visibleStep} of ${item.edges.length}.`}`;
    document.querySelectorAll('.flow-link').forEach((button) => {
      button.setAttribute('aria-current', Number(button.dataset.flowIndex) === selectedFlow ? 'page' : 'false');
    });
    document.querySelectorAll('.flow-number').forEach((button) => {
      const isSelected = Number(button.dataset.flowIndex) === selectedFlow;
      button.setAttribute('aria-current', isSelected ? 'page' : 'false');
      button.setAttribute('aria-expanded', String(showFlowPrompt && isSelected));
    });
  }

  function render() {
    const item = flows[selectedFlow];
    renderNavigation();
    renderFlowPrompt(item);
    renderArchitectureDiagram(architectureFlow);
    renderDiagram(item);
    updateStep(item);
    if (selectedNode !== null) inspectNode(selectedNode);
    else if (selectedArchitectureNode !== null) inspectArchitectureNode(selectedArchitectureNode);
    else {
      byId('inspector-heading').textContent = 'Explore a block';
      byId('inspector-description').textContent = 'Click a block in either diagram to inspect its role in the MCU and this flow.';
      byId('inspector-facts').replaceChildren();
    }
  }

  function stopPlayback() {
    if (timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
    isPlaying = false;
  }

  function advance() {
    const item = flows[selectedFlow];
    if (stepIndex >= item.edges.length) {
      stopPlayback();
      render();
      return;
    }
    stepIndex += 1;
    if (stepIndex >= item.edges.length) stopPlayback();
    render();
  }

  function startPlayback() {
    if (mode !== 'auto' || isPlaying) return;
    if (stepIndex >= flows[selectedFlow].edges.length) stepIndex = 0;
    isPlaying = true;
    render();
    timer = window.setInterval(advance, Number(speedSelect.value));
  }

  function closeNavigation() {
    byId('course-nav').classList.remove('open');
    byId('nav-toggle').setAttribute('aria-expanded', 'false');
    byId('nav-backdrop').hidden = true;
  }

  byId('next-button').addEventListener('click', () => {
    stopPlayback();
    advance();
  });
  byId('previous-button').addEventListener('click', () => {
    stopPlayback();
    stepIndex = Math.max(0, stepIndex - 1);
    render();
  });
  byId('reset-button').addEventListener('click', () => {
    stopPlayback();
    stepIndex = 0;
    selectedNode = null;
    selectedArchitectureNode = null;
    render();
  });
  byId('play-button').addEventListener('click', startPlayback);
  byId('pause-button').addEventListener('click', () => {
    stopPlayback();
    render();
  });
  byId('mode-select').addEventListener('change', (event) => {
    stopPlayback();
    mode = event.target.value;
    updateStep(flows[selectedFlow]);
  });
  speedSelect.addEventListener('change', () => {
    if (isPlaying) {
      stopPlayback();
      startPlayback();
    }
  });
  byId('theme-toggle').addEventListener('click', () => {
    const light = document.body.classList.toggle('light-theme');
    byId('theme-toggle').setAttribute('aria-label', light ? 'Switch to dark theme' : 'Switch to light theme');
    byId('theme-toggle').title = light ? 'Switch to dark theme' : 'Switch to light theme';
  });

  byId('flow-prompt-close').addEventListener('click', () => {
    showFlowPrompt = false;
    renderFlowPrompt(flows[selectedFlow]);
    document.querySelectorAll('.flow-number').forEach((button) => {
      button.setAttribute('aria-expanded', 'false');
    });
  });
  byId('nav-toggle').addEventListener('click', () => {
    const open = byId('course-nav').classList.toggle('open');
    byId('nav-toggle').setAttribute('aria-expanded', String(open));
    byId('nav-backdrop').hidden = !open;
  });
  byId('nav-backdrop').addEventListener('click', closeNavigation);
  document.addEventListener('keydown', (event) => {
    if (event.target instanceof HTMLElement && ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(event.target.tagName)) return;
    if (event.key === 'ArrowRight') byId('next-button').click();
    if (event.key === 'ArrowLeft') byId('previous-button').click();
    if (event.key === 'Escape') closeNavigation();
  });
  window.addEventListener('beforeunload', stopPlayback);

  if (flows.length !== 22) {
    throw new Error(`Visualise STM32 must have 22 detailed flows; found ${flows.length}.`);
  }
  render();
})();
