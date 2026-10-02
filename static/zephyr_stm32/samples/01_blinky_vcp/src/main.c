/*
 * Copyright (c) 2026 Apt Computing Labs
 * "Where Knowledge Meets Innovation"
 *
 * Mastering Zephyr RTOS on STM32
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/printk.h>

#define SLEEP_TIME_MS   1000

/* Devicetree node identifier for led0 alias */
#define LED0_NODE DT_ALIAS(led0)

#if !DT_NODE_HAS_STATUS(LED0_NODE, okay)
#error "Unsupported board: led0 devicetree alias is not defined!"
#endif

static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

int main(void)
{
    int ret;
    uint32_t count = 0;

    printk("\n=========================================\n");
    printk("  Mastering Zephyr RTOS on STM32 - Lab 1 \n");
    printk("  Board: %s\n", CONFIG_BOARD);
    printk("=========================================\n");

    if (!gpio_is_ready_dt(&led)) {
        printk("Error: GPIO device %s is not ready!\n", led.port->name);
        return 0;
    }

    ret = gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);
    if (ret < 0) {
        printk("Error %d: Failed to configure LED pin!\n", ret);
        return 0;
    }

    printk("LED initialized successfully on port %s, pin %d\n", 
           led.port->name, led.pin);

    while (1) {
        ret = gpio_pin_toggle_dt(&led);
        if (ret < 0) {
            printk("Error toggling LED!\n");
            return 0;
        }

        count++;
        printk("[%u s] Heartbeat Ping #%u - LED Toggled\n", 
               (count * SLEEP_TIME_MS) / 1000, count);

        k_msleep(SLEEP_TIME_MS);
    }

    return 0;
}
