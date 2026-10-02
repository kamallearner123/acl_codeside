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

#define EXT_LED_NODE    DT_ALIAS(ext_led)
#define EXT_BTN_NODE    DT_ALIAS(ext_button)

static const struct gpio_dt_spec ext_led = GPIO_DT_SPEC_GET(EXT_LED_NODE, gpios);
static const struct gpio_dt_spec ext_btn = GPIO_DT_SPEC_GET(EXT_BTN_NODE, gpios);

static struct gpio_callback button_cb_data;

void button_pressed_handler(const struct device *dev, struct gpio_callback *cb,
                            uint32_t pins)
{
    gpio_pin_toggle_dt(&ext_led);
    printk("[ISR] Button triggered on pin %d! External LED toggled.\n", ext_btn.pin);
}

int main(void)
{
    int ret;

    printk("\n=== Zephyr RTOS Lab 2: DTS Overlays & GPIO Interrupts ===\n");

    if (!gpio_is_ready_dt(&ext_led) || !gpio_is_ready_dt(&ext_btn)) {
        printk("Error: One or more GPIO peripheral ports not ready!\n");
        return 0;
    }

    ret = gpio_pin_configure_dt(&ext_led, GPIO_OUTPUT_INACTIVE);
    if (ret < 0) {
        printk("Failed to configure ext LED: %d\n", ret);
        return 0;
    }

    ret = gpio_pin_configure_dt(&ext_btn, GPIO_INPUT);
    if (ret < 0) {
        printk("Failed to configure ext Button: %d\n", ret);
        return 0;
    }

    ret = gpio_pin_interrupt_configure_dt(&ext_btn, GPIO_INT_EDGE_TO_ACTIVE);
    if (ret < 0) {
        printk("Failed to configure button interrupt: %d\n", ret);
        return 0;
    }

    gpio_init_callback(&button_cb_data, button_pressed_handler, BIT(ext_btn.pin));
    gpio_add_callback(ext_btn.port, &button_cb_data);

    printk("DTS Overlay configured successfully!\n");
    printk("LED on Port: %s Pin: %d | Button on Port: %s Pin: %d\n",
           ext_led.port->name, ext_led.pin, ext_btn.port->name, ext_btn.pin);

    while (1) {
        k_sleep(K_FOREVER);
    }

    return 0;
}
