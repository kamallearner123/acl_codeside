/*
 * Copyright (c) 2026 Apt Computing Labs
 * "Where Knowledge Meets Innovation"
 *
 * Mastering Zephyr RTOS on STM32
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/shell/shell.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(main_app, LOG_LEVEL_INF);

static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

static int cmd_led_on(const struct shell *sh, size_t argc, char **argv)
{
    ARG_UNUSED(argc); ARG_UNUSED(argv);
    gpio_pin_set_dt(&led, 1);
    shell_print(sh, "Onboard LED turned ON.");
    return 0;
}

static int cmd_led_off(const struct shell *sh, size_t argc, char **argv)
{
    ARG_UNUSED(argc); ARG_UNUSED(argv);
    gpio_pin_set_dt(&led, 0);
    shell_print(sh, "Onboard LED turned OFF.");
    return 0;
}

SHELL_STATIC_SUBCMD_SET_CREATE(sub_led,
    SHELL_CMD(on, NULL, "Turn onboard LED on", cmd_led_on),
    SHELL_CMD(off, NULL, "Turn onboard LED off", cmd_led_off),
    SHELL_SUBCMD_SET_END
);

SHELL_CMD_REGISTER(led, &sub_led, "Control onboard LED state", NULL);

int main(void)
{
    if (!gpio_is_ready_dt(&led)) {
        LOG_ERR("LED port not ready");
        return 0;
    }
    gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);

    LOG_INF("Zephyr Diagnostic Shell ready. Type 'help' or press TAB.");

    while (1) {
        k_sleep(K_FOREVER);
    }
    return 0;
}
