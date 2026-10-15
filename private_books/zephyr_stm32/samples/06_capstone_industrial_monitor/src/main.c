/*
 * Copyright (c) 2026 Apt Computing Labs
 * "Where Knowledge Meets Innovation"
 *
 * Mastering Zephyr RTOS on STM32
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/drivers/watchdog.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/shell/shell.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(capstone_app, LOG_LEVEL_INF);

#define WDT_TIMEOUT_MS   2000
#define FEED_INTERVAL_MS 500

static const struct device *const wdt = DEVICE_DT_GET(DT_NODELABEL(iwdg));
static int wdt_channel_id;
static bool fault_injected = false;

static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

static int cmd_inject_fault(const struct shell *sh, size_t argc, char **argv)
{
    ARG_UNUSED(argc); ARG_UNUSED(argv);
    shell_print(sh, "[CRITICAL] Injecting deadlock fault! Watchdog will bite in 2000ms...");
    fault_injected = true;
    return 0;
}

SHELL_CMD_REGISTER(fault_inject, NULL, "Simulate fatal deadlock to trigger Watchdog", cmd_inject_fault);

int main(void)
{
    int ret;
    struct wdt_timeout_cfg wdt_config;

    LOG_INF("==================================================");
    LOG_INF("  Zephyr RTOS Industrial Capstone Gateway Initialized ");
    LOG_INF("==================================================");

    if (gpio_is_ready_dt(&led)) {
        gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);
    }

    if (!device_is_ready(wdt)) {
        LOG_ERR("Watchdog device %s not ready!", wdt->name);
        return 0;
    }

    wdt_config.flags = WDT_FLAG_RESET_SOC;
    wdt_config.window.min = 0;
    wdt_config.window.max = WDT_TIMEOUT_MS;
    wdt_config.callback = NULL;

    wdt_channel_id = wdt_install_timeout(wdt, &wdt_config);
    if (wdt_channel_id < 0) {
        LOG_ERR("Failed to install watchdog timeout: %d", wdt_channel_id);
        return 0;
    }

    ret = wdt_setup(wdt, WDT_OPT_PAUSE_IN_SLEEP);
    if (ret < 0) {
        LOG_ERR("Failed to start watchdog: %d", ret);
        return 0;
    }

    LOG_INF("Hardware Watchdog active (%d ms timeout). Supervisor running.", WDT_TIMEOUT_MS);

    while (1) {
        if (!fault_injected) {
            wdt_feed(wdt, wdt_channel_id);
            if (gpio_is_ready_dt(&led)) {
                gpio_pin_toggle_dt(&led);
            }
            LOG_INF("Supervisor heartbeat: Watchdog refreshed.");
        } else {
            LOG_WRN("Fault active! Watchdog starving...");
        }

        k_msleep(FEED_INTERVAL_MS);
    }

    return 0;
}
