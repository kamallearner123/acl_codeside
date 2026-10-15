/*
 * Copyright (c) 2026 Apt Computing Labs
 * "Where Knowledge Meets Innovation"
 *
 * Mastering Zephyr RTOS on STM32
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/drivers/pwm.h>
#include <zephyr/sys/printk.h>

#define PWM_PERIOD_NS   PWM_MSEC(20U)
#define STEP_NS         PWM_USEC(500U)

static const struct pwm_dt_spec pwm_led = PWM_DT_SPEC_GET(DT_ALIAS(pwm_led0));

int main(void)
{
    uint32_t pulse = 0;
    bool dir_up = true;

    printk("\n=== Zephyr Lab 4: I2C Sensor & Hardware PWM Engine ===\n");

    if (!pwm_is_ready_dt(&pwm_led)) {
        printk("Error: PWM device %s not ready!\n", pwm_led.dev->name);
        return 0;
    }

    printk("Breathing LED initialized on %s (channel %d)\n",
           pwm_led.dev->name, pwm_led.channel);

    while (1) {
        pwm_set_pulse_dt(&pwm_led, pulse);

        if (dir_up) {
            pulse += STEP_NS;
            if (pulse >= PWM_PERIOD_NS) {
                pulse = PWM_PERIOD_NS;
                dir_up = false;
            }
        } else {
            if (pulse <= STEP_NS) {
                pulse = 0;
                dir_up = true;
            } else {
                pulse -= STEP_NS;
            }
        }

        k_msleep(25);
    }

    return 0;
}
