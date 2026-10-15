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

#define STACK_SIZE 2048
#define PRODUCER_PRIORITY 6
#define CONSUMER_PRIORITY 4

struct sensor_data {
    uint32_t timestamp_ms;
    int32_t temperature_c_x10;
    int32_t humidity_pct;
    uint32_t sequence_id;
};

K_MSGQ_DEFINE(sensor_msgq, sizeof(struct sensor_data), 10, 4);

static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

void producer_thread_entry(void *arg1, void *arg2, void *arg3)
{
    ARG_UNUSED(arg1); ARG_UNUSED(arg2); ARG_UNUSED(arg3);
    uint32_t seq = 0;
    int32_t base_temp = 220;

    while (1) {
        struct sensor_data reading;
        reading.timestamp_ms = k_uptime_get_32();
        reading.temperature_c_x10 = base_temp + (seq % 15);
        reading.humidity_pct = 45 + (seq % 10);
        reading.sequence_id = ++seq;

        printk("[Producer] Publishing #%u (Temp: %d.%d C, Hum: %d%%)\n",
               reading.sequence_id,
               reading.temperature_c_x10 / 10,
               reading.temperature_c_x10 % 10,
               reading.humidity_pct);

        if (k_msgq_put(&sensor_msgq, &reading, K_MSEC(50)) != 0) {
            printk("[Producer] Warning: Queue full! Dropped.\n");
        }

        k_msleep(1000);
    }
}

void consumer_thread_entry(void *arg1, void *arg2, void *arg3)
{
    ARG_UNUSED(arg1); ARG_UNUSED(arg2); ARG_UNUSED(arg3);
    struct sensor_data item;
    int32_t temp_sum = 0;
    uint32_t count = 0;

    while (1) {
        if (k_msgq_get(&sensor_msgq, &item, K_FOREVER) == 0) {
            count++;
            temp_sum += item.temperature_c_x10;
            int32_t avg = temp_sum / count;

            printk("[Consumer] Processed #%u @ %u ms | Avg Temp: %d.%d C\n",
                   item.sequence_id, item.timestamp_ms, avg / 10, avg % 10);

            gpio_pin_set_dt(&led, 1);
            k_busy_wait(20000);
            gpio_pin_set_dt(&led, 0);
        }
    }
}

K_THREAD_DEFINE(producer_tid, STACK_SIZE, producer_thread_entry,
                NULL, NULL, NULL, PRODUCER_PRIORITY, 0, 0);

K_THREAD_DEFINE(consumer_tid, STACK_SIZE, consumer_thread_entry,
                NULL, NULL, NULL, CONSUMER_PRIORITY, 0, 0);

int main(void)
{
    printk("\n=== Zephyr Kernel Lab 3: Multithreading & Message Queues ===\n");
    if (gpio_is_ready_dt(&led)) {
        gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);
    }

    while (1) {
        k_sleep(K_FOREVER);
    }
    return 0;
}
