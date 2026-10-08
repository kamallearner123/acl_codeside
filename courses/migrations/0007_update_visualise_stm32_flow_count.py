from django.db import migrations


OLD_DESCRIPTION = (
    '<p>Follow a C statement from the Cortex-M4 CPU through memory, '
    'buses, peripherals, interrupts, and DMA. Use the interactive '
    'visualiser to step through 22 embedded-system flows.</p>'
    '<p>Includes reset and startup, Flash and SRAM, GPIO, RCC, timers, '
    'NVIC interrupts, DMA, UART, SPI, I2C, ADC, bxCAN, watchdogs, '
    'faults, and low-power behavior.</p>'
)
NEW_DESCRIPTION = (
    '<p>Follow a C statement from the Cortex-M4 CPU through memory, '
    'buses, peripherals, interrupts, and DMA. Use the interactive '
    'visualiser to step through 23 embedded-system flows, including a '
    'detailed Cortex-M bus and component block diagram.</p>'
    '<p>Includes reset and startup, Flash and SRAM, GPIO, RCC, timers, '
    'NVIC interrupts, DMA, UART, SPI, I2C, ADC, bxCAN, watchdogs, '
    'faults, and low-power behavior.</p>'
)


def update_course_description(apps, schema_editor):
    Course = apps.get_model('courses', 'Course')
    Course.objects.using(schema_editor.connection.alias).filter(
        slug='visualise-stm32',
        description=OLD_DESCRIPTION,
    ).update(description=NEW_DESCRIPTION)


def restore_course_description(apps, schema_editor):
    Course = apps.get_model('courses', 'Course')
    Course.objects.using(schema_editor.connection.alias).filter(
        slug='visualise-stm32',
        description=NEW_DESCRIPTION,
    ).update(description=OLD_DESCRIPTION)


class Migration(migrations.Migration):

    dependencies = [
        ('courses', '0006_visualise_stm32_course'),
    ]

    operations = [
        migrations.RunPython(
            update_course_description,
            restore_course_description,
        ),
    ]
