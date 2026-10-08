from django.db import migrations


COURSE = {
    'title': 'Visualise STM32',
    'slug': 'visualise-stm32',
    'short_description': (
        'Explore STM32L476RG data and control flows through interactive, '
        'step-by-step visualisations.'
    ),
    'description': (
        '<p>Follow a C statement from the Cortex-M4 CPU through memory, '
        'buses, peripherals, interrupts, and DMA. Use the interactive '
        'visualiser to step through 22 embedded-system flows.</p>'
        '<p>Includes reset and startup, Flash and SRAM, GPIO, RCC, timers, '
        'NVIC interrupts, DMA, UART, SPI, I2C, ADC, bxCAN, watchdogs, '
        'faults, and low-power behavior.</p>'
    ),
    'duration_hours': 12,
    'skill_level': 'Beginner to Intermediate',
    'technologies': 'STM32L476RG, Cortex-M4, Embedded C, GPIO, DMA, UART, SPI, I2C, ADC, bxCAN',
    'training_mode': 'Self-paced interactive course',
}


def add_visualise_stm32_course(apps, schema_editor):
    Course = apps.get_model('courses', 'Course')
    Course.objects.using(schema_editor.connection.alias).get_or_create(
        slug=COURSE['slug'],
        defaults=COURSE,
    )


def remove_visualise_stm32_course(apps, schema_editor):
    Course = apps.get_model('courses', 'Course')
    Course.objects.using(schema_editor.connection.alias).filter(
        slug=COURSE['slug'],
        title=COURSE['title'],
    ).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('courses', '0005_course_training_mode'),
    ]

    operations = [
        migrations.RunPython(
            add_visualise_stm32_course,
            remove_visualise_stm32_course,
        ),
    ]
