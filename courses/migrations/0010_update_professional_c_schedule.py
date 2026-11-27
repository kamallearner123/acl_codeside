from django.db import migrations

def update_professional_c_schedule(apps, schema_editor):
    Course = apps.get_model('courses', 'Course')
    Course.objects.filter(slug='professional-c-in-embedded-systems').update(
        timing="Mon to Fri • 6:00 PM – 7:30 PM IST",
        start_date="Starts 2nd Dec 2026 to 22nd Dec 2026 (Live Interactive Classes)",
    )

def reverse_update(apps, schema_editor):
    pass

class Migration(migrations.Migration):

    dependencies = [
        ('courses', '0009_groupfeedback'),
    ]

    operations = [
        migrations.RunPython(update_professional_c_schedule, reverse_update),
    ]
