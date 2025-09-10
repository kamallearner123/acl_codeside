from django.db import models


class Course(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    duration_hours = models.PositiveIntegerField(default=0)
    image = models.URLField(blank=True, null=True)

    def __str__(self):
        return self.title


class Event(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='events')
    title = models.CharField(max_length=200)
    date = models.DateField()
    location = models.CharField(max_length=200, blank=True)
    youtube_link = models.URLField(blank=True)

    def __str__(self):
        return f"{self.title} ({self.date})"
