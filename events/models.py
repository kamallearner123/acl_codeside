from django.db import models

class Event(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    date = models.DateTimeField()
    location = models.CharField(max_length=200)
    description = models.TextField()
    registration_link = models.URLField(blank=True)
    image_url = models.URLField(blank=True, null=True)
    
    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"{self.title} - {self.date.date()}"
