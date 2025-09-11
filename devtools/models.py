from django.db import models


class DevTool(models.Model):
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    link = models.URLField(blank=True)

    def __str__(self):
        return self.name
