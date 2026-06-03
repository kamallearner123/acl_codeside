from django.db import models
from django.utils import timezone

class Post(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    author = models.CharField(max_length=100)
    excerpt = models.CharField(max_length=500)
    content = models.TextField()
    published_date = models.DateTimeField(default=timezone.now)
    image_url = models.URLField(blank=True)

    class Meta:
        ordering = ['-published_date']

    def __str__(self):
        return self.title
