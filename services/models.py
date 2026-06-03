from django.db import models

class Service(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    icon_class = models.CharField(max_length=50, blank=True, help_text="FontAwesome class e.g. 'fas fa-briefcase'")
    short_description = models.CharField(max_length=500)
    description = models.TextField()
    is_featured = models.BooleanField(default=False)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['order', 'title']

    def __str__(self):
        return self.title
