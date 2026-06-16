from django.db import models


class DevTool(models.Model):
    CATEGORY_CHOICES = [
        ('debugging', 'Debugging Tools'),
        ('security', 'Security Tools'),
        ('observability', 'Observability Tools'),
        ('general', 'General Tools'),
         ('ai', 'AI Tools'),
    ]
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    github_url = models.URLField(blank=True, help_text="Link to the open-source repository")
    download_url = models.URLField(blank=True, help_text="Direct link to download the executable file")
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default='general')

    def __str__(self):
        return self.name
