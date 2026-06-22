from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

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

class Rating(models.Model):
    tool = models.ForeignKey(DevTool, on_delete=models.CASCADE, related_name='ratings')
    stars = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.tool.name} - {self.stars} Stars"
