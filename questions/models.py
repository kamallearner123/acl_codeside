from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse


class Question(models.Model):
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ]
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    description = models.TextField()
    difficulty = models.CharField(max_length=10, choices=DIFFICULTY_CHOICES, default='easy')
    
    # Example input/output
    example_input = models.TextField(help_text="Example input for the problem")
    example_output = models.TextField(help_text="Expected output for the example")
    
    # Constraints and hints
    constraints = models.TextField(blank=True, help_text="Problem constraints")
    hints = models.TextField(blank=True, help_text="Hints to solve the problem")
    
    # Template code
    template_code = models.TextField(
        default="def solution():\n    # Write your code here\n    pass",
        help_text="Template code for users to start with"
    )
    
    # Test cases (JSON format)
    test_cases = models.JSONField(
        default=list,
        help_text="Test cases in JSON format: [{'input': '...', 'expected': '...'}, ...]"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    # Statistics
    total_submissions = models.PositiveIntegerField(default=0)
    successful_submissions = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['difficulty', 'title']
    
    def __str__(self):
        return f"{self.title} ({self.get_difficulty_display()})"
    
    def get_absolute_url(self):
        return reverse('questions:detail', kwargs={'slug': self.slug})
    
    @property
    def success_rate(self):
        if self.total_submissions == 0:
            return 0
        return round((self.successful_submissions / self.total_submissions) * 100, 1)


class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    questions = models.ManyToManyField(Question, related_name='tags', blank=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['name']
