from django.db import models


class Course(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    short_description = models.CharField(max_length=500, blank=True)
    description = models.TextField(blank=True)
    duration_weeks = models.PositiveIntegerField(default=0)
    duration_hours = models.PositiveIntegerField(default=0)
    skill_level = models.CharField(max_length=50, blank=True)
    technologies = models.CharField(max_length=200, blank=True, help_text="Comma separated list, e.g., 'Rust, Linux'")
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


class Review(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='reviews')
    reviewer_name = models.CharField(max_length=100)
    rating = models.PositiveIntegerField(default=5, help_text="Rating from 1 to 5")
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.rating}/5 by {self.reviewer_name} for {self.course.title}"
