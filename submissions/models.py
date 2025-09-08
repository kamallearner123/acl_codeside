from django.db import models
from django.contrib.auth.models import User
from questions.models import Question


class Submission(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('accepted', 'Accepted'),
        ('wrong_answer', 'Wrong Answer'),
        ('runtime_error', 'Runtime Error'),
        ('time_limit_exceeded', 'Time Limit Exceeded'),
        ('compilation_error', 'Compilation Error'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='submissions')
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='submissions')
    code = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Execution details
    runtime = models.FloatField(null=True, blank=True, help_text="Runtime in seconds")
    memory_usage = models.IntegerField(null=True, blank=True, help_text="Memory usage in KB")
    
    # Results
    test_cases_passed = models.PositiveIntegerField(default=0)
    total_test_cases = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    
    # Metadata
    submitted_at = models.DateTimeField(auto_now_add=True)
    language = models.CharField(max_length=20, default='python')
    
    class Meta:
        ordering = ['-submitted_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.question.title} ({self.get_status_display()})"
    
    @property
    def is_successful(self):
        return self.status == 'accepted'
    
    @property
    def success_percentage(self):
        if self.total_test_cases == 0:
            return 0
        return round((self.test_cases_passed / self.total_test_cases) * 100, 1)
