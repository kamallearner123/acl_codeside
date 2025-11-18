from django.db import models


class CodeExecution(models.Model):
    """Model to store code execution history"""
    code = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    execution_output = models.TextField(blank=True)
    miri_trace = models.JSONField(null=True, blank=True)
    status = models.CharField(max_length=50, default='pending')
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Execution {self.id} - {self.created_at}"
