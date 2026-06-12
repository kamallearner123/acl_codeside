

# Create your models here.
from django.db import models

class Contact(models.Model):

    name = models.CharField(max_length=100)

    email = models.EmailField()

    subject = models.CharField(max_length=200)

    message = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)
    
    admin_email_sent = models.BooleanField(default=False)
    
    user_email_sent = models.BooleanField(default=False)

    def __str__(self):
        return self.name
