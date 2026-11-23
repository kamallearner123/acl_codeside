

# Create your models here.
from django.db import models

class Contact(models.Model):

    name = models.CharField(max_length=100)

    email = models.EmailField()

    subject = models.CharField(max_length=200)

    message = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)
    
    dpdp_consent = models.BooleanField(default=False, help_text="Consent given under DPDP Act 2023")
    dpdp_consent_timestamp = models.DateTimeField(null=True, blank=True)
    
    admin_email_sent = models.BooleanField(default=False)
    
    user_email_sent = models.BooleanField(default=False)

    def __str__(self):
        return self.name
