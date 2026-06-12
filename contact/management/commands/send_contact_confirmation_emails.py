from django.core.management.base import BaseCommand
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
from contact.models import Contact

class Command(BaseCommand):
    help = 'Send confirmation emails to users who submitted contact form (within 24 hours)'

    def handle(self, *args, **options):
        # Get contacts that haven't received confirmation email yet
        # Within the last 24 hours
        twenty_four_hours_ago = timezone.now() - timedelta(hours=24)
        
        contacts = Contact.objects.filter(
            user_email_sent=False,
            admin_email_sent=True,
            created_at__gte=twenty_four_hours_ago
        )
        
        count = 0
        for contact in contacts:
            try:
                # Compose confirmation email to user
                user_subject = "We Received Your Message - Apt Computing Labs"
                user_message = f"""Hi {contact.name},

Thank you for contacting Apt Computing Labs!

We have received your message and appreciate you reaching out. Our team will review your inquiry and get back to you within 24 hours.

Your Message Details:
Subject: {contact.subject}
Message: {contact.message}

Best regards,
Apt Computing Labs Team
info@aptcomputinglabs.com
+91 9739858111"""
                
                # Send confirmation email to user
                send_mail(
                    user_subject,
                    user_message,
                    settings.DEFAULT_FROM_EMAIL,
                    [contact.email],
                    fail_silently=False,
                )
                
                # Mark that user email was sent
                contact.user_email_sent = True
                contact.save()
                
                count += 1
                self.stdout.write(self.style.SUCCESS(f'✓ Sent confirmation to {contact.email}'))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'✗ Failed to send to {contact.email}: {str(e)}'))
        
        self.stdout.write(self.style.SUCCESS(f'\nTotal confirmation emails sent: {count}'))
