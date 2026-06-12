from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.contrib import messages
from django.conf import settings
from .models import Contact
import logging

logger = logging.getLogger(__name__)

def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')
        
        # Save contact to database
        contact = Contact.objects.create(
            name=name,
            email=email,
            subject=subject,
            message=message
        )
        
        # Compose email to admin
        full_subject = f"Contact Form: {subject}"
        full_message = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        
        try:
            # Send email to admin immediately
            send_mail(
                full_subject,
                full_message,
                settings.DEFAULT_FROM_EMAIL,
                ['info@aptcomputinglabs.com', 'kamal@aptcomputinglabs.com'],
                fail_silently=False,
            )

            # Mark that admin email was sent
            contact.admin_email_sent = True
            contact.save()

            # Send confirmation email to user
            user_subject = 'Thank you for contacting Apt Computing Labs'
            user_message = (
                f"Hi {name},\n\n"
                "Thanks for reaching out to Apt Computing Labs. We have received your message and will get back to you shortly.\n\n"
                "Summary of your message:\n"
                f"Subject: {subject}\n"
                f"Message: {message}\n\n"
                "— Apt Computing Labs"
            )

            send_mail(
                user_subject,
                user_message,
                settings.DEFAULT_FROM_EMAIL,
                [email],
                fail_silently=False,
            )

            contact.user_email_sent = True
            contact.save()

            messages.success(request, 'Your message has been sent successfully! A confirmation email was sent to you.')
        except Exception as e:
            logger.exception('Error sending contact emails')
            # Save was already performed; inform user that the message was received
            messages.warning(request, 'Your message was received but we could not send the confirmation email.')
        
        return redirect('home')
    
    return render(request, 'contact/contact.html')
