from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.contrib import messages
from django.conf import settings

def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')
        
        # Compose email
        full_subject = f"Contact Form: {subject}"
        full_message = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        
        try:
            send_mail(
                full_subject,
                full_message,
                settings.DEFAULT_FROM_EMAIL,
                ['info@aptcomputinglabs.com', 'kamal@aptcomputinglabs.com'],
                fail_silently=False,
            )
            messages.success(request, 'Your message has been sent successfully!')
        except Exception as e:
            messages.error(request, f'Failed to send message: {str(e)}')
        
        return redirect('home')
    
    return render(request, 'contact/contact.html')
