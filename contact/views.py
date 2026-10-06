import hmac
import logging
import secrets
import time

from django.conf import settings
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.core.cache import cache
from django.core.mail import send_mail
from django.core.validators import validate_email
from django.shortcuts import redirect, render

from .models import Contact

logger = logging.getLogger(__name__)

SESSION_KEY = 'pending_contact'
CODE_TTL = 10 * 60
MAX_ATTEMPTS = 5
MAX_SENDS_PER_EMAIL_HOUR = 3
MAX_SENDS_PER_IP_HOUR = 10


def _client_ip(request):
    return request.META.get('REMOTE_ADDR', 'unknown')


def _over_limit(key, limit):
    count = cache.get(key, 0)
    if count >= limit:
        return True
    cache.set(key, count + 1, 3600)
    return False


def _send_code(pending):
    send_mail(
        'Your Apt Computing Labs verification code',
        f"Hi {pending['name']},\n\n"
        f"Your verification code is: {pending['code']}\n"
        "It expires in 10 minutes. If you did not request this, ignore this email.\n\n"
        "— Apt Computing Labs",
        settings.DEFAULT_FROM_EMAIL,
        [pending['email']],
        fail_silently=False,
    )


def _deliver(pending):
    contact = Contact.objects.create(
        name=pending['name'], email=pending['email'],
        subject=pending['subject'], message=pending['message'],
    )
    send_mail(
        f"Contact Form: {pending['subject']}",
        f"Name: {pending['name']}\n"
        f"Email: {pending['email']} (verified)\n"
        f"Company: {pending['company']}\n"
        f"Phone: {pending['phone']}\n\n"
        f"Message:\n{pending['message']}",
        settings.DEFAULT_FROM_EMAIL,
        ['kamal@aptcomputinglabs.com'],
        fail_silently=False,
    )
    contact.admin_email_sent = True
    contact.user_email_sent = True
    contact.save()


def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()[:100]
        email = request.POST.get('email', '').strip().lower()
        subject = request.POST.get('subject', '').strip()[:200]
        message = request.POST.get('message', '').strip()[:5000]
        company = request.POST.get('company', 'Not provided').strip()[:200]
        phone = request.POST.get('phone', 'Not provided').strip()[:50]

        try:
            validate_email(email)
        except ValidationError:
            messages.error(request, 'Please enter a valid email address.')
            return redirect('contact:contact')
        if not (name and subject and message):
            messages.error(request, 'Please fill in all required fields.')
            return redirect('contact:contact')

        if (_over_limit(f'contact_ip_{_client_ip(request)}', MAX_SENDS_PER_IP_HOUR)
                or _over_limit(f'contact_email_{email}', MAX_SENDS_PER_EMAIL_HOUR)):
            messages.error(request, 'Too many requests. Please try again later.')
            return redirect('contact:contact')

        pending = {
            'name': name, 'email': email, 'subject': subject,
            'message': message, 'company': company, 'phone': phone,
            'code': f'{secrets.randbelow(10**6):06d}',
            'expires': time.time() + CODE_TTL, 'attempts': 0,
        }
        try:
            _send_code(pending)
        except Exception:
            logger.exception('Error sending verification email')
            messages.error(request, 'We could not send a verification email. Please check the address and try again.')
            return redirect('contact:contact')

        request.session[SESSION_KEY] = pending
        messages.success(request, f'We sent a 6-digit verification code to {email}.')
        return redirect('contact:verify')

    return render(request, 'contact/contact.html')


def verify(request):
    pending = request.session.get(SESSION_KEY)
    if not pending:
        messages.error(request, 'No pending message. Please fill in the contact form.')
        return redirect('contact:contact')

    if request.method == 'POST':
        if time.time() > pending['expires']:
            del request.session[SESSION_KEY]
            messages.error(request, 'The code has expired. Please submit the form again.')
            return redirect('contact:contact')

        pending['attempts'] += 1
        if pending['attempts'] > MAX_ATTEMPTS:
            del request.session[SESSION_KEY]
            messages.error(request, 'Too many incorrect attempts. Please submit the form again.')
            return redirect('contact:contact')

        entered = request.POST.get('code', '').strip()
        if not hmac.compare_digest(entered, pending['code']):
            request.session[SESSION_KEY] = pending
            messages.error(request, 'Incorrect code. Please try again.')
            return redirect('contact:verify')

        del request.session[SESSION_KEY]
        try:
            _deliver(pending)
            messages.success(request, 'Email verified and your message has been sent. We will get back to you shortly.')
        except Exception:
            logger.exception('Error sending contact emails')
            messages.warning(request, 'Your message was received but we could not send the notification email.')
        return redirect('home')

    return render(request, 'contact/verify.html', {'email': pending['email']})
