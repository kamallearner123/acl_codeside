from django.views.generic import ListView, DetailView
from django.shortcuts import get_object_or_404, redirect, render
from django.http import Http404
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.core.mail import send_mail
from django.core.validators import validate_email
from django.contrib import messages
import hmac
import os
import secrets
import time
import logging
from django.conf import settings
from django.views.static import serve
from .models import Course

logger = logging.getLogger(__name__)

ZEPHYR_BOOK_DIR = os.path.join(settings.BASE_DIR, 'static', 'zephyr_stm32')

STM32_C_BOOK_DIR = os.path.join(settings.BASE_DIR, 'static', 'stm32_c_firmware')

def stm32_c_book_index(request):
    """Serve the root index of the Programming STM32 with C digital book."""
    return serve(request, 'index.html', document_root=STM32_C_BOOK_DIR)

def stm32_c_book_serve(request, subpath):
    """Serve chapters, stylesheets, scripts and assets for the Programming STM32 with C book."""
    if not subpath:
        subpath = 'index.html'
    elif subpath.endswith('/'):
        subpath = os.path.join(subpath, 'index.html')
    return serve(request, subpath, document_root=STM32_C_BOOK_DIR)

STM32_RUST_BOOK_DIR = os.path.join(settings.BASE_DIR, 'static', 'stm32_rust_firmware')

def stm32_rust_book_index(request):
    """Serve the root index of the Embedded Rust with STM32 digital book."""
    return serve(request, 'index.html', document_root=STM32_RUST_BOOK_DIR)

def stm32_rust_book_serve(request, subpath):
    """Serve chapters, stylesheets, scripts and assets for the Embedded Rust with STM32 book."""
    if not subpath:
        subpath = 'index.html'
    elif subpath.endswith('/'):
        subpath = os.path.join(subpath, 'index.html')
    return serve(request, subpath, document_root=STM32_RUST_BOOK_DIR)

CORTEXM_BOOK_DIR = os.path.join(settings.BASE_DIR, 'static', 'cortexm_architecture')

def cortexm_book_index(request):
    """Serve the root index of the ARM Cortex-M Architecture digital book."""
    return serve(request, 'index.html', document_root=CORTEXM_BOOK_DIR)

def cortexm_book_serve(request, subpath):
    """Serve chapters, stylesheets, scripts and assets for the ARM Cortex-M Architecture book."""
    if not subpath:
        subpath = 'index.html'
    elif subpath.endswith('/'):
        subpath = os.path.join(subpath, 'index.html')
    return serve(request, subpath, document_root=CORTEXM_BOOK_DIR)

def zephyr_book_index(request):
    """Serve the root index of the Exploring Zephyr using STM32 digital book."""
    return serve(request, 'index.html', document_root=ZEPHYR_BOOK_DIR)

def zephyr_book_serve(request, subpath):
    """Serve chapters, stylesheets, scripts, images and assets for the Zephyr digital book."""
    if not subpath:
        subpath = 'index.html'
    elif subpath.endswith('/'):
        subpath = os.path.join(subpath, 'index.html')
    return serve(request, subpath, document_root=ZEPHYR_BOOK_DIR)

class CourseListView(ListView):
    model = Course
    template_name = 'courses/list.html'
    context_object_name = 'courses'
    
    def get_queryset(self):
        """Return courses deduplicated by title (case-insensitive), keeping the first occurrence."""
        qs = list(super().get_queryset().order_by('title'))
        seen = set()
        unique = []
        for c in qs:
            key = (c.title or '').strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

class CourseDetailView(DetailView):
    model = Course
    template_name = 'courses/detail.html'
    context_object_name = 'course'

def removed_course(request):
    raise Http404

SESSION_KEY = 'pending_enrollment'
CODE_TTL = 10 * 60
MAX_ATTEMPTS = 5
MAX_SENDS_PER_EMAIL_HOUR = 3
MAX_SENDS_PER_IP_HOUR = 10


def _over_limit(key, limit):
    count = cache.get(key, 0)
    if count >= limit:
        return True
    cache.set(key, count + 1, 3600)
    return False


def _deliver_enrollment(pending, course):
    name, email = pending['name'], pending['email']
    send_mail(
        f"ACL: {course.title}: Participant registration",
        f"Course: {course.title}\n"
        f"Name: {name}\n"
        f"Email: {email} (verified)\n"
        f"Phone: {pending['phone']}\n"
        f"Experience: {pending['experience']}\n",
        settings.DEFAULT_FROM_EMAIL,
        ['kamal@aptcomputinglabs.com'],
        fail_silently=False,
    )
    send_mail(
        f"Registration Confirmation: {course.title}",
        f"Hi {name},\n\n"
        f"Thank you for registering for the course '{course.title}'.\n"
        "We have received your details and our team will get in touch with you shortly.\n\n"
        "Registration Details:\n"
        f"Phone: {pending['phone']}\n"
        f"Experience: {pending['experience']}\n\n"
        "— Apt Computing Labs",
        settings.DEFAULT_FROM_EMAIL,
        [email],
        fail_silently=False,
    )


def enroll_course(request, slug):
    course = get_object_or_404(Course, slug=slug)
    if request.method != 'POST':
        return redirect('courses:detail', slug=slug)

    name = request.POST.get('name', '').strip()[:100]
    email = request.POST.get('email', '').strip().lower()
    phone = request.POST.get('phone', '').strip()[:50]
    experience = request.POST.get('experience', '').strip()[:1000]

    try:
        validate_email(email)
    except ValidationError:
        messages.error(request, 'Please enter a valid email address.')
        return redirect('courses:detail', slug=slug)
    if not (name and phone and experience):
        messages.error(request, 'Please fill in all required fields.')
        return redirect('courses:detail', slug=slug)

    ip = request.META.get('REMOTE_ADDR', 'unknown')
    if _over_limit(f'enroll_ip_{ip}', MAX_SENDS_PER_IP_HOUR) or _over_limit(f'enroll_email_{email}', MAX_SENDS_PER_EMAIL_HOUR):
        messages.error(request, 'Too many requests. Please try again later.')
        return redirect('courses:detail', slug=slug)

    pending = {
        'slug': slug, 'name': name, 'email': email, 'phone': phone,
        'experience': experience,
        'code': f'{secrets.randbelow(10**6):06d}',
        'expires': time.time() + CODE_TTL, 'attempts': 0,
    }
    try:
        send_mail(
            'Your Apt Computing Labs verification code',
            f"Hi {name},\n\nYour verification code to enroll in '{course.title}' is: {pending['code']}\n"
            "It expires in 10 minutes. If you did not request this, ignore this email.\n\n"
            "— Apt Computing Labs",
            settings.DEFAULT_FROM_EMAIL,
            [email],
            fail_silently=False,
        )
    except Exception:
        logger.exception('Error sending enrollment verification email')
        messages.error(request, 'We could not send a verification email. Please check the address and try again.')
        return redirect('courses:detail', slug=slug)

    request.session[SESSION_KEY] = pending
    messages.success(request, f'We sent a 6-digit verification code to {email}.')
    return redirect('courses:enroll_verify', slug=slug)


def enroll_verify(request, slug):
    course = get_object_or_404(Course, slug=slug)
    pending = request.session.get(SESSION_KEY)
    if not pending or pending.get('slug') != slug:
        messages.error(request, 'No pending enrollment. Please fill in the enrollment form.')
        return redirect('courses:detail', slug=slug)

    if request.method == 'POST':
        if time.time() > pending['expires']:
            del request.session[SESSION_KEY]
            messages.error(request, 'The code has expired. Please enroll again.')
            return redirect('courses:detail', slug=slug)

        pending['attempts'] += 1
        if pending['attempts'] > MAX_ATTEMPTS:
            del request.session[SESSION_KEY]
            messages.error(request, 'Too many incorrect attempts. Please enroll again.')
            return redirect('courses:detail', slug=slug)

        if not hmac.compare_digest(request.POST.get('code', '').strip(), pending['code']):
            request.session[SESSION_KEY] = pending
            messages.error(request, 'Incorrect code. Please try again.')
            return redirect('courses:enroll_verify', slug=slug)

        del request.session[SESSION_KEY]
        try:
            _deliver_enrollment(pending, course)
            messages.success(request, 'Email verified and your registration has been submitted! A confirmation email has been sent to you.')
        except Exception:
            logger.exception('Error sending enrollment emails')
            messages.warning(request, 'Your registration was received but we could not send the confirmation email.')
        return redirect('courses:detail', slug=slug)

    return render(request, 'courses/enroll_verify.html', {'email': pending['email'], 'course': course})
