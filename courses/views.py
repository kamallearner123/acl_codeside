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
from .models import Course, GroupFeedback

logger = logging.getLogger(__name__)

COURSE_GROUPS = (
    {
        'title': 'STM32+C+Rust+Autmotive',
        'icon': 'fa-microchip',
        'url_name': 'courses:stm32_automotive_group',
        'slugs': {
            'arm-cortex-m-architecture',
            'stm32-firmware-development-with-c',
            'embedded-rust-with-stm32',
            'visualise-stm32',
        },
    },
    {
        'title': 'Embedded Systems and IoT',
        'icon': 'fa-microchip',
        'slugs': {
            'c-system-programming',
            'exploring-zephyr-using-stm32',
            'iot-devices',
            'linux-os-concepts',
            'rust-programming',
        },
    },
    {
        'title': 'AI/ML and Agentic AI',
        'icon': 'fa-brain',
        'slugs': {
            'agentic-ai',
            'agentic-ai-learn-by-examples',
            'agentic-ai-python-automation',
            'ml-python',
            'python-programming',
        },
    },
    {
        'title': 'Cyber security',
        'icon': 'fa-shield-alt',
        'slugs': {'networking-security'},
    },
)

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

VISUALISE_STM32_DIR = os.path.join(settings.BASE_DIR, 'static', 'visualise_stm32')

def visualise_stm32_book_index(request):
    """Serve the interactive STM32 data-flow visualiser."""
    return serve(request, 'index.html', document_root=VISUALISE_STM32_DIR)

def visualise_stm32_book_serve(request, subpath):
    """Serve assets for the interactive STM32 data-flow visualiser."""
    if not subpath:
        subpath = 'index.html'
    elif subpath.endswith('/'):
        subpath = os.path.join(subpath, 'index.html')
    return serve(request, subpath, document_root=VISUALISE_STM32_DIR)

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

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        grouped_courses = [
            {
                'title': group['title'],
                'icon': group['icon'],
                'url_name': group.get('url_name'),
                'courses': [],
            }
            for group in COURSE_GROUPS
        ]
        groups_by_slug = {
            slug: group
            for definition, group in zip(COURSE_GROUPS, grouped_courses)
            for slug in definition['slugs']
        }
        other_courses = {
            'title': 'Other Courses',
            'icon': 'fa-book-open',
            'courses': [],
        }

        for course in context['courses']:
            group = groups_by_slug.get(course.slug, other_courses)
            group['courses'].append(course)

        context['course_groups'] = [
            group for group in grouped_courses if group['courses']
        ]
        if other_courses['courses']:
            context['course_groups'].append(other_courses)
        return context

class CourseDetailView(DetailView):
    model = Course
    template_name = 'courses/detail.html'
    context_object_name = 'course'


STM32_AUTOMOTIVE_COURSE_SLUGS = (
    'arm-cortex-m-architecture',
    'stm32-firmware-development-with-c',
    'embedded-rust-with-stm32',
    'visualise-stm32',
)

FEEDBACK_SESSION_KEY = 'pending_stm32_group_feedback'
FEEDBACK_CODE_TTL = 10 * 60
FEEDBACK_MAX_ATTEMPTS = 5
FEEDBACK_MAX_SENDS_PER_EMAIL_HOUR = 3
FEEDBACK_MAX_SENDS_PER_IP_HOUR = 10

STM32_GROUP_COURSES = (
    ('arm-cortex-m-architecture', 'ARM Cortex-M Architecture for Embedded Engineers'),
    ('stm32-firmware-development-with-c', 'Programming STM32 with C'),
    ('embedded-rust-with-stm32', 'Embedded Rust with STM32'),
    ('visualise-stm32', 'Visualise STM32'),
)


def stm32_automotive_group(request):
    courses_by_slug = {
        course.slug: course
        for course in Course.objects.filter(slug__in=STM32_AUTOMOTIVE_COURSE_SLUGS)
    }
    courses = [
        courses_by_slug[slug]
        for slug, _title in STM32_GROUP_COURSES
        if slug in courses_by_slug
    ]
    display_titles = dict(STM32_GROUP_COURSES)
    for course in courses:
        course.group_display_title = display_titles[course.slug]
    return render(request, 'courses/stm32_automotive_group.html', {
        'group_title': 'STM32+C+Rust+Autmotive',
        'group_courses': courses,
        'feedback_topics': GroupFeedback.Topic.choices,
        'feedback_courses': tuple((title, title) for _slug, title in STM32_GROUP_COURSES),
        'feedback_form_action': 'courses:stm32_group_feedback',
    })


def stm32_group_feedback(request):
    if request.method != 'POST':
        return redirect('courses:stm32_automotive_group')

    name = request.POST.get('name', '').strip()
    email = request.POST.get('email', '').strip().lower()
    course = request.POST.get('course', '').strip()
    topic = request.POST.get('topic', '').strip()
    message = request.POST.get('message', '').strip()

    try:
        validate_email(email)
    except ValidationError:
        messages.error(request, 'Please enter a valid email address.')
        return redirect('courses:stm32_automotive_group')

    valid_courses = {title for _slug, title in STM32_GROUP_COURSES}
    valid_topics = {value for value, _label in GroupFeedback.Topic.choices}
    if not name or len(name) > 100 or not message or len(message) > 3000:
        messages.error(request, 'Please provide your name and a message of no more than 3,000 characters.')
        return redirect('courses:stm32_automotive_group')
    if course and course not in valid_courses:
        messages.error(request, 'Please select a course from this group.')
        return redirect('courses:stm32_automotive_group')
    if topic not in valid_topics:
        messages.error(request, 'Please choose a feedback topic.')
        return redirect('courses:stm32_automotive_group')

    ip = request.META.get('REMOTE_ADDR', 'unknown')
    if (
        _over_limit(f'stm32_feedback_ip_{ip}', FEEDBACK_MAX_SENDS_PER_IP_HOUR)
        or _over_limit(f'stm32_feedback_email_{email}', FEEDBACK_MAX_SENDS_PER_EMAIL_HOUR)
    ):
        messages.error(request, 'Too many verification requests. Please try again later.')
        return redirect('courses:stm32_automotive_group')

    pending = {
        'name': name,
        'email': email,
        'course': course,
        'topic': topic,
        'message': message,
        'code': f'{secrets.randbelow(10**6):06d}',
        'expires': time.time() + FEEDBACK_CODE_TTL,
        'attempts': 0,
    }
    try:
        send_mail(
            'Verify your Apt Computing Labs group feedback',
            f'Hi {name},\n\nYour verification code is {pending["code"]}. '
            'It expires in 10 minutes. Your feedback is saved only after you verify this code.\n\n'
            'If you did not request this, you can ignore this email.\n\n— Apt Computing Labs',
            settings.DEFAULT_FROM_EMAIL,
            [email],
            fail_silently=False,
        )
    except Exception:
        logger.exception('Error sending STM32 group feedback verification email')
        messages.error(request, 'We could not send a verification email. Please check the address and try again.')
        return redirect('courses:stm32_automotive_group')

    request.session[FEEDBACK_SESSION_KEY] = pending
    messages.success(request, f'We sent a verification code to {email}. Your feedback is saved after verification.')
    return redirect('courses:stm32_group_feedback_verify')


def stm32_group_feedback_verify(request):
    pending = request.session.get(FEEDBACK_SESSION_KEY)
    if not pending:
        messages.error(request, 'There is no feedback awaiting verification. Please submit the form again.')
        return redirect('courses:stm32_automotive_group')

    if request.method == 'POST':
        if time.time() > pending['expires']:
            del request.session[FEEDBACK_SESSION_KEY]
            messages.error(request, 'The verification code expired. Please submit the feedback again.')
            return redirect('courses:stm32_automotive_group')

        pending['attempts'] += 1
        if pending['attempts'] > FEEDBACK_MAX_ATTEMPTS:
            del request.session[FEEDBACK_SESSION_KEY]
            messages.error(request, 'Too many incorrect attempts. Please submit the feedback again.')
            return redirect('courses:stm32_automotive_group')

        code = request.POST.get('code', '').strip()
        if not hmac.compare_digest(code, pending['code']):
            request.session[FEEDBACK_SESSION_KEY] = pending
            messages.error(request, 'That code did not match. Please try again.')
            return redirect('courses:stm32_group_feedback_verify')

        del request.session[FEEDBACK_SESSION_KEY]
        feedback = GroupFeedback.objects.create(
            name=pending['name'],
            email=pending['email'],
            course=pending['course'],
            topic=pending['topic'],
            message=pending['message'],
        )
        email_delivery_failed = False
        try:
            send_mail(
                'Verified feedback for STM32+C+Rust+Autmotive',
                f'Topic: {feedback.get_topic_display()}\n'
                f'Course: {feedback.course or "Whole group"}\n'
                f'Name: {feedback.name}\n'
                f'Email: {feedback.email} (verified)\n\n'
                f'{feedback.message}',
                settings.DEFAULT_FROM_EMAIL,
                ['kamal@aptcomputinglabs.com'],
                fail_silently=False,
            )
        except Exception:
            email_delivery_failed = True
            logger.exception('Verified STM32 group feedback was saved but ACL notification failed')

        try:
            send_mail(
                'We received your Apt Computing Labs feedback',
                f'Hi {feedback.name},\n\nYour email is verified and we have saved your feedback '
                f'about the STM32+C+Rust+Autmotive group.\n\n— Apt Computing Labs',
                settings.DEFAULT_FROM_EMAIL,
                [feedback.email],
                fail_silently=False,
            )
        except Exception:
            email_delivery_failed = True
            logger.exception('Verified STM32 group feedback was saved but confirmation email failed')

        if email_delivery_failed:
            messages.warning(request, 'Your verified feedback was saved, but an email notification could not be delivered.')
        else:
            messages.success(request, 'Email verified. Your feedback has been sent to Apt Computing Labs.')
        return redirect('courses:stm32_automotive_group')

    return render(request, 'courses/group_feedback_verify.html', {
        'email': pending['email'],
        'group_title': 'STM32+C+Rust+Autmotive',
    })

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
