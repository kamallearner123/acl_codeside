from django.views.generic import ListView, DetailView
from django.shortcuts import get_object_or_404, redirect
from django.core.mail import send_mail
from django.contrib import messages
from django.conf import settings
import logging
from .models import Course

logger = logging.getLogger(__name__)

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

def enroll_course(request, slug):
    if request.method == 'POST':
        course = get_object_or_404(Course, slug=slug)
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        experience = request.POST.get('experience')

        subject = f"ACL: {course.title}: Participant registration"
        message = (
            f"Course: {course.title}\n"
            f"Name: {name}\n"
            f"Email: {email}\n"
            f"Phone: {phone}\n"
            f"Experience: {experience}\n"
        )
        
        try:
            # Send email to admin
            send_mail(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                ['kamal@aptcomputinglabs.com'],
                fail_silently=False,
            )

            # Send reply to participant
            user_subject = f"Registration Confirmation: {course.title}"
            user_message = (
                f"Hi {name},\n\n"
                f"Thank you for registering for the course '{course.title}'.\n"
                "We have received your details and our team will get in touch with you shortly.\n\n"
                "Upcoming Info Session:\n"
                "Sunday, September 6 | 6:00–7:00 PM IST\n"
                "Join the Google Meet to know more about the course: https://meet.google.com/ecn-rkai-bra\n\n"
                "Registration Details:\n"
                f"Phone: {phone}\n"
                f"Experience: {experience}\n\n"
                "— Apt Computing Labs"
            )

            send_mail(
                user_subject,
                user_message,
                settings.DEFAULT_FROM_EMAIL,
                [email],
                fail_silently=False,
            )

            messages.success(request, 'Your registration has been submitted successfully! A confirmation email has been sent to you.')
        except Exception as e:
            logger.exception('Error sending enrollment emails')
            messages.warning(request, 'Your registration was received but we could not send the confirmation email.')

        return redirect('courses:detail', slug=slug)
    
    return redirect('courses:detail', slug=slug)