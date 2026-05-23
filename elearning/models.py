from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from courses.models import Course


class StudentProfile(models.Model):
    ROLE_STUDENT = 'student'
    ROLE_CHOICES = [(ROLE_STUDENT, 'Student')]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='student_profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_STUDENT)
    enrolled_courses = models.ManyToManyField(Course, blank=True, related_name='enrolled_students')
    bio = models.TextField(blank=True)
    avatar_url = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Student Profile'
        verbose_name_plural = 'Student Profiles'

    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username} (Student)"

    @property
    def full_name(self):
        return self.user.get_full_name() or self.user.username

    @property
    def enrolled_count(self):
        return self.enrolled_courses.count()


class CourseProgress(models.Model):
    STATUS_CHOICES = [
        ('not_started', 'Not Started'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
    ]

    student = models.ForeignKey(StudentProfile, on_delete=models.CASCADE, related_name='progress')
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='not_started')
    progress_pct = models.PositiveSmallIntegerField(default=0)  # 0-100
    topic_progress = models.JSONField(default=dict, blank=True)  # { "mod_id_topic_index": { "completed": true, "rating": 4, "assignment": true } }
    last_accessed = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('student', 'course')
        verbose_name = 'Course Progress'

    def __str__(self):
        return f"{self.student.user.username} – {self.course.title} ({self.progress_pct}%)"
