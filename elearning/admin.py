from django.contrib import admin
from .models import StudentProfile, CourseProgress


@admin.register(StudentProfile)
class StudentProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'enrolled_count', 'created_at')
    search_fields = ('user__username', 'user__email', 'user__first_name')
    filter_horizontal = ('enrolled_courses',)

    @admin.display(description='Enrolled Courses')
    def enrolled_count(self, obj):
        return obj.enrolled_courses.count()


@admin.register(CourseProgress)
class CourseProgressAdmin(admin.ModelAdmin):
    list_display = ('student', 'course', 'status', 'progress_pct', 'last_accessed')
    list_filter = ('status', 'course')
    search_fields = ('student__user__username', 'course__title')
