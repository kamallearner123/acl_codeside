from django.contrib import admin
from .models import Course, Event, GroupFeedback


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ('title', 'duration_hours')
    prepopulated_fields = {'slug': ('title',)}


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ('title', 'course', 'date')
    list_filter = ('course', 'date')


@admin.register(GroupFeedback)
class GroupFeedbackAdmin(admin.ModelAdmin):
    list_display = ('topic', 'name', 'email', 'course', 'created_at')
    list_filter = ('topic', 'created_at')
    search_fields = ('name', 'email', 'course', 'message')
    readonly_fields = ('created_at',)
