from django.contrib import admin
from .models import Submission


@admin.register(Submission)
class SubmissionAdmin(admin.ModelAdmin):
    list_display = ['user', 'question', 'status', 'test_cases_passed', 'total_test_cases', 'runtime', 'submitted_at']
    list_filter = ['status', 'language', 'submitted_at']
    search_fields = ['user__username', 'question__title']
    readonly_fields = ['submitted_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'question', 'status', 'language')
        }),
        ('Code', {
            'fields': ('code',)
        }),
        ('Execution Results', {
            'fields': ('test_cases_passed', 'total_test_cases', 'runtime', 'memory_usage', 'error_message')
        }),
        ('Metadata', {
            'fields': ('submitted_at',),
            'classes': ('collapse',)
        })
    )
