from django.contrib import admin
from .models import Question, Tag


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ['title', 'difficulty', 'is_active', 'total_submissions', 'success_rate', 'created_at']
    list_filter = ['difficulty', 'is_active', 'created_at']
    search_fields = ['title', 'description']
    prepopulated_fields = {'slug': ('title',)}
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'slug', 'description', 'difficulty', 'is_active')
        }),
        ('Examples and Templates', {
            'fields': ('example_input', 'example_output', 'template_code')
        }),
        ('Additional Information', {
            'fields': ('constraints', 'hints', 'test_cases'),
            'classes': ('collapse',)
        }),
        ('Statistics', {
            'fields': ('total_submissions', 'successful_submissions'),
            'classes': ('collapse',)
        })
    )


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ['name']
    search_fields = ['name']
