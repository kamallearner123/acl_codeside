from django.contrib import admin
from .models import CodeExecution


@admin.register(CodeExecution)
class CodeExecutionAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_at', 'status')
    list_filter = ('status', 'created_at')
    search_fields = ('code',)
    readonly_fields = ('created_at',)
