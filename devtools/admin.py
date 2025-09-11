from django.contrib import admin
from .models import DevTool


@admin.register(DevTool)
class DevToolAdmin(admin.ModelAdmin):
    list_display = ('name', 'link')
