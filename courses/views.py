from django.views.generic import ListView, TemplateView
from .models import Course


class CourseListView(ListView):
    model = Course
    template_name = 'courses/list.html'
    context_object_name = 'courses'


class CSystemProgrammingView(TemplateView):
    template_name = 'courses/c_system_programming.html'


class PythonProgrammingView(TemplateView):
    template_name = 'courses/python_programming.html'


class RustProgrammingView(TemplateView):
    template_name = 'courses/rust_programming.html'


class LinuxOSConceptsView(TemplateView):
    template_name = 'courses/linux_os_concepts.html'


class IoTDevicesView(TemplateView):
    template_name = 'courses/iot_devices.html'


class NetworkingSecurityView(TemplateView):
    template_name = 'courses/networking_security.html'