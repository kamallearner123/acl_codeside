from django.urls import path
from . import views

app_name = 'courses'

urlpatterns = [
    path('', views.CourseListView.as_view(), name='list'),
    path('c-system-programming/', views.CSystemProgrammingView.as_view(), name='c_system_programming'),
    path('python-programming/', views.PythonProgrammingView.as_view(), name='python_programming'),
    path('rust-programming/', views.RustProgrammingView.as_view(), name='rust_programming'),
    path('linux-os-concepts/', views.LinuxOSConceptsView.as_view(), name='linux_os_concepts'),
    path('iot-devices/', views.IoTDevicesView.as_view(), name='iot_devices'),
    path('networking-security/', views.NetworkingSecurityView.as_view(), name='networking_security'),
]
