from django.urls import path
from . import views
from .views import contact

app_name = 'contact'

urlpatterns = [
    path("contact/",contact, name='contact'),
    path("contact/verify/", views.verify, name="verify"),
]
