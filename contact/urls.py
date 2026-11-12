from django.urls import path
from . import views
from .views import contact

app_name = 'contact'

urlpatterns = [
    path("", contact, name='contact'),
    path("verify/", views.verify, name="verify"),
    path("contact/", contact),
    path("contact/verify/", views.verify),
]
