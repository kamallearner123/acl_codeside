from django.urls import path
from . import views

app_name = 'courses'

urlpatterns = [
    path('', views.CourseListView.as_view(), name='list'),
    path('stm32-c-rust-automotive/', views.stm32_automotive_group, name='stm32_automotive_group'),
    path('stm32-c-rust-automotive/feedback/', views.stm32_group_feedback, name='stm32_group_feedback'),
    path('stm32-c-rust-automotive/feedback/verify/', views.stm32_group_feedback_verify, name='stm32_group_feedback_verify'),
    path('agentic-ai/', views.removed_course, name='removed_agentic_ai'),
    path('exploring-zephyr-using-stm32/book/', views.zephyr_book_index, name='zephyr_book_index'),
    path('exploring-zephyr-using-stm32/book/<path:subpath>', views.zephyr_book_serve, name='zephyr_book_serve'),
    path('stm32-firmware-development-with-c/book/', views.stm32_c_book_index, name='stm32_c_book_index'),
    path('stm32-firmware-development-with-c/book/<path:subpath>', views.stm32_c_book_serve, name='stm32_c_book_serve'),
    path('embedded-rust-with-stm32/book/', views.stm32_rust_book_index, name='stm32_rust_book_index'),
    path('embedded-rust-with-stm32/book/<path:subpath>', views.stm32_rust_book_serve, name='stm32_rust_book_serve'),
    path('arm-cortex-m-architecture/book/', views.cortexm_book_index, name='cortexm_book_index'),
    path('arm-cortex-m-architecture/book/<path:subpath>', views.cortexm_book_serve, name='cortexm_book_serve'),
    path('visualise-stm32/book/', views.visualise_stm32_book_index, name='visualise_stm32_book_index'),
    path('visualise-stm32/book/<path:subpath>', views.visualise_stm32_book_serve, name='visualise_stm32_book_serve'),
    path('<slug:slug>/', views.CourseDetailView.as_view(), name='detail'),
    path('<slug:slug>/enroll/', views.enroll_course, name='enroll'),
    path('<slug:slug>/enroll/verify/', views.enroll_verify, name='enroll_verify'),
]
