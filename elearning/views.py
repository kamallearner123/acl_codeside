from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_POST
from django.conf import settings
from courses.models import Course
from .models import StudentProfile, CourseProgress
import json
import os


def landing(request):
    """Public landing page with student login form embedded."""
    if request.user.is_authenticated and hasattr(request.user, 'student_profile'):
        return redirect('elearning:dashboard')
    courses = Course.objects.all()
    return render(request, 'elearning/landing.html', {'courses': courses})


def student_login(request):
    """Handle POST login from the landing page form."""
    if request.method == 'POST':
        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '').strip()

        # Resolve username from email
        from django.contrib.auth.models import User
        try:
            user_obj = User.objects.get(email__iexact=email)
            username = user_obj.username
        except User.DoesNotExist:
            messages.error(request, 'No account found with that email address.')
            return redirect('elearning:landing')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            # Ensure student profile exists
            StudentProfile.objects.get_or_create(user=user)
            login(request, user)
            return redirect('elearning:dashboard')
        else:
            messages.error(request, 'Invalid credentials. Please try again.')
            return redirect('elearning:landing')
    return redirect('elearning:landing')


@login_required(login_url='/elearning/')
def student_logout(request):
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('elearning:landing')


@login_required(login_url='/elearning/')
def dashboard(request):
    """Student dashboard — enrolled courses, progress, all available courses."""
    profile, _ = StudentProfile.objects.get_or_create(user=request.user)
    enrolled = profile.enrolled_courses.all()

    # Build progress map
    progress_qs = CourseProgress.objects.filter(student=profile).select_related('course')
    progress_map = {cp.course_id: cp for cp in progress_qs}

    enrolled_with_progress = []
    for course in enrolled:
        cp = progress_map.get(course.id)
        enrolled_with_progress.append({
            'course': course,
            'progress': cp.progress_pct if cp else 0,
            'status': cp.status if cp else 'not_started',
        })

    all_courses = Course.objects.all()
    available_courses = [c for c in all_courses if c not in enrolled]

    context = {
        'profile': profile,
        'enrolled_with_progress': enrolled_with_progress,
        'available_courses': available_courses,
        'enrolled_count': enrolled.count(),
        'completed_count': sum(1 for e in enrolled_with_progress if e['status'] == 'completed'),
        'in_progress_count': sum(1 for e in enrolled_with_progress if e['status'] == 'in_progress'),
    }
    return render(request, 'elearning/dashboard.html', context)


@login_required(login_url='/elearning/')
def enroll_course(request, course_id):
    """Enroll current student into a course."""
    course = get_object_or_404(Course, pk=course_id)
    profile, _ = StudentProfile.objects.get_or_create(user=request.user)
    profile.enrolled_courses.add(course)
    CourseProgress.objects.get_or_create(student=profile, course=course)
    messages.success(request, f'Successfully enrolled in "{course.title}"!')
    return redirect('elearning:dashboard')


@login_required(login_url='/elearning/')
def course_detail(request, course_id):
    """Course detail page — reads Header.json for links and calendar sessions."""
    course = get_object_or_404(Course, pk=course_id)
    profile, _ = StudentProfile.objects.get_or_create(user=request.user)

    folder_name = course.title.replace(' ', '')
    json_path = os.path.join(settings.BASE_DIR, 'eLearningCourse', folder_name, 'Header.json')

    header = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                header = json.load(f)
            except json.JSONDecodeError:
                header = {}

    # Load curriculum
    curriculum_path = os.path.join(settings.BASE_DIR, 'eLearningCourse', folder_name, 'Curriculum.json')
    curriculum = []
    if os.path.exists(curriculum_path):
        with open(curriculum_path, 'r') as f:
            try:
                curriculum = json.load(f)
            except json.JSONDecodeError:
                curriculum = []

    total_hours = sum(
        float(m['duration'].replace(' hrs', '').replace(' hr', ''))
        for m in curriculum if 'duration' in m
    )

    # Convert DD/MM/YYYY → YYYY-MM-DD for each calendar entry
    raw_sessions = header.get('Calender', [])
    calendar_sessions = []
    for s in raw_sessions:
        try:
            day, month, year = s['Date'].split('/')
            iso_date = f"{year}-{month}-{day}"
        except Exception:
            iso_date = ''
        calendar_sessions.append({
            'iso_date': iso_date,
            'zoom': s.get('Zoom', ''),
            'assignment': s.get('Assignment', ''),
            'mcq_link': s.get('MCQ Link', ''),
            'recording': s.get('Video Recording', ''),
        })

    cp = CourseProgress.objects.filter(student=profile, course=course).first()
    is_enrolled = profile.enrolled_courses.filter(pk=course_id).exists()
    topic_progress = cp.topic_progress if (cp and cp.topic_progress) else {}

    context = {
        'course': course,
        'header': header,
        'github_link': header.get('GithubLink', ''),
        'gdrive_link': header.get('Gdrive', ''),
        'zoom_link': header.get('ZoomLink', ''),
        'mcqs_link': header.get('MCQs', ''),
        'calendar_sessions': calendar_sessions,
        'calendar_sessions_json': json.dumps(calendar_sessions),
        'curriculum': curriculum,
        'total_hours': total_hours,
        'progress': cp.progress_pct if cp else 0,
        'status': cp.status if cp else 'not_started',
        'is_enrolled': is_enrolled,
        'topic_progress': topic_progress,
        'topic_progress_json': json.dumps(topic_progress),
    }
    return render(request, 'elearning/course_detail.html', context)


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@login_required(login_url='/elearning/')
@csrf_exempt
def update_topic_progress(request, course_id):
    """AJAX endpoint to update checkbox, rating, and assignment complete for a topic."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            topic_key = data.get('topic_key')  # e.g., "01_2" (module 01, topic index 2)
            completed = data.get('completed', False)
            rating = data.get('rating', 0)
            assignment = data.get('assignment', False)

            course = get_object_or_404(Course, pk=course_id)
            profile, _ = StudentProfile.objects.get_or_create(user=request.user)
            cp, _ = CourseProgress.objects.get_or_create(student=profile, course=course)

            if not cp.topic_progress:
                cp.topic_progress = {}

            cp.topic_progress[topic_key] = {
                'completed': completed,
                'rating': int(rating),
                'assignment': assignment
            }

            # Recalculate course completion pct
            folder_name = course.title.replace(' ', '')
            curriculum_path = os.path.join(settings.BASE_DIR, 'eLearningCourse', folder_name, 'Curriculum.json')
            total_topics = 0
            if os.path.exists(curriculum_path):
                with open(curriculum_path, 'r') as f:
                    try:
                        curr = json.load(f)
                        for mod in curr:
                            total_topics += len(mod.get('topics', []))
                    except Exception:
                        pass

            if total_topics > 0:
                completed_count = sum(1 for k, v in cp.topic_progress.items() if v.get('completed'))
                cp.progress_pct = int((completed_count / total_topics) * 100)
                if cp.progress_pct >= 100:
                    cp.status = 'completed'
                elif cp.progress_pct > 0:
                    cp.status = 'in_progress'
                else:
                    cp.status = 'not_started'

            cp.save()
            return JsonResponse({
                'status': 'success',
                'progress_pct': cp.progress_pct,
                'course_status': cp.status
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)


@login_required(login_url='/elearning/')
def student_profile(request):
    """Student profile view — edit profile fields and change password."""
    profile, _ = StudentProfile.objects.get_or_create(user=request.user)
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'update_profile':
            first_name = request.POST.get('first_name', '').strip()
            last_name = request.POST.get('last_name', '').strip()
            email = request.POST.get('email', '').strip().lower()
            bio = request.POST.get('bio', '').strip()
            
            if not email:
                messages.error(request, 'Email address is required.')
            else:
                from django.contrib.auth.models import User
                if User.objects.filter(email__iexact=email).exclude(pk=request.user.pk).exists():
                    messages.error(request, 'This email address is already in use by another account.')
                else:
                    request.user.first_name = first_name
                    request.user.last_name = last_name
                    request.user.email = email
                    request.user.save()
                    
                    profile.bio = bio
                    
                    # Avatar picture upload
                    avatar_file = request.FILES.get('avatar')
                    if avatar_file:
                        from django.core.files.storage import FileSystemStorage
                        import uuid
                        ext = os.path.splitext(avatar_file.name)[1].lower()
                        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                            filename = f"avatar_{request.user.pk}_{uuid.uuid4().hex[:8]}{ext}"
                            avatar_dir = os.path.join(settings.MEDIA_ROOT, 'avatars')
                            if not os.path.exists(avatar_dir):
                                os.makedirs(avatar_dir)
                            fs = FileSystemStorage(location=avatar_dir, base_url=settings.MEDIA_URL + 'avatars/')
                            saved_name = fs.save(filename, avatar_file)
                            profile.avatar_url = settings.MEDIA_URL + 'avatars/' + saved_name
                        else:
                            messages.error(request, 'Invalid image format. Allowed formats: JPG, PNG, GIF, WEBP.')
                    
                    profile.save()
                    messages.success(request, 'Your profile has been updated successfully!')
            return redirect('elearning:profile')
            
        elif action == 'change_password':
            old_password = request.POST.get('old_password', '').strip()
            new_password = request.POST.get('new_password', '').strip()
            confirm_password = request.POST.get('confirm_password', '').strip()
            
            if not request.user.check_password(old_password):
                messages.error(request, 'Incorrect current password.')
            elif not new_password or len(new_password) < 6:
                messages.error(request, 'New password must be at least 6 characters long.')
            elif new_password != confirm_password:
                messages.error(request, 'New passwords do not match.')
            else:
                request.user.set_password(new_password)
                request.user.save()
                update_session_auth_hash(request, request.user)
                messages.success(request, 'Your password has been changed successfully!')
            return redirect('elearning:profile')
            
    context = {
        'profile': profile,
    }
    return render(request, 'elearning/profile.html', context)
