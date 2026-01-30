from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView
from django.contrib import messages
from django.http import JsonResponse, HttpResponseForbidden
from .models import Question, Comment
from .forms import SubmissionForm, CommentForm
from submissions.models import Submission
import json
import sys
import io
import contextlib
from django.db.models import Count, Q, Min


class QuestionListView(LoginRequiredMixin, ListView):
    model = Question
    template_name = 'questions/list.html'
    context_object_name = 'questions'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Question.objects.filter(is_active=True).prefetch_related('tags')
        
        # Search filter
        search_query = self.request.GET.get('search')
        if search_query:
            queryset = queryset.filter(
                Q(title__icontains=search_query) | 
                Q(description__icontains=search_query)
            )
            
        # Difficulty filter
        difficulty = self.request.GET.get('difficulty')
        if difficulty and difficulty in ['easy', 'medium', 'hard']:
            queryset = queryset.filter(difficulty=difficulty)
            
        # Tag filter
        tag_slug = self.request.GET.get('tag')
        if tag_slug:
            queryset = queryset.filter(tags__slug=tag_slug)
            
        # Company filter
        company_slug = self.request.GET.get('company')
        if company_slug:
            queryset = queryset.filter(companies__slug=company_slug)
            
        # Status filter (Solved/Unsolved)
        status = self.request.GET.get('status')
        if status:
            solved_questions = Submission.objects.filter(
                user=self.request.user, 
                status='accepted'
            ).values_list('question_id', flat=True)
            
            if status == 'solved':
                queryset = queryset.filter(id__in=solved_questions)
            elif status == 'unsolved':
                queryset = queryset.exclude(id__in=solved_questions)
                
        return queryset.distinct()
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['current_difficulty'] = self.request.GET.get('difficulty', '')
        context['current_search'] = self.request.GET.get('search', '')
        context['current_tag'] = self.request.GET.get('tag', '')
        context['current_company'] = self.request.GET.get('company', '')
        context['current_status'] = self.request.GET.get('status', '')
        from .models import Tag, Company
        context['all_tags'] = Tag.objects.all()
        context['all_companies'] = Company.objects.all()
        
        if self.request.user.is_authenticated:
            context['user_solved_ids'] = Submission.objects.filter(
                user=self.request.user, 
                status='accepted'
            ).values_list('question_id', flat=True)
            
        return context


class QuestionDetailView(LoginRequiredMixin, DetailView):
    model = Question
    template_name = 'questions/detail.html'
    context_object_name = 'question'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SubmissionForm()
        context['comment_form'] = CommentForm()
        context['comments'] = self.object.comments.filter(parent=None).prefetch_related('replies')
        context['user_submissions'] = Submission.objects.filter(
            user=self.request.user, 
            question=self.object
        ).order_by('-submitted_at')[:5]
        return context


@login_required
def post_comment(request, slug):
    question = get_object_or_404(Question, slug=slug)
    if request.method == 'POST':
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.user = request.user
            comment.question = question
            parent_id = request.POST.get('parent_id')
            if parent_id:
                comment.parent = get_object_or_404(Comment, id=parent_id)
            comment.save()
            messages.success(request, 'Your comment has been posted.')
    return redirect('questions:detail', slug=slug)


@login_required
def submit_solution(request, slug):
    question = get_object_or_404(Question, slug=slug)
    
    if request.method == 'POST':
        form = SubmissionForm(request.POST)
        if form.is_valid():
            submission = form.save(commit=False)
            submission.user = request.user
            submission.question = question
            
            # Execute the code and check against test cases
            result = execute_code(submission.code, question.test_cases)
            
            submission.status = result['status']
            submission.test_cases_passed = result['passed']
            submission.total_test_cases = result['total']
            submission.error_message = result.get('error', '')
            submission.runtime = result.get('runtime', 0)
            
            submission.save()
            
            # Update question statistics
            question.total_submissions += 1
            if submission.status == 'accepted':
                question.successful_submissions += 1
            question.save()
            
            # Update user profile statistics
            user_profile = request.user.profile
            user_profile.total_submissions += 1
            if submission.status == 'accepted':
                # Check if this problem was already solved by the user to avoid double points
                already_solved = Submission.objects.filter(
                    user=request.user,
                    question=question,
                    status='accepted'
                ).exclude(id=submission.id).exists()
                
                if not already_solved:
                    user_profile.problems_solved += 1
                    # Award points based on difficulty
                    points_map = {'easy': 10, 'medium': 30, 'hard': 50}
                    user_profile.points += points_map.get(question.difficulty, 10)
                    
                    # Simple level calculation: 100 points per level
                    user_profile.level = (user_profile.points // 100) + 1
            user_profile.save()
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': submission.status,
                    'passed': submission.test_cases_passed,
                    'total': submission.total_test_cases,
                    'error': submission.error_message,
                    'runtime': submission.runtime
                })
            
            messages.success(request, f'Solution submitted! Status: {submission.get_status_display()}')
            return redirect('questions:detail', slug=slug)
    
    return redirect('questions:detail', slug=slug)


def execute_code(code, test_cases):
    """
    Execute user code against test cases
    This is a simplified version - in production you'd use a sandboxed environment
    """
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # Create a restricted globals environment
        safe_globals = {
            '__builtins__': {
                'len': len,
                'range': range,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'min': min,
                'max': max,
                'sum': sum,
                'sorted': sorted,
                'reversed': reversed,
                'enumerate': enumerate,
                'zip': zip,
            }
        }
        
        # Execute the user code
        exec(code, safe_globals)
        
        passed = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            try:
                # Extract function name from code (simplified)
                func_name = 'solution'  # Default function name
                if 'def ' in code:
                    lines = code.split('\n')
                    for line in lines:
                        if line.strip().startswith('def '):
                            func_name = line.split('def ')[1].split('(')[0].strip()
                            break
                
                # Get the function from globals
                if func_name in safe_globals:
                    func = safe_globals[func_name]
                    
                    # Parse input and execute
                    test_input = test_case.get('input', '')
                    expected_output = test_case.get('expected', '')
                    
                    # This is simplified - you'd need proper input parsing
                    if test_input:
                        result = func(eval(test_input))
                    else:
                        result = func()
                    
                    if str(result) == str(expected_output):
                        passed += 1
                
            except Exception:
                continue
        
        sys.stdout = old_stdout
        
        status = 'accepted' if passed == total else 'wrong_answer'
        
        return {
            'status': status,
            'passed': passed,
            'total': total,
            'runtime': 0.1  # Mock runtime
        }
        
    except SyntaxError as e:
        sys.stdout = old_stdout
        return {
            'status': 'compilation_error',
            'passed': 0,
            'total': len(test_cases),
            'error': str(e)
        }
    except Exception as e:
        sys.stdout = old_stdout
        return {
            'status': 'runtime_error',
            'passed': 0,
            'total': len(test_cases),
            'error': str(e)
        }


@login_required
def solved_stats(request):
    """Superuser-only view: list questions with number of distinct users who solved them."""
    if not request.user.is_superuser:
        return HttpResponseForbidden("You do not have permission to view this page.")

    stats = Question.objects.annotate(
        solvers=Count('submissions__user', filter=Q(submissions__status='accepted'), distinct=True)
    ).order_by('-solvers', 'title')

    return render(request, 'questions/stats.html', {'stats': stats})


@login_required
def question_solvers(request, slug):
    """Superuser-only view: list users who solved a given question (first accepted time)."""
    if not request.user.is_superuser:
        return HttpResponseForbidden("You do not have permission to view this page.")

    question = get_object_or_404(Question, slug=slug)

    solvers = (
        Submission.objects
        .filter(question=question, status='accepted')
        .values('user__id', 'user__username')
        .annotate(first_accepted=Min('submitted_at'))
        .order_by('first_accepted')
    )

    return render(request, 'questions/solvers.html', {'question': question, 'solvers': solvers})
