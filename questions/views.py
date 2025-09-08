from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView
from django.contrib import messages
from django.http import JsonResponse
from .models import Question
from .forms import SubmissionForm
from submissions.models import Submission
import json
import sys
import io
import contextlib
from django.db.models import Count, Q, Min
from django.http import HttpResponseForbidden


class QuestionListView(LoginRequiredMixin, ListView):
    model = Question
    template_name = 'questions/list.html'
    context_object_name = 'questions'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Question.objects.filter(is_active=True)
        difficulty = self.request.GET.get('difficulty')
        if difficulty and difficulty in ['easy', 'medium', 'hard']:
            queryset = queryset.filter(difficulty=difficulty)
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['current_difficulty'] = self.request.GET.get('difficulty', '')
        return context


class QuestionDetailView(LoginRequiredMixin, DetailView):
    model = Question
    template_name = 'questions/detail.html'
    context_object_name = 'question'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = SubmissionForm()
        context['user_submissions'] = Submission.objects.filter(
            user=self.request.user, 
            question=self.object
        ).order_by('-submitted_at')[:5]
        return context


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
                user_profile.problems_solved += 1
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
