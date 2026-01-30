import sys
# Touched to trigger reload
import io
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .curriculum import CURRICULUM

import markdown

def index(request):
    """
    Renders the main Python Tutor page with the full curriculum content.
    The content is processed to convert Markdown to HTML.
    """
    # Clone the curriculum to avoid modifying the original globally
    processed_curriculum = []
    
    for section in CURRICULUM:
        new_section = section.copy()
        new_topics = []
        for topic in section.get('topics', []):
            new_topic = topic.copy()
            # Convert markdown content to HTML
            # Using 'fenced_code' extension for code blocks if needed, though we handle code separately
            if 'content' in new_topic:
                new_topic['content'] = markdown.markdown(
                    new_topic['content'],
                    extensions=['fenced_code', 'nl2br']
                )
            new_topics.append(new_topic)
        new_section['topics'] = new_topics
        processed_curriculum.append(new_section)

    context = {
        'curriculum': processed_curriculum
    }
    return render(request, 'tutor/index.html', context)

@csrf_exempt
def run_code(request):
    """
    Executes Python code sent from the frontend and returns the output.
    WARNING: precise 'exec' is used here for educational purposes in a controlled environment.
    In a production public-facing app, this would require a secure sandbox (e.g., Docker/Firejail).
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            code = data.get('code', '')
            
            # Create a buffer to capture stdout
            output_buffer = io.StringIO()
            
            # Save original stdout
            original_stdout = sys.stdout
            sys.stdout = output_buffer
            
            # Create a safe-ish dictionary for execution context
            # We allow basic builtins but restrict access to sensitive modules if possible
            # For this tutor, we want them to be able to import things, so we won't restrict too much
            # but we generally clean the scope.
            exec_globals = {}
            
            try:
                exec(code, exec_globals)
                output = output_buffer.getvalue()
                
                # If no output was printed but code ran, maybe show a success message or the last expression
                if not output:
                    output = "[Code executed successfully with no output]"
                    
            except Exception as e:
                # Capture runtime errors
                output = f"Error: {str(e)}"
            finally:
                # Restore stdout
                sys.stdout = original_stdout
                
            return JsonResponse({'output': output, 'status': 'success'})
            
        except json.JSONDecodeError:
            return JsonResponse({'output': 'Invalid JSON', 'status': 'error'}, status=400)
        except Exception as e:
            return JsonResponse({'output': f"Server Error: {str(e)}", 'status': 'error'}, status=500)
            
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)
