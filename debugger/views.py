from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .models import CodeExecution
from .services.rust_executor import RustExecutor
from .services.miri_parser import MiriParser
from .services.python_executor import PythonExecutor
from .services.cpp_executor import CPPExecutor


def get_dsa_problems():
    """Return a curated list of DSA problems with basic metadata."""
    problems = [
        {"title": "Reverse a linked list (iterative & recursive)",
         "description": "Reverse singly linked list using iterative and recursive approaches.",
         "snippet": "# see codes.python for implementation",
         "example": "Use when you need to reverse order of a playlist or undo a sequence of operations."},

        {"title": "Maximum sum subarray (Kadane's)",
         "description": "Linear time algorithm to find max contiguous subarray sum.",
         "snippet": "# see codes.python for implementation",
         "example": "Detect best performing time window (sales, metrics) over a timeline."},

        {"title": "Binary search on a sorted array",
         "description": "Classic O(log n) search on sorted arrays.",
         "snippet": "# see codes.python for implementation",
         "example": "Find timestamps or ids quickly in sorted logs or indices."},

        {"title": "Fibonacci generator",
         "description": "Yield Fibonacci sequence lazily using generators.",
         "snippet": "# see codes.python for implementation",
         "example": "Stream large Fibonacci sequences without storing all values in memory."},

        {"title": "Check balanced parentheses",
         "description": "Use a stack to validate matching pairs for (), {}, [].",
         "snippet": "# see codes.python for implementation",
         "example": "Validate user-submitted expressions or check code formatting in editors."},
    ]

    # Attach generated language snippets
    for p in problems:
        if 'codes' not in p:
            p['codes'] = generate_language_snippets(p)

    return problems


def generate_language_snippets(problem):
    """Generate simple python/cpp/rust snippets for a problem title."""
    title = problem.get('title','').lower()
    codes = {'python': '', 'cpp': '', 'rust': ''}
    if 'reverse a linked list' in title:
        codes['python'] = '''class Node:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

def reverse(head):
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev
'''
        codes['cpp'] = '''#include <bits/stdc++.h>
using namespace std;
struct Node{int val; Node* next;};
Node* reverse(Node* head){ Node* prev=nullptr; Node* cur=head; while(cur){ Node* nxt=cur->next; cur->next=prev; prev=cur; cur=nxt; } return prev; }
'''
        codes['rust'] = '''// Placeholder: implement linked list reverse using Option<Box<Node>> in Rust
fn main(){ println!("See Python/C++ implementations and port to Rust."); }
'''

    elif 'kadane' in title or 'maximum sum subarray' in title:
        codes['python'] = '''def kadane(a):
    max_ending = 0
    max_so_far = float('-inf')
    for x in a:
        max_ending = max(x, max_ending + x)
        max_so_far = max(max_so_far, max_ending)
    return max_so_far

print(kadane([1,-2,3,4,-1,2]))
'''
        codes['cpp'] = '''#include <bits/stdc++.h>
using namespace std;
int kadane(vector<int>& a){ int max_ending=0, max_sofar=INT_MIN; for(int x: a){ max_ending = max(x, max_ending + x); max_sofar = max(max_sofar, max_ending);} return max_sofar; }
int main(){ vector<int> a={1,-2,3,4,-1,2}; cout<<kadane(a)<<"\n"; }
'''
        codes['rust'] = '''fn kadane(a: &[i64]) -> i64{ let mut max_ending=0i64; let mut max_sofar=i64::MIN; for &x in a{ max_ending = std::cmp::max(x, max_ending + x); max_sofar = std::cmp::max(max_sofar, max_ending);} max_sofar }
fn main(){ let a=[1, -2, 3, 4, -1, 2]; println!("{}", kadane(&a)); }
'''

    elif 'binary search' in title:
        codes['python'] = '''def binary_search(a, x):
    l, r = 0, len(a)-1
    while l<=r:
        m=(l+r)//2
        if a[m]==x:
            return m
        if a[m]<x:
            l=m+1
        else:
            r=m-1
    return -1

print(binary_search([1,2,3,4,5], 4))
'''
        codes['cpp'] = '''#include <bits/stdc++.h>
using namespace std; int bs(vector<int>& a,int x){ int l=0,r=a.size()-1; while(l<=r){ int m=(l+r)/2; if(a[m]==x) return m; if(a[m]<x) l=m+1; else r=m-1;} return -1;} int main(){ vector<int> a={1,2,3,4,5}; cout<<bs(a,4)<<"\n"; }
'''
        codes['rust'] = '''fn binary_search(a: &[i32], x:i32)->i32{ let mut l=0i32; let mut r=(a.len() as i32)-1; while l<=r{ let m=((l+r)/2) as usize; if a[m]==x { return m as i32;} if a[m]<x { l=(m as i32)+1;} else { r=(m as i32)-1;} } -1 }
fn main(){ let a=[1,2,3,4,5]; println!("{}", binary_search(&a,4)); }
'''

    elif 'fibonacci' in title or 'fib' in title:
        codes['python'] = '''def fib():
    a,b=0,1
    while True:
        yield a
        a,b=b,a+b

g= fib()
for _ in range(10): print(next(g))
'''
        codes['cpp'] = '''#include <bits/stdc++.h>
using namespace std;int main(){ long long a=0,b=1; for(int i=0;i<10;i++){ cout<<a<<"\n"; long long t=b; b=a+b; a=t;} }
'''
        codes['rust'] = '''fn main(){ let mut a=0i128; let mut b=1i128; for _ in 0..10{ println!("{}", a); let t=b; b=a+b; a=t;} }
'''

    else:
        snippet = problem.get('snippet','')
        codes['python'] = f"# Placeholder Python snippet for: {problem.get('title')}\nprint(\"{problem.get('example','')[:200]}\")\n"
        codes['cpp'] = f"// Placeholder C++ snippet for: {problem.get('title')}\n#include <iostream>\nint main(){{ std::cout<<\"See description: {problem.get('example','')[:200]}\"<<std::endl; return 0; }}\n"
        codes['rust'] = f"// Placeholder Rust snippet for: {problem.get('title')}\nfn main(){{ println!(\"See description: {problem.get('example','')[:200]}\"); }}\n"

    return codes


def dsa_practice(request):
    """Render the DSA practice landing page"""
    problems = get_dsa_problems()
    return render(request, 'debugger/dsa_practice.html', {'problems': problems})


def dsa_problem(request, idx: int):
    """Render a single DSA problem page by index."""
    problems = get_dsa_problems()
    try:
        problem = problems[idx]
    except Exception:
        # Index out of range -> 404
        from django.http import Http404

        raise Http404("Problem not found")

    # Compute prev/next indices for navigation
    prev_idx = idx - 1 if idx - 1 >= 0 else None
    next_idx = idx + 1 if idx + 1 < len(problems) else None

    return render(
        request,
        'debugger/dsa_problem.html',
        {
            'problem': problem,
            'index': idx,
            'prev_idx': prev_idx,
            'next_idx': next_idx,
        },
    )


@csrf_exempt
@require_http_methods(["POST"])
def execute_cpp(request):
    """Compile and execute C++ code and return output"""
    import logging
    logger = logging.getLogger(__name__)
    try:
        data = json.loads(request.body)
        code = data.get('code', '')
        stdin_data = data.get('stdin', None)

        logger.info(f'🧾 Received C++ code length: {len(code)}')

        if not code:
            return JsonResponse({'error': 'No code provided'}, status=400)

        executor = CPPExecutor(timeout=10)
        result = executor.execute(code, stdin_data=stdin_data)

        return JsonResponse({
            'success': result.get('success', False),
            'stdout': result.get('stdout', ''),
            'stderr': result.get('stderr', ''),
            'execution_time': result.get('execution_time', 0.0)
        })

    except Exception as e:
        logger.error(f'❌ Error executing C++ code: {str(e)}')
        return JsonResponse({'error': str(e)}, status=500)


def index(request):
    """Home page with language selection"""
    return render(request, 'debugger/home.html')


def rust_editor(request):
    """Rust debugger interface"""
    return render(request, 'debugger/rust_editor.html')


def python_editor(request):
    """Python programming interface"""
    return render(request, 'debugger/python_editor.html')


@csrf_exempt
@require_http_methods(["POST"])
def execute_code(request):
    """Execute Rust code with Miri and return memory trace"""
    import logging
    logger = logging.getLogger(__name__)
    
    # Log the incoming HTTP request
    logger.info('=' * 80)
    logger.info('📥 [HTTP REQUEST RECEIVED - RUST]')
    logger.info(f'   Method: {request.method}')
    logger.info(f'   Path: {request.path}')
    logger.info(f'   Content-Type: {request.content_type}')
    logger.info(f'   Remote Address: {request.META.get("REMOTE_ADDR")}')
    logger.info('=' * 80)
    
    try:
        data = json.loads(request.body)
        code = data.get('code', '')
        use_miri = data.get('use_miri', True)  # Default to Miri execution
        
        logger.info(f'🦀 Rust code length: {len(code)} characters')
        logger.info(f'🔧 Use Miri: {use_miri}')
        
        if not code:
            return JsonResponse({'error': 'No code provided'}, status=400)
        
        # Create execution record
        execution = CodeExecution.objects.create(code=code, status='running')
        
        # Execute code
        executor = RustExecutor()
        
        if use_miri:
            # Execute with Miri for memory tracing
            result = executor.execute_with_miri(code)
            
            # Parse Miri output
            parser = MiriParser()
            memory_trace = parser.parse(result.get('miri_output', ''))
            
            # Update execution record
            execution.execution_output = result.get('stdout', '')
            execution.miri_trace = memory_trace
            execution.status = 'completed' if result.get('success') else 'failed'
            execution.save()
            
            return JsonResponse({
                'execution_id': execution.id,
                'success': result.get('success', False),
                'stdout': result.get('stdout', ''),
                'stderr': result.get('stderr', ''),
                'memory_trace': memory_trace,
                'mode': 'miri'
            })
        else:
            # Execute normally without Miri
            result = executor.execute_normal(code)
            
            # Update execution record
            execution.execution_output = result.get('stdout', '')
            execution.status = 'completed' if result.get('success') else 'failed'
            execution.save()
            
            return JsonResponse({
                'execution_id': execution.id,
                'success': result.get('success', False),
                'stdout': result.get('stdout', ''),
                'stderr': result.get('stderr', ''),
                'mode': 'normal'
            })
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_execution(request, execution_id):
    """Get execution details"""
    try:
        execution = CodeExecution.objects.get(id=execution_id)
        return JsonResponse({
            'id': execution.id,
            'code': execution.code,
            'output': execution.execution_output,
            'memory_trace': execution.miri_trace,
            'status': execution.status,
            'created_at': execution.created_at.isoformat(),
        })
    except CodeExecution.DoesNotExist:
        return JsonResponse({'error': 'Execution not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def execute_python_code(request):
    """Execute Python code and return output"""
    import logging
    logger = logging.getLogger(__name__)
    
    # Log the incoming HTTP request
    logger.info('=' * 80)
    logger.info('📥 [HTTP REQUEST RECEIVED]')
    logger.info(f'   Method: {request.method}')
    logger.info(f'   Path: {request.path}')
    logger.info(f'   Content-Type: {request.content_type}')
    logger.info(f'   Remote Address: {request.META.get("REMOTE_ADDR")}')
    logger.info(f'   Timestamp: {request.META.get("HTTP_DATE", "N/A")}')
    logger.info('=' * 80)
    
    try:
        data = json.loads(request.body)
        code = data.get('code', '')
        
        logger.info(f'📝 Code length: {len(code)} characters')
        logger.info(f'📝 First 100 chars: {code[:100]}...')
        
        if not code:
            logger.warning('⚠️  No code provided in request')
            return JsonResponse({'error': 'No code provided'}, status=400)
        
        logger.info('🚀 Executing Python code...')
        
        # Execute Python code
        executor = PythonExecutor(timeout=30)
        result = executor.execute(code)
        
        logger.info(f'✅ Execution completed - Success: {result.get("success", False)}')
        logger.info(f'⏱️  Execution time: {result.get("execution_time", 0.0):.3f}s')
        
        return JsonResponse({
            'success': result.get('success', False),
            'stdout': result.get('stdout', ''),
            'stderr': result.get('stderr', ''),
            'execution_time': result.get('execution_time', 0.0),
            'plots': result.get('plots', [])
        })
    
    except Exception as e:
        logger.error(f'❌ Error executing Python code: {str(e)}')
        return JsonResponse({'error': str(e)}, status=500)
