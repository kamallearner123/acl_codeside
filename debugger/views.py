from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .models import CodeExecution
from .services.rust_executor import RustExecutor
from .services.miri_parser import MiriParser
from .services.python_executor import PythonExecutor


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
