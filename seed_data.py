import os
import django
import sys

sys.path.append('/home/kamal/Documents/1.Github/acl_codeside')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "leetcode_clone.settings")
django.setup()

from services.models import Service
from blogs.models import Post
from events.models import Event
from django.utils import timezone
from datetime import timedelta

print("Seeding Services...")
services_data = [
    {"title": "Industry Training & MaaS", "slug": "industry-training", "icon_class": "fas fa-briefcase", "short_description": "Comprehensive corporate training and Model-as-a-Service solutions.", "description": "Full description of our industry training program.", "is_featured": True, "order": 1},
    {"title": "Rust Automotive Program", "slug": "rust-automotive", "icon_class": "fab fa-rust", "short_description": "Pioneering Rust in automotive software systems.", "description": "Full description of Rust Automotive.", "is_featured": True, "order": 2},
    {"title": "Automotive IDS/IDPS", "slug": "automotive-ids", "icon_class": "fas fa-shield-alt", "short_description": "Advanced intrusion detection and prevention systems.", "description": "Details about Automotive IDS.", "is_featured": False, "order": 3},
    {"title": "Building AI Agents", "slug": "building-ai-agents", "icon_class": "fas fa-robot", "short_description": "Design and implement autonomous AI agents for real-world tasks.", "description": "Agent architectures, reinforcement learning basics, and safe deployment practices.", "is_featured": False, "order": 4},
]
for data in services_data:
    Service.objects.get_or_create(slug=data['slug'], defaults=data)

print("Seeding Blogs...")
blogs_data = [
    {"title": "Why Rust is the Future of Automotive", "slug": "rust-future-automotive", "author": "Kamal", "excerpt": "Exploring memory safety in critical automotive systems.", "content": "Full article content goes here...", "image_url": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?auto=format&fit=crop&q=80&w=800"},
    {"title": "Building V2X Architecture", "slug": "building-v2x", "author": "Dhanush", "excerpt": "A deep dive into Vehicle-to-Everything communication protocols.", "content": "Full article content goes here...", "image_url": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=800"},
]
for data in blogs_data:
    Post.objects.get_or_create(slug=data['slug'], defaults=data)

print("Seeding Events...")
now = timezone.now()
events_data = [
    {"title": "Rust for Embedded Systems Workshop", "slug": "rust-embedded-workshop", "date": now + timedelta(days=10), "location": "Bangalore Campus / Online", "description": "A 2-day hands-on workshop.", "registration_link": "https://example.com/register"},
    {"title": "Automotive Cybersecurity Summit", "slug": "auto-cyber-summit", "date": now + timedelta(days=30), "location": "Online", "description": "Discussing the latest trends in IDS/IDPS.", "registration_link": "https://example.com/register"},
]
for data in events_data:
    Event.objects.get_or_create(slug=data['slug'], defaults=data)

print("Seeding Complete!")

# --- Courses ---
try:
    from courses.models import Course
    print("Seeding Courses...")
    courses_data = [
        {
            'title': 'Agentic AI',
            'slug': 'agentic-ai',
            'short_description': 'An intensive, hands-on engineering program focused on building, optimizing, and evaluating production-ready RAG pipelines and LLM agents.',
            'description': """
<div class="space-y-8">
  <div class="bg-gray-50 p-6 rounded-xl border border-gray-100 shadow-sm">
    <h3 class="text-xl font-bold text-brand-navy mb-4">COURSE SYLLABUS & DETAILS</h3>
    <p class="mb-4 text-brand-textSecondary text-lg">An intensive, hands-on engineering program focused on building, optimizing, and evaluating production-ready Retrieval-Augmented Generation (RAG) pipelines and LLM agents.</p>
    <ul class="list-none space-y-3 mt-4">
      <li class="flex items-start"><i class="fas fa-clock mt-1 text-brand-coral mr-3"></i> <strong>Duration:</strong> <span class="ml-2">10 Working Days (8 hours/day, 80 hours total)</span></li>
      <li class="flex items-start"><i class="fas fa-chalkboard-teacher mt-1 text-brand-coral mr-3"></i> <strong>Format:</strong> <span class="ml-2">Daily lecture & interactive demo, hands-on lab, and checkpoint review</span></li>
      <li class="flex items-start"><i class="fas fa-bullseye mt-1 text-brand-coral mr-3"></i> <strong>Key Outcomes:</strong> <span class="ml-2">Production-grade RAG application, tool-using autonomous agent, and Git-tracked capstone project</span></li>
    </ul>
  </div>

  <div>
    <h3 class="text-2xl font-bold text-brand-navy mb-6 border-b-2 border-brand-coral inline-block pb-2">Week 1: Foundations of RAG Pipelines</h3>
    
    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-3"><span class="text-brand-coral">Day 1:</span> Environment Setup, Prompt Engineering & LangChain RAG Primitives</h4>
      <ul class="list-disc pl-5 mb-4 space-y-2 text-brand-textSecondary">
        <li><strong>Core Topics:</strong> Local Python environment synchronization, Git repository strategy, LLM API credentials orchestration, vector database ecosystem, and project directory design.</li>
        <li><strong>LangChain Ingestion Functions:</strong> Technical deep dive into LangChain document utility modules, including <code>PyPDFLoader</code>, <code>TextLoader</code>, and programmatic extraction utilities. Introduction to Runnable wrappers for data parsing.</li>
        <li><strong>Prompt Engineering:</strong> Prompt chains, deterministic structured outputs with Pydantic schemas, programmatic data extraction, classification, rewriting, and token-level hallucination detection patterns.</li>
      </ul>
      <div class="bg-brand-light/50 p-4 rounded-lg border-l-4 border-brand-coral shadow-sm">
        <strong class="text-brand-navy"><i class="fas fa-laptop-code mr-2"></i>HANDS-ON LAB:</strong> Construct a modular prompt management workflow that enforces strict JSON output layouts with integrated Pydantic schema validation, and write a LangChain-based parsing routine to extract raw context from structured inputs.
      </div>
    </div>

    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-3"><span class="text-brand-coral">Day 2:</span> Document Parsing, Splitting & Chunking Architecture</h4>
      <ul class="list-disc pl-5 mb-4 space-y-2 text-brand-textSecondary">
        <li><strong>Core Topics:</strong> Document loaders and ingestion layers, semantic and syntactic text parsing pipelines, strategic text chunking (fixed-size vs. recursive character splitting), chunk overlapping heuristics, and optimized metadata schema design.</li>
        <li><strong>Engineering Impact:</strong> Mitigating semantic fragmentation, managing context boundaries, and optimizing target text block extraction for maximum downstream relevance.</li>
      </ul>
      <div class="bg-brand-light/50 p-4 rounded-lg border-l-4 border-brand-coral shadow-sm">
        <strong class="text-brand-navy"><i class="fas fa-laptop-code mr-2"></i>HANDS-ON LAB:</strong> Ingest messy, real-world regulatory documentation into a clean, queryable document schema, spinning up automated comparative chunk size and chunk overlap experiments.
      </div>
    </div>

    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-3"><span class="text-brand-coral">Day 3:</span> Mathematical Foundations of Text Embeddings</h4>
      <ul class="list-disc pl-5 mb-4 space-y-2 text-brand-textSecondary">
        <li><strong>Core Topics:</strong> The mechanics of text embeddings, multi-dimensional vector space representations, cosine similarity vs. Euclidean distance metrics, token sequence constraints, and the absolute mathematical rationale behind semantic retrieval.</li>
        <li><strong>Engineering Impact:</strong> Vector alignment verification, identifying semantic drift in specialized domains, and measuring embedding distribution quality.</li>
      </ul>
      <div class="bg-brand-light/50 p-4 rounded-lg border-l-4 border-brand-coral shadow-sm">
        <strong class="text-brand-navy"><i class="fas fa-laptop-code mr-2"></i>HANDS-ON LAB:</strong> Programmatically extract vector embeddings from the raw document chunks generated on Day 2, script manual matrix similarity operations to observe indexing performance, and inspect nearest-neighbor cluster anomalies.
      </div>
    </div>

    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-3"><span class="text-brand-coral">Day 4:</span> Deep Dive into Vector Databases</h4>
      <ul class="list-disc pl-5 mb-4 space-y-2 text-brand-textSecondary">
        <li><strong>Core Topics:</strong> Vector database internals (Chroma and FAISS architectures), HNSW vs. Flat indexing strategies, strict structural metadata filtering, storage persistence configurations, and identifying optimal stores based on workload constraints.</li>
      </ul>
      <div class="bg-brand-light/50 p-4 rounded-lg border-l-4 border-brand-coral shadow-sm">
        <strong class="text-brand-navy"><i class="fas fa-laptop-code mr-2"></i>HANDS-ON LAB:</strong> Index identical document chunks into Chroma and FAISS, executing a suite of stress-tests to compare query latency, structural search quality, and data retrieval throughput.
      </div>
    </div>

    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-3"><span class="text-brand-coral">Day 5:</span> Synthesis – End-to-End Basic RAG Pipeline</h4>
      <ul class="list-disc pl-5 mb-4 space-y-2 text-brand-textSecondary">
        <li><strong>Core Topics:</strong> Connecting the pipeline links: Ingestion &rarr; Ingestion Loader Parsing &rarr; Chunking Mechanics &rarr; Embedding Generation &rarr; Indexing &rarr; Query-Time Retrieval &rarr; Context Augmentation &rarr; Token Generation.</li>
      </ul>
      <div class="bg-brand-light/50 p-4 rounded-lg border-l-4 border-brand-coral shadow-sm">
        <strong class="text-brand-navy"><i class="fas fa-laptop-code mr-2"></i>HANDS-ON LAB:</strong> Architect an end-to-end Document-QA conversational chatbot capable of executing reliable queries across isolated document corpuses with verbatim inline citation mappings and interactive metadata tracking source boards.
      </div>
    </div>
  </div>

  <div>
    <h3 class="text-2xl font-bold text-brand-navy mb-6 border-b-2 border-brand-coral inline-block pb-2 mt-4">Week 2: Advanced Optimization & Autonomous Agents</h3>
    
    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-3"><span class="text-brand-coral">Day 6:</span> Advanced Retrieval & Context Engineering</h4>
      <ul class="list-disc pl-5 mb-4 space-y-2 text-brand-textSecondary">
        <li><strong>Core Topics:</strong> Optimizing precision/recall, tuning dynamic Top-K values, cross-encoder reranking algorithms, query expansion/rewriting frameworks, and context window compression strategies.</li>
      </ul>
      <div class="bg-brand-light/50 p-4 rounded-lg border-l-4 border-brand-coral shadow-sm">
        <strong class="text-brand-navy"><i class="fas fa-laptop-code mr-2"></i>HANDS-ON LAB:</strong> Set up an isolated testing loop to benchmark retrieval quality shifts across alternating algorithm configurations, logging metrics to track system optimizations.
      </div>
    </div>

    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-3"><span class="text-brand-coral">Day 7:</span> LangChain Expression Language (LCEL) & Infrastructure</h4>
      <ul class="list-disc pl-5 mb-4 space-y-2 text-brand-textSecondary">
        <li><strong>Core Topics:</strong> Composing complex logic with LCEL primitives, Custom Retrievers, dynamic prompt serialization, stateful memory tracking abstractions (Buffer, Window, Summary), and deep tracing/observability instrumentation.</li>
      </ul>
      <div class="bg-brand-light/50 p-4 rounded-lg border-l-4 border-brand-coral shadow-sm">
        <strong class="text-brand-navy"><i class="fas fa-laptop-code mr-2"></i>HANDS-ON LAB:</strong> Refactor the monolithic Day 5 conversational backend into a highly composable, observable, and thread-safe chain with localized session logging and tracing hooks.
      </div>
    </div>

    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-3"><span class="text-brand-coral">Day 8:</span> Autonomous System Design – LLM Agents</h4>
      <ul class="list-disc pl-5 mb-4 space-y-2 text-brand-textSecondary">
        <li><strong>Core Topics:</strong> The core agentic control loops: Perceive, Plan, Act, and Reflect. Executing ReAct frameworks, functional tool-calling APIs, handling parsing edge cases, and runtime failure recovery patterns.</li>
      </ul>
      <div class="bg-brand-light/50 p-4 rounded-lg border-l-4 border-brand-coral shadow-sm">
        <strong class="text-brand-navy"><i class="fas fa-laptop-code mr-2"></i>HANDS-ON LAB:</strong> Build an autonomous routing agent that evaluates user input to selectively pull knowledge-base documentation, execute auxiliary mathematical helper scripts, or respond immediately using generalized parametric knowledge.
      </div>
    </div>

    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-3"><span class="text-brand-coral">Day 9:</span> Evaluation Frameworks & Prompt Version Control</h4>
      <ul class="list-disc pl-5 mb-4 space-y-2 text-brand-textSecondary">
        <li><strong>Core Topics:</strong> Treating prompts as system source code, implementation of deterministic unit tests, Git-driven prompt history/rollback workflows, and structuring human-in-the-loop and LLM-as-a-judge A/B validation metrics.</li>
      </ul>
      <div class="bg-brand-light/50 p-4 rounded-lg border-l-4 border-brand-coral shadow-sm">
        <strong class="text-brand-navy"><i class="fas fa-laptop-code mr-2"></i>HANDS-ON LAB:</strong> Develop a regression testing suite containing targeted validation cases, running competitive A/B tests between prompt versions to document safety metrics.
      </div>
    </div>

    <div class="mb-8 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-3"><span class="text-brand-coral">Day 10:</span> Capstone Production Optimization & Deployment</h4>
      <ul class="list-disc pl-5 mb-4 space-y-2 text-brand-textSecondary">
        <li><strong>Core Topics:</strong> Code review patterns for large model deployments, continuous logging, robust parameter configuration, and compiling comprehensive technical developer documentation.</li>
      </ul>
      <div class="bg-brand-light/50 p-4 rounded-lg border-l-4 border-brand-coral shadow-sm">
        <strong class="text-brand-navy"><i class="fas fa-laptop-code mr-2"></i>HANDS-ON LAB:</strong> Finalize, optimize, and push your end-to-end repository with comprehensive multi-tier testing data, architecture schematics, and an explicit dependency setup blueprint.
      </div>
    </div>
  </div>

  <div class="bg-brand-navy text-white p-8 rounded-2xl shadow-xl mt-12 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] relative overflow-hidden">
    <div class="absolute inset-0 bg-brand-navy/90"></div>
    <div class="relative z-10">
        <h3 class="text-3xl font-bold mb-4 text-white flex items-center"><i class="fas fa-rocket mr-3 text-brand-coral"></i> Capstone Project Profile</h3>
        <div class="mb-6 bg-white/10 p-5 rounded-lg border border-white/20 backdrop-blur-sm">
            <p class="text-lg"><strong class="text-brand-coral uppercase tracking-wider text-sm block mb-1">Objective</strong> Build a comprehensive, production-grade RAG Conversational System running over a specialized, proprietary enterprise knowledge base.</p>
        </div>
        <ul class="list-none mb-8 space-y-3">
          <li class="flex items-center text-gray-200"><i class="fas fa-check-circle text-brand-coral mr-3"></i> Fully tracked and cleanly segmented Git repository.</li>
          <li class="flex items-center text-gray-200"><i class="fas fa-check-circle text-brand-coral mr-3"></i> At least one functional, autonomous Tool-Calling agent capability.</li>
        </ul>
        <h4 class="text-xl font-bold mb-4 text-white border-b border-gray-600 pb-2">Core Deliverables</h4>
        <ul class="list-none space-y-3">
          <li class="flex items-center text-gray-200"><i class="fas fa-file-alt text-brand-coral mr-3"></i> Integrated prompt variation history log with an adjacent validation sheet.</li>
          <li class="flex items-center text-gray-200"><i class="fas fa-book text-brand-coral mr-3"></i> Production-ready markdown documentation detailing onboarding instructions.</li>
    </div>
  </div>
  
  <div class="mt-10 mb-4 text-center flex flex-col sm:flex-row justify-center gap-4 print:hidden">
    <a href="/agentic-ai/" target="_blank" rel="noopener noreferrer" class="bg-brand-coral hover:bg-opacity-90 text-white px-8 py-3.5 rounded-full font-bold shadow-lg transition-transform hover:-translate-y-0.5 inline-flex items-center justify-center">
      Go to full content <i class="fas fa-arrow-right ml-2"></i>
    </a>
    <button type="button" onclick="window.print()" class="bg-gray-800 hover:bg-gray-700 text-white px-8 py-3.5 rounded-full font-bold shadow-lg transition-transform hover:-translate-y-0.5 inline-flex items-center justify-center">
      <i class="fas fa-file-pdf mr-2"></i> Download as PDF
    </button>
  </div>
</div>
""",
            'duration_weeks': 2,
            'duration_hours': 80,
            'skill_level': 'Advanced',
            'technologies': 'Python, LangChain, RAG, LLMs, Agents, Vector DB',
            'image': 'https://images.unsplash.com/photo-1677442135703-178738ea3db5?auto=format&fit=crop&q=80&w=800'
        }
    ]
    for data in courses_data:
        Course.objects.update_or_create(slug=data['slug'], defaults=data)
except Exception:
    # If courses app isn't available in this environment, skip silently
    pass
