import os
import django
import sys

sys.path.append('/home/kamal/Documents/1.Github/acl_codeside')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "leetcode_clone.settings")
django.setup()

from courses.models import Course

html_content = """
<div class="space-y-8">
  <div class="bg-brand-navy text-white p-6 rounded-xl shadow-lg border-l-4 border-brand-coral">
    <h3 class="text-xl font-bold mb-3 flex items-center"><i class="fas fa-list-ul mr-2 text-brand-coral"></i> What to Expect from this Course</h3>
    <ul class="space-y-2 text-gray-200">
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>24 problem-driven sessions</strong> blending Python (20%), Automation (20%), AI (20%), and Agentic AI (40%).</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Hands-on weekday format</strong> (Mon, Tue, Wed, 6:00pm - 7:30pm) designed for working professionals and students.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>4 dedicated project days</strong> including Web Automation and RAG knowledge agents.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Production-ready capstone</strong>: Build a complete Enterprise Automation Assistant.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Full access to materials</strong> and API access, with recordings maintained for 3 months.</li>
    </ul>
  </div>

  <div class="bg-gray-50 p-6 rounded-xl border border-gray-100 shadow-sm">
    <h3 class="text-xl font-bold text-brand-navy mb-4">Course Philosophy</h3>
    <p class="mb-2 text-brand-textSecondary text-lg"><strong>Traditional:</strong> Technology &rarr; Concept &rarr; Exercise</p>
    <p class="mb-4 text-brand-textSecondary text-lg"><strong>Problem-driven:</strong> Problem &rarr; Failure &rarr; Question &rarr; Concept &rarr; Technology &rarr; Implementation &rarr; Evaluation</p>
    <p class="text-brand-textSecondary text-lg">The course treats AI technologies as engineering solutions rather than topics to memorize. Each day starts with a realistic problem and introduces the technology only when learners encounter the limitation that requires it.</p>
  </div>

  <div>
    <h3 class="text-2xl font-bold text-brand-navy mb-6 border-b-2 border-brand-coral inline-block pb-2">The Learning Journey (24-Day Curriculum)</h3>

    <!-- Day 1 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 1:</span> Python Fundamentals &amp; Variables</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You need to process data for an AI model, but you lack the basic programming syntax to write scripts.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Python basics, variables, data types, string manipulation.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Write a script to calculate and format a dataset.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build a solid foundation in Python syntax.</p>
    </div>

    <!-- Day 2 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 2:</span> Data Structures for AI</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> AI models return complex data. How do you store, access, and manipulate this information efficiently?</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Lists, Dictionaries, Sets, Tuples, and basic JSON mapping.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Create a nested dictionary to represent an AI's tool schema.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Master data organization for AI workflows.</p>
    </div>

    <!-- Day 3 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 3:</span> Control Flow &amp; Logic</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Your script needs to make decisions based on an AI's response, but it only runs in a straight line.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> If/else statements, loops, error handling (try/except).</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Write a script that retries a failed operation automatically.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build resilient logic flows.</p>
    </div>

    <!-- Day 4 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 4:</span> Interacting with APIs</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> The LLM exists on a remote server. How does your local script talk to it?</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> HTTP requests, REST APIs, headers, payloads, the requests library.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Send a GET request to a public API and parse the JSON.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Connect your code to the outside world.</p>
    </div>

    <!-- Day 5 -->
    <div class="mb-6 pl-4 border-l border-gray-200 bg-brand-coral/5 p-4 rounded-r-lg">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 5:</span> Project Session: Public Data Fetcher</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Focus:</strong> Dedicated hands-on session to build the Week 1 project.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Build:</strong> A Python application that connects to an external API, retrieves data, cleans it, and outputs a formatted JSON file.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Complete a working data pipeline.</p>
    </div>

    <!-- Day 6 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 6:</span> Web Scraping Basics</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You need information from a website to feed your AI, but the site has no API.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> HTML structure, BeautifulSoup, CSS selectors, data extraction.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Extract a table of data from a static webpage.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Turn any public website into a data source.</p>
    </div>

    <!-- Day 7 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 7:</span> Browser Automation</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> The data you need is hidden behind a login screen and dynamic JavaScript.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Selenium / Playwright, browser drivers, interacting with elements, waits.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Write a script that logs into a site and clicks a button to download a file.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Automate complex browser interactions.</p>
    </div>

    <!-- Day 8 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 8:</span> Working with Data (Pandas)</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You have a massive CSV file that needs cleaning before an AI can process it.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Pandas basics, DataFrames, filtering, applying functions, exporting.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Load a messy CSV, remove nulls, and export a clean dataset.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Prepare real-world data for AI ingestion.</p>
    </div>

    <!-- Day 9 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 9:</span> Scheduling &amp; Background Tasks</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Your automation script works, but you have to run it manually every day.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Cron jobs, background tasks, the schedule library, logging.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Set up a script to run automatically every hour and log its results.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Create truly autonomous background jobs.</p>
    </div>

    <!-- Day 10 -->
    <div class="mb-6 pl-4 border-l border-gray-200 bg-brand-coral/5 p-4 rounded-r-lg">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 10:</span> Project Session: Daily Report Generator</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Focus:</strong> Dedicated hands-on session to build the automation project.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Build:</strong> An automated workflow that scrapes data, analyzes it with Pandas, and sends a scheduled summary email.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Complete a fully automated reporting pipeline.</p>
    </div>

    <!-- Day 11 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 11:</span> The AI Landscape &amp; LLM Internals</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You ask an AI a question, but it gives a generic or slightly off-topic answer. Why isn't it "thinking" like a human?</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> ML basics, transformers, next-token prediction, tokens vs words.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Count tokens in a prompt and observe how context shifts outputs.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Gain a mental model of how LLMs generate text.</p>
    </div>

    <!-- Day 12 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 12:</span> Prompt Engineering</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You have a complex problem, but the LLM gives a simplistic answer. How do you systematically improve it?</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> System prompts, context windows, providing constraints, output formatting.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Design a prompt that forces the LLM to analyze step-by-step.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build a reusable problem-solving prompt template.</p>
    </div>

    <!-- Day 13 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 13:</span> Few-Shot Prompting &amp; Roles</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> The LLM doesn't match the specific tone or structure your business requires.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Role assignment, few-shot examples, JSON structured output.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Provide 3 examples to force the LLM to return valid JSON matching your schema.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Create highly specialized, predictable AI behavior.</p>
    </div>

    <!-- Day 14 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 14:</span> Overcoming Hallucinations</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> The LLM confidently gives you incorrect information when it doesn't know the answer.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Knowledge cutoffs, probabilistic generation, grounding, forcing 'I don't know'.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Design a prompt that distinguishes Known vs Unknown data.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build trustworthy AI interactions.</p>
    </div>

    <!-- Day 15 -->
    <div class="mb-6 pl-4 border-l border-gray-200 bg-brand-coral/5 p-4 rounded-r-lg">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 15:</span> Project Session: Robust AI Summarizer</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Focus:</strong> Dedicated hands-on session to build the AI project.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Build:</strong> An application that takes long, complex documents, extracts key facts, and reliably flags unknown areas without hallucinating.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Complete a production-grade AI summarizer.</p>
    </div>

    <!-- Day 16 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 16:</span> Function Calling (Tools)</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Ask an LLM to calculate the current weather. It can't. How do we give it access to our Python scripts?</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Tool schemas, arguments, ReAct paradigm, tool routing.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build a weather tool and trace the complete LLM-tool cycle.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build your first tool-enabled LLM.</p>
    </div>

    <!-- Day 17 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 17:</span> Intro to RAG</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> The LLM has no knowledge of your private company documents.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Embeddings, vector databases, chunks, semantic search.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Convert a text document into embeddings and perform a similarity search.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Understand the core mechanics of Retrieval-Augmented Generation.</p>
    </div>

    <!-- Day 18 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 18:</span> Advanced RAG Pipelines</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Your RAG system retrieves the wrong paragraphs, leading to bad AI answers.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Chunk overlap, Top-K, metadata filtering, re-ranking.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Measure retrieval relevance and improve it using re-ranking.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Improve retrieval quality for enterprise search.</p>
    </div>

    <!-- Day 19 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 19:</span> Visual Workflows &amp; n8n</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Writing Python glue code for every API integration is slowing you down.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Visual programming, n8n workflows, webhooks, native integrations.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build an automated data pipeline using n8n.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Rapidly prototype AI workflows without code.</p>
    </div>

    <!-- Day 20 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 20:</span> LangGraph Basics</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You need an agent that classifies, retrieves, verifies, and executes. A single prompt loop is too fragile.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> State, nodes, edges, conditional routing, cyclic graphs.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build your first multi-node LangGraph workflow.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Control complex agent logic deterministically.</p>
    </div>

    <!-- Day 21 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 21:</span> Memory &amp; Persistence</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Your agent fails at step 5 of a 10-step process. How do you resume without starting over?</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Checkpointing, state persistence, conversational memory, streaming.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Pause, persist, and resume a LangGraph agent.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build resilient, recoverable agent workflows.</p>
    </div>

    <!-- Day 22 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 22:</span> Multi-Agent Architectures</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> One huge agent trying to research, code, and review is making mistakes.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Supervisor/orchestrator, specialized workers, agent-to-agent communication.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Design a supervisor with a researcher agent and a writer agent.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Design scalable multi-agent architectures.</p>
    </div>

    <!-- Day 23 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 23:</span> Guardrails &amp; Human-in-the-Loop</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Your agent is about to delete a database table automatically. How do you stop it?</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Hard stops, human approval gates, preventing prompt injection.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Add a human approval node before a destructive tool execution.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build a safe, production-ready guarded agent.</p>
    </div>

    <!-- Day 24 -->
    <div class="mb-6 pl-4 border-l border-gray-200 bg-brand-coral/5 p-4 rounded-r-lg border-l-4 border-brand-coral">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 24:</span> Major Capstone: Enterprise Automation Assistant</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Focus:</strong> Final hands-on session to build the major capstone.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Build:</strong> An end-to-end multi-agent system that autonomously retrieves internal docs, queries APIs, analyzes results, and awaits human approval.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Graduate with a complete Automated Enterprise Assistant.</p>
    </div>

  </div>

  <div class="bg-brand-navy text-white p-8 rounded-2xl shadow-xl mt-12 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] relative overflow-hidden">
    <div class="absolute inset-0 bg-brand-navy/90"></div>
    <div class="relative z-10">
        <h3 class="text-3xl font-bold mb-4 text-white flex items-center"><i class="fas fa-rocket mr-3 text-brand-coral"></i> Weekly Projects &amp; Capstone</h3>
        
        <div class="mb-6 bg-white/10 p-5 rounded-lg border border-white/20 backdrop-blur-sm">
            <h4 class="text-xl font-bold text-brand-coral mb-2">Module 1 Project: API Data Fetcher</h4>
            <p class="text-gray-200">Extract data automatically using Python requests and transform it cleanly into structured JSON.</p>
        </div>
        <div class="mb-6 bg-white/10 p-5 rounded-lg border border-white/20 backdrop-blur-sm">
            <h4 class="text-xl font-bold text-brand-coral mb-2">Module 2 Project: Daily Report Generator</h4>
            <p class="text-gray-200">Automatically scrape browser data, process it via Pandas, and send an automated scheduled email.</p>
        </div>
        <div class="mb-6 bg-white/10 p-5 rounded-lg border border-white/20 backdrop-blur-sm">
            <h4 class="text-xl font-bold text-brand-coral mb-2">Module 3 Project: AI Document Summarizer</h4>
            <p class="text-gray-200">A robust prompt-engineered summarizer that effectively handles long context without hallucinating.</p>
        </div>
        <div class="bg-brand-coral/20 p-5 rounded-lg border border-brand-coral backdrop-blur-sm mt-8">
            <h4 class="text-xl font-bold text-brand-coral mb-2">Module 4 Capstone: Enterprise Agentic Assistant</h4>
            <p class="text-gray-200">A full-blown LangGraph multi-agent architecture capable of retrieving internal documentation, generating logic, and calling real software tools safely behind a human approval layer.</p>
        </div>
    </div>
  </div>

  <div class="mt-8 text-center sm:flex-row justify-center gap-4">
    <a href="/courses/" class="inline-block px-8 py-3 bg-brand-coral text-white font-bold rounded-full hover:bg-opacity-90 shadow-lg">Browse more courses</a>
  </div>
</div>
"""

Course.objects.update_or_create(
    slug="agentic-ai-python-automation",
    defaults={
        "title": "Agentic AI with Python & Automation",
        "short_description": "A 24-Session Problem-Driven Curriculum for Working Professionals and Students.",
        "description": html_content,
        "duration_hours": 36,
        "skill_level": "Beginner to Advanced",
        "technologies": "Python, Selenium, Pandas, LangChain, LangGraph, Agents, APIs",
        "fee": "4000/-",
        "timing": "Mon, Tue, Wed, 6:00pm to 7:30pm",
        "start_date": "15th Sep 2026 - 4th Nov 2026",
        "trainers": "Kamal Kumar Mukiri, Dhanush Boyapati",
        "recordings_info": "Sessions are recorded and shared and maintained for 3 months.",
        "materials_info": "Materials will be provided and hands on programs with APIs with limited access.",
        "prerequisites": "Basic understanding of programming concepts.",
        "training_mode": "Online",
        "image": "https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&q=80&w=800"
    }
)

print("Updated 'Agentic AI with Python & Automation' course to match exact layout!")
