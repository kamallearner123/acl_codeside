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
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>24 problem-driven sessions</strong> blending Python, ML, and advanced Agentic AI engineering.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Hands-on weekend format</strong> (Sat &amp; Sun, 4:00pm - 5:30pm) designed for working professionals.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>4 dedicated project days</strong> including AI Problem Solvers and RAG knowledge agents.</li>
      <li><i class="fas fa-check-circle text-brand-coral mr-2"></i><strong>Production-ready capstone</strong>: Build a complete Automotive Engineering Assistant.</li>
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
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 1:</span> The AI Landscape: From Basic ML to LLMs</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You hear terms like Machine Learning, Deep Learning, and Large Language Models, but how do they connect, and why are LLMs so revolutionary?</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Traditional Programming vs. Machine Learning &bull; Neural Networks &bull; The Transformer Architecture &bull; What makes an LLM "Large".</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Compare a simple ML classification task with a generative LLM task conceptually.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build a solid conceptual foundation of the AI landscape before diving into LLM internals.</p>
    </div>

    <!-- Day 2 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 2:</span> Python Fundamentals for Agentic AI</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You need to interact with APIs and process structured data, but you don't know the core programming tools required for AI development.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Python basics &bull; Variables &bull; Dictionaries &bull; API requests (requests library) &bull; JSON parsing</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Write a script to fetch data from a public API and parse the JSON response.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build a solid Python foundation for AI integration.</p>
    </div>

    <!-- Day 3 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 3:</span> What is an LLM actually doing under the hood?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You ask an AI a question, but it gives a generic or slightly off-topic answer. Why isn't it "thinking" the way a human does, and how does it actually generate its response?</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Next-token prediction &bull; Context windows &bull; Tokens vs. Words &bull; Why embeddings (numbers) are used instead of text.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Count tokens in a prompt, and observe how changing one word shifts the probability of the AI's next word.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Gain a mental model of an LLM to write better prompts and understand limitations.</p>
    </div>

    <!-- Day 4 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 4:</span> Prompt Engineering &amp; Overcoming Hallucinations</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You need an LLM to give precise answers but it sometimes hallucinates or gives generic responses.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Context &bull; System vs user instructions &bull; Prompt structure &bull; Few-shot prompting &bull; Grounding.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Design a prompt that gives accurate results and distinguishes Known &rarr; Unknown.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Design a robust prompt template that distinguishes known and unknown information.</p>
    </div>

<!-- Day 5 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 5:</span> How do you make an LLM behave like a specialist?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Take a generic ChatGPT and make it behave like a senior software architect.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Role • Domain context • Rules • Examples • Constraints • Expected output</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Prompt engineering • Few-shot prompting • Structured output • JSON • Prompt evaluation</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Create and test a specialist prompt.</p>
    </div>


    <!-- Day 6 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 6:</span> Can an LLM actually perform work?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Ask an LLM to calculate something, read data, search information and perform an operation. Determine what the LLM can and cannot do by itself.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Tools • Function calling • Tool schemas • Arguments • Tool results • LLM-tool loop</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build a calculator tool and trace the complete call cycle.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build the first working tool-enabled LLM.</p>
    </div>

    <!-- Day 7 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 7:</span> How do you make AI follow a repeatable process?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Build a system that executes Understand &rarr; Analyze &rarr; Calculate &rarr; Verify &rarr; Respond. Identify why a single prompt becomes unreliable.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Prompt chaining • Structured outputs • Intermediate results • Validation • Deterministic workflows</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build an AI Problem Solver that classifies, decomposes, solves, validates and formats a problem.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Week 1 project: AI Problem Solver.</p>
    </div>

    
    <!-- Day 8 -->
    <div class="mb-6 pl-4 border-l border-gray-200 bg-brand-coral/5 p-4 rounded-r-lg">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 8:</span> Project Session: AI Problem Solver</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Focus:</strong> Dedicated hands-on session to build the Week 1 project.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Build:</strong> A Python application that accepts a problem, classifies it, breaks it into steps, uses an LLM, validates the output and produces structured results.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Complete a working AI Problem Solver.</p>
    </div>

    <!-- Day 9 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 9:</span> Why isn't ChatGPT enough?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Give an AI a task requiring search, reading, extraction, comparison, analysis and reporting. Determine what must happen for the system to operate as an agent.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Assistant vs workflow vs agent • Planning • Decision-making • Agentic behavior • Multi-step execution</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Map a real manual process into an agent-style workflow.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Design an agent for a real business task.</p>
    </div>

    <!-- Day 10 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 10:</span> Can AI use your software?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Build an AI that can call a calculator, weather API, database or file reader. The LLM must decide which tool to use.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Function calling • Pydantic • JSON Schema • Tool descriptions • Tool selection • ReAct</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build a two-tool agent and trace its decisions.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build an agent that selects and calls tools.</p>
    </div>

    <!-- Day 11 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 11:</span> Why doesn't AI know what's inside my documents?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Give an LLM a large private document and ask a question about information buried deep inside it. Build a system that retrieves the relevant content.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Documents • Chunks • Embeddings • Vector search • Retrieved context • Grounded generation</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build the first RAG pipeline.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Document &rarr; chunks &rarr; embeddings &rarr; retrieval &rarr; answer.</p>
    </div>

    <!-- Day 12 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 12:</span> Why does RAG sometimes retrieve the wrong information?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Create a knowledge base containing similar documents. Compare keyword search and semantic search and investigate why retrieval quality changes the final answer.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Chunk size • Overlap • Embeddings • Similarity • Top-k • Metadata • Hybrid search • Re-ranking</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Measure retrieval relevance and compare strategies.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Improve retrieval quality using experiments.</p>
    </div>

    <!-- Day 13 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 13:</span> Can we build this without writing all the glue code?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Rebuild the RAG pipeline using LangChain and compare the abstraction with the Python implementation.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Models • Prompt templates • Retrievers • Chains • LCEL • Output parsers</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Rebuild the pipeline using LCEL.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Use LangChain as an engineering abstraction.</p>
    </div>

    <!-- Day 14 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 14:</span> Can we automate workflows visually using n8n?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> You want to connect multiple tools and an LLM, but writing and maintaining Python glue code for every integration is becoming too slow.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Visual programming • n8n workflows • Webhooks • HTTP requests • Built-in integrations</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build an automated AI email responder or data processing pipeline in n8n.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Create a no-code/low-code AI automated workflow.</p>
    </div>

    <!-- Day 15 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 15:</span> Can the RAG system decide when to use tools?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Answer a question that requires both private-document retrieval and an external operation, such as calculating a value or querying a database.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> RAG + tools • Agent decisions • Tool routing • Error handling</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build a company knowledge agent.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Week 2 project: Company Knowledge Agent.</p>
    </div>

    
    <!-- Day 16 -->
    <div class="mb-6 pl-4 border-l border-gray-200 bg-brand-coral/5 p-4 rounded-r-lg">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 16:</span> Project Session: Company Knowledge Agent</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Focus:</strong> Dedicated hands-on session to build the Week 2 project.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Build:</strong> A RAG-powered agent that searches company documents, answers with evidence, selects tools when required and handles missing information.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Complete a working Company Knowledge Agent.</p>
    </div>

    <!-- Day 17 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 17:</span> Why isn't a simple agent enough?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Design an agent that must classify, retrieve, analyze, verify, obtain approval and execute. Some paths branch, some repeat and some require humans.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Nodes • Edges • State • Conditional routing • Loops</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Build the first LangGraph workflow.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Control complex agent workflows with LangGraph.</p>
    </div>

    <!-- Day 18 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 18:</span> What happens when an agent fails halfway?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Make an eight-step agent fail after step five. Determine how to resume without starting from the beginning.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> State • Checkpoints • Persistence • Threads • Resume • Streaming • Interrupt/resume</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Pause, persist and resume an agent.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build a recoverable agent workflow.</p>
    </div>

    <!-- Day 19 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 19:</span> Why use multiple agents?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Build one large agent responsible for research, coding, validation and reporting. Identify where specialization improves control and where it creates unnecessary complexity.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Supervisor/orchestrator • Specialized agents • Agent-to-agent communication • Architecture trade-offs</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Design a supervisor with research, coding and reviewer agents.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Design a justified multi-agent architecture.</p>
    </div>

    <!-- Day 20 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 20:</span> How can an agent become dangerous?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Expose an agent to malicious retrieved content, prompt injection and dangerous tools. Determine what permissions and approval gates are required.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> Prompt injection • Indirect injection • Tool abuse • Excessive permissions • Data leakage • Guardrails • Human approval</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Attack a deliberately vulnerable agent and harden it.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Build a guarded agent.</p>
    </div>

    <!-- Day 21 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 21:</span> How do you turn your agent into a product?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Take a working terminal-based agent and turn it into a usable application.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> UI • API • Configuration • Logging • Tracing • Evaluation • Deployment • MCP awareness</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Wrap the agent with Streamlit or Gradio.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Deploy a usable agent application.</p>
    </div>

    <!-- Day 22 -->
    <div class="mb-6 pl-4 border-l border-gray-200">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 22:</span> How do you build and ship a real agent?</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Problem:</strong> Build an end-to-end engineering assistant combining RAG, tools, memory, routing, human approval and guardrails.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Discover:</strong> RAG • Tools • Memory • LangGraph • Conditional routing • Human-in-the-loop • Guardrails • Deployment</p>
      <p class="text-brand-textSecondary mb-1"><strong>Practice:</strong> Integrate all components and demonstrate the complete workflow.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Week 3 project / Major capstone: production-style agentic system.</p>
    </div>



    <!-- Day 23 -->
    <div class="mb-6 pl-4 border-l border-gray-200 bg-brand-coral/5 p-4 rounded-r-lg">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 23:</span> Project Session: Recoverable Agent Workflow</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Focus:</strong> Dedicated hands-on session to build the Week 3 project.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Build:</strong> A multi-node LangGraph agent with conditional routing, persistence/checkpointing and human-in-the-loop approval, wrapped in a basic UI.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Complete a working Recoverable Agent Workflow.</p>
    </div>

    <!-- Day 24 -->
    <div class="mb-6 pl-4 border-l border-gray-200 bg-brand-coral/5 p-4 rounded-r-lg border-l-4 border-brand-coral">
      <h4 class="text-lg font-bold text-brand-navy mb-2"><span class="text-brand-coral">Day 24:</span> Major Capstone: Automotive Engineering Assistant</h4>
      <p class="text-brand-textSecondary mb-1"><strong>Focus:</strong> Final hands-on session to build and present the major capstone.</p>
      <p class="text-brand-textSecondary mb-1"><strong>Build:</strong> An end-to-end system that searches engineering documentation, retrieves specifications, calls diagnostic tools, and requests human approval.</p>
      <p class="text-brand-textSecondary"><strong>Outcome:</strong> Graduate with a production-ready Automotive Engineering Assistant.</p>
    </div>
  </div>

  <div class="bg-brand-navy text-white p-8 rounded-2xl shadow-xl mt-12 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] relative overflow-hidden">
    <div class="absolute inset-0 bg-brand-navy/90"></div>
    <div class="relative z-10">
        <h3 class="text-3xl font-bold mb-4 text-white flex items-center"><i class="fas fa-rocket mr-3 text-brand-coral"></i> Weekly Projects & Capstone</h3>
        
        <div class="mb-6 bg-white/10 p-5 rounded-lg border border-white/20 backdrop-blur-sm">
            <h4 class="text-xl font-bold text-brand-coral mb-2">Week 1: AI Problem Solver</h4>
            <p class="text-gray-200">A Python application that accepts a problem, classifies it, breaks it into steps, uses an LLM, validates the output and produces structured results.</p>
        </div>

        <div class="mb-6 bg-white/10 p-5 rounded-lg border border-white/20 backdrop-blur-sm">
            <h4 class="text-xl font-bold text-brand-coral mb-2">Week 2: Company Knowledge Agent</h4>
            <p class="text-gray-200">A RAG-powered agent that searches company documents, answers with evidence, selects tools when required and handles missing information.</p>
        </div>

        <div class="mb-6 bg-white/10 p-5 rounded-lg border border-white/20 backdrop-blur-sm">
            <h4 class="text-xl font-bold text-brand-coral mb-2">Week 3: Recoverable Agent Workflow</h4>
            <p class="text-gray-200">A multi-node LangGraph agent with conditional routing, persistence/checkpointing and human-in-the-loop approval, wrapped in a basic UI.</p>
        </div>

        <div class="bg-brand-coral/20 p-5 rounded-lg border border-brand-coral backdrop-blur-sm mt-8">
            <h4 class="text-xl font-bold text-brand-coral mb-2">Major Capstone: Automotive Engineering Assistant</h4>
            <p class="text-gray-200 mb-2"><strong>Scenario:</strong> User asks: "Why is diagnostic service X failing?"</p>
            <p class="text-gray-200">The system understands the question, searches engineering documentation, retrieves relevant specifications, decides whether additional tools are required, calls a diagnostic-data tool, analyzes the result, requests human approval before an action, remembers context and produces an evidence-backed report.</p>
        </div>
    </div>
  </div>

  <div class="mt-8 text-center sm:flex-row justify-center gap-4">
    <a href="/courses/" class="inline-block px-8 py-3 bg-brand-coral text-white font-bold rounded-full hover:bg-opacity-90 shadow-lg">Browse more courses</a>
  </div>
</div>
"""

Course.objects.update_or_create(
    slug="agentic-ai-learn-by-examples",
    defaults={
        "title": "Agentic AI - Learn by Examples",
        "short_description": "A 24-Day Problem-Driven Curriculum for Working Professionals.",
        "description": html_content,
        "duration_hours": 36,
        "skill_level": "Beginner to Advanced",
        "technologies": "Python, LangChain, RAG, LangGraph, Agents, Claude, n8n",
        "fee": "7000/- + GST",
        "timing": "Weekend course, 4:00pm to 5:30pm",
        "start_date": "12th Sep 2026 - 29th Nov 2026",
        "trainers": "Kamal Kumar Mukiri, Dhanush Boyapati",
        "recordings_info": "Sessions are recorded and shared and maintained for 3 months.",
        "materials_info": "Materials will be provided and hands on programs with APIs with limited access.",
        "prerequisites": "Basic understanding of Python and programming concepts.",
        "training_mode": "Online",
        "image": "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?auto=format&fit=crop&q=80&w=800"
    }
)

print("Updated 'Agentic AI - Learn by Examples' course!")
