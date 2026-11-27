"""
generate_part6_helpers.py - Masterclass articles for Modules 27 to 32
"""

# -------------------------------------------------------------
# MODULE 27: Scalable I/O Multiplexing (epoll)
# -------------------------------------------------------------
MOD27_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 6 • Linux System Programming</span>
          <h1 class="chapter-h1">Module 27: Scalable I/O Multiplexing (epoll)</h1>
          <p class="chapter-lead">Master high-concurrency event-driven network I/O in C. Deep dive into the limits of select and poll (the C10K problem), Linux epoll kernel architecture (Red-Black tree and ready-list wait queues), Edge-Triggered (EPOLLET) vs Level-Triggered (EPOLLLT) mechanics, and non-blocking socket drains.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 60m</div>
            <div class="meta-item">🎯 Standard: Linux epoll(7) / Non-Blocking POSIX Sockets</div>
            <div class="meta-item">⚡ Target: High-Concurrency Linux Network Daemons</div>
          </div>
        </div>

        <h2 id="sec-27-1">27.1 The C10K Problem & The Death of select/poll</h2>
        <p>
          In early Unix network programming, handling multiple concurrent network connections relied on <code>select(2)</code> or <code>poll(2)</code>.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Algorithmic Scaling: select/poll vs epoll</div>
          <pre style="margin:0;color:#94A3B8;">
1. select() / poll() - O(N) Linear Scanning:
   User Space ──(Passes 10,000 FDs array)──▶ Kernel
                                               │ Iterates across all 10,000 FDs linearly
                                               ▼
   User Space ◀──(Returns 10,000 FDs array)── Kernel
   (User must iterate 10,000 FDs in user space to find the 2 active sockets!)

2. epoll - O(1) Ready-List Traversal:
   User Space ──(epoll_wait)──▶ Kernel
                                  │ Reads only the active 'rdllist' (populated by ISRs)
                                  ▼
   User Space ◀──(Returns ONLY the 2 active FDs!)── Kernel
          </pre>
        </div>

        <table style="width:100%;border-collapse:collapse;margin:1.5rem 0;font-size:0.9rem;">
          <thead>
            <tr style="background:#1E293B;color:#F8FAFC;border-bottom:2px solid #334155;">
              <th style="padding:10px;text-align:left;">Dimension</th>
              <th style="padding:10px;text-align:left;">select(2)</th>
              <th style="padding:10px;text-align:left;">poll(2)</th>
              <th style="padding:10px;text-align:left;">epoll(7)</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Computational Complexity</strong></td>
              <td style="padding:10px;"><code>O(N)</code></td>
              <td style="padding:10px;"><code>O(N)</code></td>
              <td style="padding:10px;"><strong style="color:#10B981;"><code>O(1)</code></strong> (proportional only to active events)</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>Max File Descriptors</strong></td>
              <td style="padding:10px;">Hardcoded 1024 (<code>FD_SETSIZE</code>)</td>
              <td style="padding:10px;">Limited by process FD limit</td>
              <td style="padding:10px;"><strong style="color:#10B981;">Unlimited</strong> (Millions of sockets)</td>
            </tr>
            <tr style="border-bottom:1px solid #334155;">
              <td style="padding:10px;"><strong>User-Kernel Copies</strong></td>
              <td style="padding:10px;">Copies full bitmask every syscall</td>
              <td style="padding:10px;">Copies full array every syscall</td>
              <td style="padding:10px;"><strong style="color:#10B981;">Zero</strong> (State maintained in kernel RB-Tree)</td>
            </tr>
          </tbody>
        </table>

        <h2 id="sec-27-2">27.2 Linux epoll Kernel Architecture</h2>
        <p>
          When you invoke <code>epoll_create1(EPOLL_CLOEXEC)</code>, the Linux kernel allocates an internal <code>struct eventpoll</code> holding two central data structures:
        </p>
        <ol>
          <li><strong>Interest List (Red-Black Tree - <code>rbr</code>):</strong> Stores all monitored file descriptors keyed by FD integer. Allows <code>O(log N)</code> insertion, modification, and deletion via <code>epoll_ctl()</code>.</li>
          <li><strong>Ready List (Doubly-Linked List - <code>rdllist</code>):</strong> Contains only those file descriptors that currently possess active I/O events. When incoming Ethernet packets arrive, the network driver's hardware ISR fires, and the kernel network stack places the socket onto the ready list. <code>epoll_wait()</code> simply extracts entries from this list in <code>O(1)</code> time.</li>
        </ol>

        <h2 id="sec-27-3">27.3 Edge-Triggered (EPOLLET) vs Level-Triggered (EPOLLLT)</h2>
        <p>
          Understanding the difference between Level-Triggered (the default) and Edge-Triggered mode is mandatory for writing bug-free epoll servers:
        </p>
        <ul>
          <li><strong>Level-Triggered (EPOLLLT):</strong> <code>epoll_wait()</code> continuously returns the file descriptor as long as unread data remains in the kernel socket buffer. If you read only 10 bytes out of 100 bytes, the next call to <code>epoll_wait()</code> immediately wakes up again.</li>
          <li><strong>Edge-Triggered (EPOLLET):</strong> <code>epoll_wait()</code> returns the file descriptor <em>only when a state transition occurs</em> (e.g. from empty buffer to having data). If you do not drain the socket completely, epoll will <strong>never notify you again</strong> for the remaining data, causing permanent connection lockup!</li>
        </ul>

        <div class="warning-box" style="border-left:4px solid #EF4444;background:rgba(239,68,68,0.08);padding:1rem;margin:1rem 0;border-radius:4px;">
          <strong style="color:#EF4444;">Golden Rule of Edge-Triggered epoll:</strong> When using <code>EPOLLET</code>, sockets <strong>must be non-blocking (<code>O_NONBLOCK</code>)</strong>, and reads must be wrapped in a <code>while</code> loop that drains data continuously until <code>read()</code> returns <code>-1</code> with <code>errno == EAGAIN</code> or <code>errno == EWOULDBLOCK</code>!
        </div>

        <h2 id="sec-27-4">27.4 Industrial Case Study: The Edge-Triggered Telemetry Stall</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Smart Factory Gateway Deadlock</h4>
          <p>
            An industrial IoT edge gateway connected to 2,000 manufacturing sensors started dropping sensor packets during high-load manufacturing shifts. Connections remained open, but no new telemetry was ingested.
          </p>
          <p>
            <strong>Root Cause:</strong> The developer configured epoll with <code>EPOLLET</code> (Edge-Triggered) for maximum performance, but read data using a single <code>read()</code> call with a 1,024-byte buffer:
          </p>
          <pre><code class="language-c">/* Defective Edge-Triggered Handler */
if (events[i].events &amp; EPOLLIN) {
    ssize_t bytes = read(events[i].data.fd, buffer, sizeof(buffer));
    process_telemetry(buffer, bytes);
    /* BUG: If the sensor transmitted 1,500 bytes, 476 bytes remained in kernel buffer!
       Because epoll is Edge-Triggered, it never re-notified for the remaining 476 bytes!
       The socket stalled forever awaiting another edge transition! */
}</code></pre>
          <p>
            <strong>The Fix:</strong> The read handler was rewritten to execute inside an <code>EAGAIN</code> drain loop, ensuring all buffered bytes were consumed before returning to <code>epoll_wait()</code>.
          </p>
        </div>

        <h2 id="sec-27-5">27.5 Hands-On Lab: Scalable Non-Blocking epoll TCP Echo Server</h2>
        <p>
          In this lab, you will construct a production-ready, non-blocking TCP server in C using <code>epoll_create1()</code>, <code>epoll_ctl()</code>, and <code>epoll_wait()</code> capable of serving thousands of simultaneous clients.
        </p>

        <h3>Complete Compilable Server (<code>epoll_server.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;string.h&gt;
#include &lt;unistd.h&gt;
#include &lt;fcntl.h&gt;
#include &lt;errno.h&gt;
#include &lt;sys/epoll.h&gt;
#include &lt;netinet/in.h&gt;
#include &lt;arpa/inet.h&gt;

#define MAX_EVENTS 64
#define BUFFER_SIZE 512
#define PORT 8080

static int make_socket_non_blocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int main(void) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &amp;opt, sizeof(opt));

    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    bind(server_fd, (struct sockaddr *)&amp;address, sizeof(address));
    listen(server_fd, SOMAXCONN);
    make_socket_non_blocking(server_fd);

    int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    struct epoll_event ev, events[MAX_EVENTS];

    ev.events = EPOLLIN;
    ev.data.fd = server_fd;
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &amp;ev);

    printf("[SERVER] Listening on port %d with epoll...\n", PORT);

    while (1) {
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        for (int i = 0; i &lt; nfds; i++) {
            if (events[i].data.fd == server_fd) {
                /* New incoming connection */
                struct sockaddr_in client_addr;
                socklen_t client_len = sizeof(client_addr);
                int client_fd = accept(server_fd, (struct sockaddr *)&amp;client_addr, &amp;client_len);
                make_socket_non_blocking(client_fd);

                ev.events = EPOLLIN | EPOLLET; /* Edge-Triggered */
                ev.data.fd = client_fd;
                epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &amp;ev);
                printf("[ACCEPT] Client connected: FD %d\n", client_fd);
            } else {
                /* Data available from client (Drain until EAGAIN) */
                int fd = events[i].data.fd;
                char buf[BUFFER_SIZE];
                while (1) {
                    ssize_t count = read(fd, buf, sizeof(buf));
                    if (count &gt; 0) {
                        write(fd, buf, (size_t)count); // Echo back
                    } else if (count == 0) {
                        /* Client disconnected */
                        printf("[CLOSE] Client disconnected: FD %d\n", fd);
                        epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
                        close(fd);
                        break;
                    } else {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            /* Fully drained! */
                            break;
                        }
                        perror("read error");
                        epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, NULL);
                        close(fd);
                        break;
                    }
                }
            }
        }
    }
    close(server_fd);
    return 0;
}</code></pre>

        <h2 id="sec-27-quiz">27.6 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Why does <code>epoll_wait()</code> achieve O(1) performance while <code>poll()</code> degrades at O(N)?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) epoll executes on GPU compute shaders.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) epoll inspects a dedicated ready list maintained by device driver interrupt callbacks, returning only active descriptors without scanning inactive descriptors. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) poll copies memory using DMA.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>poll()</code> must iterate through the entire array of file descriptors to check for readiness; <code>epoll</code> returns only descriptors that have already transitioned onto the kernel ready list.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What error code from <code>read()</code> signifies that a non-blocking socket has been completely drained under Edge-Triggered (EPOLLET) mode?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) <code>EINTR</code></label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) <code>EAGAIN</code> or <code>EWOULDBLOCK</code> (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) <code>ECONNRESET</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> In non-blocking sockets, <code>EAGAIN</code> / <code>EWOULDBLOCK</code> indicates that no more data is currently available in the kernel buffer and subsequent reads would block.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What internal kernel data structure does epoll use to store the registered interest list of file descriptors?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) Hash table</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) Red-Black Tree (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) FIFO circular queue</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The interest list is stored in a Red-Black Tree, enabling efficient <code>O(log N)</code> additions, removals, and modifications of monitored file descriptors.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What flag ensures that epoll automatically disables a file descriptor after delivering a single notification event?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) <code>EPOLLONESHOT</code> (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) <code>EPOLLWAKEUP</code></label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) <code>EPOLLEXCLUSIVE</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>EPOLLONESHOT</code> tells epoll to disable event reporting for that descriptor until re-armed with <code>epoll_ctl(..., EPOLL_CTL_MOD, ...)</code>, preventing multithreaded race conditions on the same socket.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: When should <code>epoll_create1(EPOLL_CLOEXEC)</code> be preferred over legacy <code>epoll_create()</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) Always; it atomically sets the close-on-exec flag on the epoll file descriptor, preventing descriptor leakage across child process spawns. (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) Only when running on ARM architectures.</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) Only when using UDP sockets.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>epoll_create1(EPOLL_CLOEXEC)</code> avoids the race condition between creation and setting <code>FD_CLOEXEC</code> via <code>fcntl</code> in multithreaded programs.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 28: Process Lifecycle & IPC
# -------------------------------------------------------------
MOD28_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 6 • Linux System Programming</span>
          <h1 class="chapter-h1">Module 28: Process Lifecycle &amp; Inter-Process Communication (IPC)</h1>
          <p class="chapter-lead">Deep dive into Linux process creation, fork() Copy-on-Write (COW) memory mechanics, execve(), zombie process reaping with waitpid(), daemonization (setsid), anonymous pipes, POSIX shared memory (shm_open, mmap), and POSIX message queues.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 60m</div>
            <div class="meta-item">🎯 Standard: POSIX.1-2008 / System V &amp; POSIX IPC</div>
            <div class="meta-item">⚡ Target: Multi-Process Linux Architectures</div>
          </div>
        </div>

        <h2 id="sec-28-1">28.1 Process Creation: fork() & Copy-on-Write (COW)</h2>
        <p>
          When a process calls <code>fork()</code>, the Linux kernel does not make an expensive duplicate copy of the parent process's physical RAM. Instead, it duplicates the parent's <strong>page tables</strong> and marks all physical memory pages as <strong>Read-Only</strong>.
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Copy-on-Write (COW) Page Fault Architecture</div>
          <pre style="margin:0;color:#94A3B8;">
Parent Page Table ──┐
                    ├──▶ [ Shared Physical Page (Read-Only) ]
Child Page Table  ──┘               │
                                    │ (Child or Parent attempts memory WRITE)
                                    ▼
                         [ MMU Write Fault (Page Fault) ]
                                    │
                                    ▼
       [ Kernel Allocates NEW Physical Page &amp; Copies 4 KB Data ]
Parent Page Table ──▶ [ Original Page (Read/Write) ]
Child Page Table  ──▶ [ Private Copy Page (Read/Write) ]
          </pre>
        </div>

        <h3>Zombie Processes & Reaping Mechanics</h3>
        <p>
          When a child process terminates via <code>exit()</code>, its virtual memory, open files, and address space are immediately deallocated. However, its entry in the kernel process table remains (a <strong>Zombie Process</strong>, state <code>Z</code>) so the parent can inspect its exit status code.
        </p>
        <p>
          If the parent fails to call <code>wait()</code> or <code>waitpid()</code>, zombies accumulate, eventually exhausting the kernel PID space (<code>/proc/sys/kernel/pid_max</code>), preventing any new processes from launching!
        </p>

        <h2 id="sec-28-2">28.2 Anonymous Pipes & Broken Pipe (SIGPIPE)</h2>
        <p>
          Pipes are unidirectional byte streams buffered in kernel memory (typically 64 KB). A pipe possesses a read descriptor (<code>pipefd[0]</code>) and a write descriptor (<code>pipefd[1]</code>).
        </p>
        <div class="warning-box" style="border-left:4px solid #EF4444;background:rgba(239,68,68,0.08);padding:1rem;margin:1rem 0;border-radius:4px;">
          <strong style="color:#EF4444;">The Fatal SIGPIPE Default Action:</strong> If a process writes to a pipe whose read end has been closed by the peer, the kernel delivers a <code>SIGPIPE</code> signal. The default disposition of <code>SIGPIPE</code> is <strong>abnormal process termination!</strong> Production daemons must set <code>signal(SIGPIPE, SIG_IGN)</code> so that <code>write()</code> returns <code>-1</code> with <code>errno == EPIPE</code> instead of crashing.
        </div>

        <h2 id="sec-28-3">28.3 POSIX Shared Memory (shm_open & mmap)</h2>
        <p>
          Shared memory is the fastest possible IPC mechanism because data does not pass through the kernel. Once mapped into the virtual address spaces of both processes via <code>mmap()</code>, reads and writes occur directly at RAM hardware speeds (zero system call overhead).
        </p>

        <h3>Lifecycle of POSIX Shared Memory</h3>
        <ol>
          <li><code>shm_open("/telemetry_shm", O_CREAT | O_RDWR, 0660)</code>: Creates a named shared memory object in <code>/dev/shm</code>.</li>
          <li><code>ftruncate(shm_fd, sizeof(telemetry_data_t))</code>: Sizes the physical allocation.</li>
          <li><code>mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0)</code>: Maps memory into process address space.</li>
          <li><code>shm_unlink("/telemetry_shm")</code>: Removes the name from the filesystem namespace (memory persists until all processes unmap).</li>
        </ol>

        <h2 id="sec-28-4">28.4 Industrial Case Study: Robot Arm Supervisor Crash</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The SIGPIPE Factory Floor Emergency Shutdown</h4>
          <p>
            An industrial articulated robotic arm performing automotive welding suddenly stopped mid-weld and collapsed into emergency limp mode.
          </p>
          <p>
            <strong>Investigation:</strong> The robot’s trajectory controller ran as a multi-process architecture. The motion supervisor communicated with a GUI diagnostic display over a local UNIX domain socket pipe.
          </p>
          <p>
            The GUI tablet battery died, closing the receiving socket. 10 milliseconds later, the motion supervisor attempted to write a telemetry frame:
          </p>
          <pre><code class="language-c">write(gui_socket_fd, &amp;coords, sizeof(coords)); /* Unshielded write! */</code></pre>
          <p>
            Because the receiving socket was closed and the application had not ignored <code>SIGPIPE</code>, the Linux kernel terminated the main motion supervisor immediately.
          </p>
          <p>
            <strong>Corrective Action:</strong> Enforced <code>signal(SIGPIPE, SIG_IGN)</code> across all daemons, and configured socket writes to use <code>send(..., MSG_NOSIGNAL)</code>.
          </p>
        </div>

        <h2 id="sec-28-5">28.5 Hands-On Lab: Zero-Copy Shared Memory IPC with Process-Shared Mutex</h2>
        <p>
          In this lab, you will create two independent processes communicating over POSIX Shared Memory synchronized by a process-shared pthread mutex.
        </p>

        <h3>Shared Memory IPC Engine (<code>shm_ipc.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;unistd.h&gt;
#include &lt;fcntl.h&gt;
#include &lt;sys/mman.h&gt;
#include &lt;sys/wait.h&gt;
#include &lt;pthread.h&gt;

typedef struct {
    pthread_mutex_t mutex;
    uint32_t counter;
} shared_block_t;

int main(void) {
    const char *shm_name = "/acl_demo_shm";
    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR | O_TRUNC, 0666);
    ftruncate(shm_fd, sizeof(shared_block_t));

    shared_block_t *block = (shared_block_t *)mmap(
        NULL, sizeof(shared_block_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    /* Initialize process-shared mutex */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&amp;attr);
    pthread_mutexattr_setpshared(&amp;attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&amp;block-&gt;mutex, &amp;attr);
    block-&gt;counter = 0;

    pid_t pid = fork();
    if (pid == 0) {
        /* Child Process: Producer */
        for (int i = 0; i &lt; 5; i++) {
            pthread_mutex_lock(&amp;block-&gt;mutex);
            block-&gt;counter += 10;
            printf("[CHILD] Incremented counter to: %u\n", block-&gt;counter);
            pthread_mutex_unlock(&amp;block-&gt;mutex);
            usleep(100000);
        }
        exit(0);
    } else {
        /* Parent Process: Consumer */
        for (int i = 0; i &lt; 5; i++) {
            pthread_mutex_lock(&amp;block-&gt;mutex);
            printf("[PARENT] Read counter value: %u\n", block-&gt;counter);
            pthread_mutex_unlock(&amp;block-&gt;mutex);
            usleep(100000);
        }
        wait(NULL); // Reap child
    }

    munmap(block, sizeof(shared_block_t));
    shm_unlink(shm_name);
    return 0;
}</code></pre>

        <h2 id="sec-28-quiz">28.6 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What physical event occurs during <code>fork()</code> under Copy-on-Write (COW)?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) All physical RAM pages are duplicated immediately.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) Page table entries are duplicated and marked read-only; physical pages are only copied when a process attempts to write to them. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) The child process runs inside swap space.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Copy-on-Write optimizes process creation by sharing existing physical memory pages until a write triggers a hardware page fault that creates a private copy.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What signal is delivered by default when writing to a pipe with no active readers, and what is its default action?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) <code>SIGUSR1</code> (Ignored)</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) <code>SIGPIPE</code> (Abnormal termination of the process) (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) <code>SIGSEGV</code> (Core dump)</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>SIGPIPE</code> is delivered when writing to a broken pipe or disconnected socket; because its default action is termination, production systems must ignore it or handle it.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What mutex attribute must be enabled to allow a pthread mutex in POSIX shared memory to synchronize two distinct processes?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) <code>PTHREAD_MUTEX_RECURSIVE</code></label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) <code>PTHREAD_PROCESS_SHARED</code> (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) <code>PTHREAD_INHERIT_SCHED</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> By default, pthread mutexes are <code>PTHREAD_PROCESS_PRIVATE</code>; setting <code>PTHREAD_PROCESS_SHARED</code> instructs the kernel futex subsystem to permit multi-process synchronization.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What filesystem directory mounts the POSIX shared memory objects created via <code>shm_open()</code> on Linux?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) <code>/tmp</code></label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) <code>/dev/shm</code> (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) <code>/var/run</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> On Linux, POSIX shared memory is backed by a tmpfs RAM filesystem mounted at <code>/dev/shm</code>.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: How can a parent process clean up finished child processes without blocking if no children have exited yet?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>wait(NULL)</code></label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>waitpid(-1, &amp;status, WNOHANG)</code> (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>sleep(1)</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The <code>WNOHANG</code> option instructs <code>waitpid()</code> to return immediately with 0 if no child has exited, preventing the parent from stalling.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 29: POSIX Threads & Multithreading
# -------------------------------------------------------------
MOD29_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 6 • Linux System Programming</span>
          <h1 class="chapter-h1">Module 29: POSIX Threads &amp; Multithreading</h1>
          <p class="chapter-lead">Deep dive into Linux Native POSIX Thread Library (NPTL) internals, clone() system calls, thread creation lifecycle, custom stack sizing and MPU guard pages, thread cancellation, thread-local storage (_Thread_local), and safe worker thread pools.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 55m</div>
            <div class="meta-item">🎯 Standard: POSIX.1c (pthreads) / ISO C11 / Linux NPTL</div>
            <div class="meta-item">⚡ Target: Multithreaded Linux &amp; RT-Preempt</div>
          </div>
        </div>

        <h2 id="sec-29-1">29.1 Linux NPTL & The clone() Syscall</h2>
        <p>
          In the Linux kernel, there is strictly no difference in internal representation between a process and a thread: both are represented by a <code>struct task_struct</code>.
        </p>
        <p>
          A thread is created via the <strong>clone(2)</strong> system call passing flags that instruct the child task to share the parent's memory, file descriptors, and signal dispositions:
        </p>
        <pre><code class="language-c">clone(child_func, stack_top,
      CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD,
      arg);</code></pre>
        <ul>
          <li><code>CLONE_VM</code>: Both tasks share the same virtual memory page tables.</li>
          <li><code>CLONE_FILES</code>: Both tasks share the same file descriptor table.</li>
          <li><code>CLONE_THREAD</code>: Placed into the same Thread Group (TGID matches PID).</li>
        </ul>

        <h2 id="sec-29-2">29.2 Thread Stacks & Guard Pages</h2>
        <p>
          While threads share the heap and data segments, <strong>every thread must have its own independent stack</strong> for function calls and local variables.
        </p>
        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Thread Stack Memory Layout &amp; Guard Page Protection</div>
          <pre style="margin:0;color:#94A3B8;">
High Memory (0x7FFF...)
    │ ┌────────────────────────────────────────────────────────┐
    │ │ Thread Stack Top (Stack Pointer %rsp starts here)      │
    │ │ Local variables, return addresses, AAPCS stack frames   │
    │ │               │ (Stack grows downward)                 │
    │ │               ▼                                        │
    │ ├────────────────────────────────────────────────────────┤
    │ │ Guard Page (4 KB, mprotect(PROT_NONE))                 │ &lt;-- Stack Overflow Traps Here!
    ▼ └────────────────────────────────────────────────────────┘
Low Memory
          </pre>
        </div>
        <p>
          On desktop Linux, the default thread stack is <strong>8 MB</strong>. In memory-constrained embedded Linux gateways (e.g. 64 MB total RAM), spawning 20 threads with default stacks would consume 160 MB of virtual address space!
        </p>
        <h3>Configuring Custom Stack Sizes</h3>
        <pre><code class="language-c">pthread_attr_t attr;
pthread_attr_init(&amp;attr);
/* Restrict stack size to 128 KB for embedded tasks */
pthread_attr_setstacksize(&amp;attr, 128 * 1024);
pthread_create(&amp;thread_id, &amp;attr, worker_task, NULL);
pthread_attr_destroy(&amp;attr);</code></pre>

        <h2 id="sec-29-3">29.3 Thread Cancellation & Cleanup Handlers</h2>
        <p>
          Terminating a thread from the outside using <code>pthread_cancel()</code> is dangerous: if the thread holds an open mutex or malloc buffer, terminating it asynchronously causes resource leaks or deadlocks.
        </p>
        <p>
          <strong>Deferred Cancellation:</strong> By default, cancellation occurs only when the thread reaches a defined POSIX <em>cancellation point</em> (such as <code>read()</code>, <code>sleep()</code>, or <code>pthread_testcancel()</code>).
        </p>
        <p>
          To ensure resources are cleanly released when cancelled, register cleanup handlers:
        </p>
        <pre><code class="language-c">void cleanup_handler(void *arg) {
    int *fd = (int *)arg;
    if (fd &amp;&amp; *fd &gt;= 0) close(*fd);
}

void *worker(void *arg) {
    int fd = open("log.txt", O_WRONLY);
    pthread_cleanup_push(cleanup_handler, &amp;fd);

    while (1) {
        pthread_testcancel(); /* Explicit cancellation checkpoint */
        usleep(10000);
    }

    pthread_cleanup_pop(1); /* Executes cleanup if cancelled or returned */
    return NULL;
}</code></pre>

        <h2 id="sec-29-4">29.4 Industrial Case Study: Drone Embedded Gateway Stack Exhaustion</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The 64 MB RAM Out-Of-Memory Reboot</h4>
          <p>
            An autonomous agricultural survey drone running embedded Linux crashed and rebooted every time it engaged all 12 multi-spectral camera channels.
          </p>
          <p>
            <strong>Investigation:</strong> The video capture software spawned a dedicated thread for each camera channel. The developers used default <code>pthread_create()</code> attributes.
          </p>
          <p>
            With 12 camera threads + 6 networking threads + 4 sensor fusion threads = 22 threads. 22 threads &times; 8 MB default stack = <strong>176 MB of virtual memory</strong> requested on a drone board with only 64 MB of physical RAM! The Linux OOM (Out-Of-Memory) killer was triggered, terminating the flight navigation service.
          </p>
          <p>
            <strong>Resolution:</strong> Explicit stack sizes were configured: 256 KB for camera ingestion, 64 KB for sensor telemetry. Total stack footprint dropped from 176 MB to 3.5 MB, resolving the issue completely.
          </p>
        </div>

        <h2 id="sec-29-5">29.5 Hands-On Lab: Implementing a Worker Thread Pool in C</h2>
        <p>
          In this lab, you will write a complete multi-threaded worker pool in C that distributes computational tasks across worker threads using a task queue.
        </p>

        <h3>Worker Thread Pool (<code>thread_pool.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;pthread.h&gt;
#include &lt;stdbool.h&gt;
#include &lt;unistd.h&gt;

#define POOL_SIZE 4
#define QUEUE_MAX 16

typedef void (*task_fn)(void *arg);

typedef struct {
    task_fn function;
    void *argument;
} task_t;

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t notify;
    pthread_t threads[POOL_SIZE];
    task_t queue[QUEUE_MAX];
    int head;
    int tail;
    int count;
    bool shutdown;
} thread_pool_t;

static void *worker_thread(void *arg) {
    thread_pool_t *pool = (thread_pool_t *)arg;
    while (1) {
        pthread_mutex_lock(&amp;pool-&gt;lock);
        while (pool-&gt;count == 0 &amp;&amp; !pool-&gt;shutdown) {
            pthread_cond_wait(&amp;pool-&gt;notify, &amp;pool-&gt;lock);
        }
        if (pool-&gt;shutdown &amp;&amp; pool-&gt;count == 0) {
            pthread_mutex_unlock(&amp;pool-&gt;lock);
            break;
        }

        task_t task = pool-&gt;queue[pool-&gt;tail];
        pool-&gt;tail = (pool-&gt;tail + 1) % QUEUE_MAX;
        pool-&gt;count--;
        pthread_mutex_unlock(&amp;pool-&gt;lock);

        /* Execute task */
        task.function(task.argument);
    }
    return NULL;
}

void print_task(void *arg) {
    int id = (int)(intptr_t)arg;
    printf("[THREAD %lu] Executed Task ID #%d\n", pthread_self(), id);
    usleep(50000);
}

int main(void) {
    thread_pool_t pool = {0};
    pthread_mutex_init(&amp;pool.lock, NULL);
    pthread_cond_init(&amp;pool.notify, NULL);

    pthread_attr_t attr;
    pthread_attr_init(&amp;attr);
    pthread_attr_setstacksize(&amp;attr, 128 * 1024); /* 128 KB custom stack */

    for (int i = 0; i &lt; POOL_SIZE; i++) {
        pthread_create(&amp;pool.threads[i], &amp;attr, worker_thread, &amp;pool);
    }
    pthread_attr_destroy(&amp;attr);

    /* Enqueue 8 tasks */
    pthread_mutex_lock(&amp;pool.lock);
    for (intptr_t i = 1; i &lt;= 8; i++) {
        pool.queue[pool.head] = (task_t){print_task, (void *)i};
        pool.head = (pool.head + 1) % QUEUE_MAX;
        pool.count++;
    }
    pthread_cond_broadcast(&amp;pool.notify);
    pthread_mutex_unlock(&amp;pool.lock);

    sleep(1);

    /* Clean shutdown */
    pthread_mutex_lock(&amp;pool.lock);
    pool.shutdown = true;
    pthread_cond_broadcast(&amp;pool.notify);
    pthread_mutex_unlock(&amp;pool.lock);

    for (int i = 0; i &lt; POOL_SIZE; i++) {
        pthread_join(pool.threads[i], NULL);
    }
    printf("Thread pool shut down cleanly.\n");
    return 0;
}</code></pre>

        <h2 id="sec-29-quiz">29.6 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What underlying Linux system call creates POSIX threads in the Native POSIX Thread Library (NPTL)?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) <code>fork()</code></label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) <code>clone()</code> (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) <code>vfork()</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>pthread_create()</code> invokes <code>clone()</code> passing flags that share virtual memory, file descriptors, and signal handlers with the calling process.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the typical default thread stack size allocated on 64-bit Linux distributions?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) 64 KB</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) 8 MB (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) 1 GB</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> The standard Linux pthreads default stack is 8 megabytes, which must be configured lower in embedded systems with limited RAM.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What ISO C11 keyword defines a variable that is instantiated uniquely per thread?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) <code>volatile</code></label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) <code>_Thread_local</code> (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) <code>_Atomic</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>_Thread_local</code> (or <code>thread_local</code> in <code>&lt;threads.h&gt;</code>) gives the variable thread-local storage duration.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What happens if a detached thread (<code>pthread_detach()</code>) terminates?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) Its thread ID remains in a zombie state until <code>pthread_join()</code> is called.</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) Its resources and stack memory are reclaimed immediately and automatically by the system. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) The entire parent process crashes.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Detached threads automatically clean up their own resources upon termination without requiring another thread to join them.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: Why should thread cancellation cleanup handlers be registered via <code>pthread_cleanup_push()</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) To ensure that acquired mutexes and open file handles are unlocked and freed if the thread is cancelled midway. (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) To make threads run at high priority.</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) To restart the thread automatically.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Cleanup handlers act like <code>finally</code> blocks in C, guaranteeing that held locks or allocated memory are not leaked if cancellation occurs.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 30: Thread Synchronization & Locks
# -------------------------------------------------------------
MOD30_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 6 • Linux System Programming</span>
          <h1 class="chapter-h1">Module 30: Thread Synchronization &amp; Locks</h1>
          <p class="chapter-lead">Deep dive into Linux synchronization primitives: futex(2) mechanics, mutex types (normal, recursive, error-checking), condition variables and the lost wakeup problem, read-write locks (pthread_rwlock_t), deadlock prevention hierarchies, and priority inheritance (PTHREAD_PRIO_INHERIT).</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 60m</div>
            <div class="meta-item">🎯 Standard: POSIX.1c / Linux futex(2) / ISO 26262</div>
            <div class="meta-item">⚡ Target: High-Reliability Concurrent Systems</div>
          </div>
        </div>

        <h2 id="sec-30-1">30.1 Fast Userspace Mutexes: Linux futex(2) Internals</h2>
        <p>
          In early Unix kernels, every mutex lock/unlock required an expensive system call into the kernel (costing hundreds of CPU cycles for context switching).
        </p>
        <p>
          Modern Linux synchronization relies on <strong>futex (Fast Userspace Mutex)</strong>:
        </p>
        <ul>
          <li><strong>Uncontended Case (99% of time):</strong> An atomic Compare-And-Swap (CAS) instruction updates a 32-bit integer in user space memory. <em>Zero system calls! Execution takes ~10 nanoseconds!</em></li>
          <li><strong>Contended Case (Collision):</strong> Only when a thread discovers the lock is already held does it invoke <code>futex(FUTEX_WAIT)</code> to put itself to sleep in a kernel wait-queue. Upon release, the holder calls <code>futex(FUTEX_WAKE)</code>.</li>
        </ul>

        <h2 id="sec-30-2">30.2 Condition Variables & The Lost Wakeup Bug</h2>
        <p>
          A condition variable (<code>pthread_cond_t</code>) allows a thread to sleep until a specific application condition becomes true.
        </p>
        <div class="warning-box" style="border-left:4px solid #EF4444;background:rgba(239,68,68,0.08);padding:1rem;margin:1rem 0;border-radius:4px;">
          <strong style="color:#EF4444;">The Spurious Wakeup Hazard:</strong> In POSIX, <code>pthread_cond_wait()</code> may wake up even if NO thread signaled the condition (due to kernel signal interrupts or multi-processor races). Therefore, you must <strong>NEVER use an <code>if</code> statement</strong> with condition variables! Always use a <strong><code>while</code> loop</strong>!
        </div>

        <pre><code class="language-c">/* FATAL BUG: Using 'if' can lead to processing empty queue */
if (queue_is_empty()) {
    pthread_cond_wait(&amp;cond, &amp;mutex); /* May wake up spuriously! */
}
item = dequeue(); /* CRASH! Queue might still be empty! */

/* PRODUCTION PATTERN: Mandatory 'while' loop */
while (queue_is_empty()) {
    pthread_cond_wait(&amp;cond, &amp;mutex); /* Re-checks condition upon waking! */
}
item = dequeue(); /* Guaranteed safe! */</code></pre>

        <h2 id="sec-30-3">30.3 Deadlock Prevention & Coffman's Conditions</h2>
        <p>
          A deadlock occurs when two or more threads are permanently blocked waiting for resources held by each other. A deadlock requires four simultaneous conditions (Coffman Conditions):
        </p>
        <ol>
          <li><strong>Mutual Exclusion:</strong> Exclusive resource locking.</li>
          <li><strong>Hold and Wait:</strong> Holding lock A while requesting lock B.</li>
          <li><strong>No Preemption:</strong> Locks cannot be forcibly confiscated.</li>
          <li><strong>Circular Wait:</strong> Thread 1 holds A and wants B; Thread 2 holds B and wants A.</li>
        </ol>

        <p>
          <strong>The Universal Solution: Strict Lock Ordering Hierarchy.</strong> Assign every mutex in your system a strict integer rank. Never acquire a lock with rank <code>N</code> while already holding a lock with rank <code>&ge; N</code>.
        </p>

        <h2 id="sec-30-4">30.4 Priority Inversion & Priority Inheritance Mutexes</h2>
        <p>
          <strong>Priority Inversion</strong> occurs when a low-priority thread holds a lock needed by a high-priority real-time thread, but a medium-priority thread preempts the low-priority thread, effectively starving the high-priority thread!
        </p>
        <p>
          <strong>The Solution: Priority Inheritance (<code>PTHREAD_PRIO_INHERIT</code>)</strong>. When a high-priority thread blocks on a lock held by a low-priority thread, the kernel dynamically boosts the low-priority thread's priority to match the blocked thread until the lock is released.
        </p>
        <pre><code class="language-c">pthread_mutexattr_t attr;
pthread_mutexattr_init(&amp;attr);
pthread_mutexattr_setprotocol(&amp;attr, PTHREAD_PRIO_INHERIT);
pthread_mutex_init(&amp;rt_mutex, &amp;attr);</code></pre>

        <h2 id="sec-30-5">30.5 Industrial Case Study: The Mars Pathfinder Priority Inversion</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The 1997 Mars Rover Watchdog Reset Incident</h4>
          <p>
            Days after landing on Mars, the Mars Pathfinder spacecraft began experiencing repeated total system resets. The watchdog timer triggered an emergency reboot, threatening the mission.
          </p>
          <p>
            <strong>Root Cause:</strong>
            Three tasks competed on the real-time kernel:
          </p>
          <ul>
            <li><strong>High-Priority Task:</strong> Attitude control &amp; bus communication (ran periodically).</li>
            <li><strong>Medium-Priority Task:</strong> Long-running communications task (no shared lock).</li>
            <li><strong>Low-Priority Task:</strong> Meteorological data gathering (infrequently acquired shared Information Bus mutex).</li>
          </ul>
          <p>
            The low-priority task acquired the bus mutex. The high-priority task woke up, requested the bus mutex, and blocked. Before the low-priority task could release the lock, the medium-priority task woke up and preempted it!
          </p>
          <p>
            The high-priority task was starved of CPU time for seconds. The watchdog timer assumed the system was hung and executed a hard reset!
          </p>
          <p>
            <strong>The Fix:</strong> JPL engineers remotely uploaded a C patch enabling <strong>Priority Inheritance</strong> on the Information Bus mutex, permanently resolving the issue.
          </p>
        </div>

        <h2 id="sec-30-6">30.6 Hands-On Lab: Bounded Thread-Safe Blocking Queue in C</h2>
        <p>
          In this lab, you will write a complete, bounded blocking queue in C that synchronizes producer and consumer threads using a mutex and two condition variables (<code>not_full</code> and <code>not_empty</code>).
        </p>

        <h3>Blocking Queue Implementation (<code>blocking_queue.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;pthread.h&gt;
#include &lt;unistd.h&gt;

#define QUEUE_CAPACITY 4

typedef struct {
    int data[QUEUE_CAPACITY];
    int head;
    int tail;
    int count;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} blocking_queue_t;

void bq_init(blocking_queue_t *q) {
    q-&gt;head = 0; q-&gt;tail = 0; q-&gt;count = 0;
    pthread_mutex_init(&amp;q-&gt;lock, NULL);
    pthread_cond_init(&amp;q-&gt;not_empty, NULL);
    pthread_cond_init(&amp;q-&gt;not_full, NULL);
}

void bq_push(blocking_queue_t *q, int value) {
    pthread_mutex_lock(&amp;q-&gt;lock);
    while (q-&gt;count == QUEUE_CAPACITY) {
        pthread_cond_wait(&amp;q-&gt;not_full, &amp;q-&gt;lock);
    }
    q-&gt;data[q-&gt;head] = value;
    q-&gt;head = (q-&gt;head + 1) % QUEUE_CAPACITY;
    q-&gt;count++;
    pthread_cond_signal(&amp;q-&gt;not_empty);
    pthread_mutex_unlock(&amp;q-&gt;lock);
}

int bq_pop(blocking_queue_t *q) {
    pthread_mutex_lock(&amp;q-&gt;lock);
    while (q-&gt;count == 0) {
        pthread_cond_wait(&amp;q-&gt;not_empty, &amp;q-&gt;lock);
    }
    int val = q-&gt;data[q-&gt;tail];
    q-&gt;tail = (q-&gt;tail + 1) % QUEUE_CAPACITY;
    q-&gt;count--;
    pthread_cond_signal(&amp;q-&gt;not_full);
    pthread_mutex_unlock(&amp;q-&gt;lock);
    return val;
}

static blocking_queue_t g_q;

void *producer(void *arg) {
    for (int i = 1; i &lt;= 10; i++) {
        bq_push(&amp;g_q, i * 10);
        printf("[PRODUCER] Pushed: %d\n", i * 10);
        usleep(50000);
    }
    return NULL;
}

void *consumer(void *arg) {
    for (int i = 1; i &lt;= 10; i++) {
        int v = bq_pop(&amp;g_q);
        printf("  [CONSUMER] Popped: %d\n", v);
        usleep(100000);
    }
    return NULL;
}

int main(void) {
    bq_init(&amp;g_q);
    pthread_t p, c;
    pthread_create(&amp;p, NULL, producer, NULL);
    pthread_create(&amp;c, NULL, consumer, NULL);
    pthread_join(p, NULL);
    pthread_join(c, NULL);
    printf("Bounded blocking queue test finished successfully!\n");
    return 0;
}</code></pre>

        <h2 id="sec-30-quiz">27.7 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Why must <code>pthread_cond_wait()</code> always be enclosed within a <code>while</code> loop instead of an <code>if</code> statement?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) To handle spurious wakeups where the thread unblocks without a signal or before the condition is satisfied. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) <code>if</code> statements are illegal in POSIX C.</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) It prevents stack overflows.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Spurious wakeups occur due to signal delivery or kernel scheduling; re-evaluating the boolean condition inside a <code>while</code> loop guarantees safety before proceeding.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the primary performance benefit of a futex-based mutex over a traditional kernel semaphore in Linux?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) In the uncontended case, acquiring the lock is a fast user-space atomic instruction with zero system call overhead. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) It eliminates the need for RAM.</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) It automatically restarts crashed threads.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Futexes avoid entering the kernel unless there is genuine contention between multiple threads, speeding up uncontended execution by orders of magnitude.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: How does Priority Inheritance resolve the priority inversion phenomenon?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) It lowers the priority of all threads to 0.</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) It temporarily boosts the priority of the lock holder to match that of the highest-priority blocked thread. (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) It kills medium-priority threads.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Temporarily raising the holder's priority prevents intermediate medium-priority threads from preempting it, allowing it to release the lock rapidly.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What happens if a thread attempts to lock a normal <code>PTHREAD_MUTEX_NORMAL</code> mutex that it already holds?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) It returns <code>EINVAL</code>.</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) Self-deadlock: the thread blocks forever waiting for itself to release the lock. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) The lock counter increments to 2.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Normal mutexes do not detect re-locking by the same thread; self-deadlock occurs unless configured as <code>PTHREAD_MUTEX_RECURSIVE</code> or <code>ERRORCHECK</code>.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What is the primary advantage of a Read-Write Lock (<code>pthread_rwlock_t</code>) over a standard Mutex?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) It permits multiple concurrent reader threads while still guaranteeing exclusive access to writers. (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) It requires no memory allocations.</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) It runs inside interrupts.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> RW-locks maximize read scalability for data structures that are frequently read but infrequently modified.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 31: POSIX Signals & Real-Time Timers
# -------------------------------------------------------------
MOD31_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 6 • Linux System Programming</span>
          <h1 class="chapter-h1">Module 31: POSIX Signals &amp; Real-Time Timers</h1>
          <p class="chapter-lead">Deep dive into asynchronous signal delivery, the strict POSIX async-signal-safety contract, converting signals to events via signalfd(2), clean shutdown architectures, and POSIX high-resolution real-time timers (timer_create, CLOCK_MONOTONIC).</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 50m</div>
            <div class="meta-item">🎯 Standard: POSIX.1-2008 / Linux signalfd(2)</div>
            <div class="meta-item">⚡ Target: Production Daemons &amp; Real-Time Controllers</div>
          </div>
        </div>

        <h2 id="sec-31-1">31.1 Signal Mechanics & The Async-Signal-Safety Trap</h2>
        <p>
          A <strong>Signal</strong> is an asynchronous software interrupt delivered by the Linux kernel to a process. When a signal arrives, the kernel interrupts whatever instruction the CPU is executing, pushes a synthetic signal frame onto the stack, and jumps to the signal handler function.
        </p>

        <div class="warning-box" style="border-left:4px solid #EF4444;background:rgba(239,68,68,0.08);padding:1rem;margin:1rem 0;border-radius:4px;">
          <strong style="color:#EF4444;">The Fatal Async-Signal-Safety Rule:</strong> Inside a traditional signal handler, you are <strong>STRICTLY FORBIDDEN</strong> from calling non-reentrant functions! Functions like <code>printf()</code>, <code>malloc()</code>, <code>free()</code>, and <code>pthread_mutex_lock()</code> acquire internal locks. If a signal interrupts a thread while that thread is already holding the internal <code>malloc</code> lock, calling <code>malloc()</code> inside the signal handler triggers an <strong>immediate, unrecoverable deadlock!</strong>
        </div>

        <p>
          Only functions explicitly listed as <em>Async-Signal-Safe</em> in <code>man 7 signal-safety</code> (such as <code>write()</code>, <code>_exit()</code>, <code>sigaction()</code>) may be invoked inside signal handlers.
        </p>

        <h2 id="sec-31-2">31.2 Modern Architecture: signalfd(2) Synchronous Consumption</h2>
        <p>
          To escape the hazards of asynchronous signal handlers, modern Linux provides <strong>signalfd(2)</strong>. It converts asynchronous signals into a standard <strong>readable file descriptor</strong>!
        </p>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">signalfd Architecture: Unifying Signals into epoll</div>
          <pre style="margin:0;color:#94A3B8;">
Kernel Signals (SIGINT, SIGTERM)
          │
          ▼ Blocked by sigprocmask() (Cannot interrupt threads!)
  [ signalfd File Descriptor ] ──(Readable Event)──▶ [ Unified epoll_wait Loop ]
                                                              │
                                                  Handles shutdown cleanly
                                                  using standard printf/free!
          </pre>
        </div>

        <h3>Setting Up signalfd with epoll</h3>
        <pre><code class="language-c">sigset_t mask;
sigemptyset(&amp;mask);
sigaddset(&amp;mask, SIGINT);
sigaddset(&amp;mask, SIGTERM);

/* 1. Block signals so they do not invoke default interrupt handlers */
sigprocmask(SIG_BLOCK, &amp;mask, NULL);

/* 2. Create signal file descriptor */
int sfd = signalfd(-1, &amp;mask, SFD_NONBLOCK | SFD_CLOEXEC);

/* 3. Register sfd directly with your epoll event loop! */
struct epoll_event ev;
ev.events = EPOLLIN;
ev.data.fd = sfd;
epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sfd, &amp;ev);</code></pre>

        <h2 id="sec-31-3">31.3 POSIX Real-Time Periodic Timers</h2>
        <p>
          Rather than relying on inaccurate <code>sleep()</code> loops, safety-critical real-time applications configure high-precision kernel timers using <code>timer_create()</code> bound to <code>CLOCK_MONOTONIC</code>:
        </p>
        <pre><code class="language-c">timer_t timerid;
struct sigevent sev = {0};
sev.sigev_notify = SIGEV_SIGNAL;
sev.sigev_signo = SIGRTMIN; /* Real-time signal */

timer_create(CLOCK_MONOTONIC, &amp;sev, &amp;timerid);

/* Configure 100 Hz periodic timer (every 10 ms) */
struct itimerspec its;
its.it_interval.tv_sec = 0;
its.it_interval.tv_nsec = 10000000; // 10 ms period
its.it_value = its.it_interval;     // Initial expiration

timer_settime(timerid, 0, &amp;its, NULL);</code></pre>

        <h2 id="sec-31-4">31.4 Industrial Case Study: Flight Simulator Lockup via syslog() in Signal</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Commercial Flight Simulator Visual Engine Hang</h4>
          <p>
            A high-fidelity FAA Level D flight simulator visual projection system randomly locked up during training exercises. The simulation froze completely, requiring a hard reboot.
          </p>
          <p>
            <strong>Investigation:</strong> Core dumps showed the rendering thread was deadlocked inside <code>malloc()</code>:
          </p>
          <pre><code class="language-text">#0  __lll_lock_wait () at futex.c
#1  __GI___libc_malloc () at malloc.c:2928
#2  syslog () at syslog.c:184
#3  &lt;signal handler called&gt; (SIGALRM)
#4  __GI___libc_malloc () at malloc.c:2910
#5  render_terrain_mesh () at terrain.c:452</code></pre>
          <p>
            <strong>Root Cause:</strong> The terrain rendering thread was allocating a dynamic vertex buffer when the <code>SIGALRM</code> timer signal fired. The signal handler called <code>syslog()</code> to record periodic telemetry.
          </p>
          <p>
            Internally, <code>syslog()</code> calls <code>malloc()</code>. Because the thread already held the internal glibc heap arena lock at the moment of interruption, the second <code>malloc()</code> call attempted to acquire the same lock, resulting in an immediate self-deadlock!
          </p>
          <p>
            <strong>Resolution:</strong> The async signal handler was completely removed and replaced with <code>timerfd_create()</code> integrated into the main epoll event loop.
          </p>
        </div>

        <h2 id="sec-31-5">31.5 Hands-On Lab: Graceful Daemon Shutdown with signalfd</h2>
        <p>
          In this lab, you will write a complete Linux server that handles <code>SIGINT</code> (Ctrl+C) and <code>SIGTERM</code> cleanly using <code>signalfd()</code> and an epoll event loop.
        </p>

        <h3>Graceful Shutdown Server (<code>signalfd_daemon.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;unistd.h&gt;
#include &lt;signal.h&gt;
#include &lt;sys/signalfd.h&gt;
#include &lt;sys/epoll.h&gt;

int main(void) {
    sigset_t mask;
    sigemptyset(&amp;mask);
    sigaddset(&amp;mask, SIGINT);
    sigaddset(&amp;mask, SIGTERM);

    /* Block signals so they are delivered synchronously through signalfd */
    if (sigprocmask(SIG_BLOCK, &amp;mask, NULL) &lt; 0) {
        perror("sigprocmask");
        return 1;
    }

    int sfd = signalfd(-1, &amp;mask, SFD_NONBLOCK | SFD_CLOEXEC);
    int epfd = epoll_create1(EPOLL_CLOEXEC);

    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = sfd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, sfd, &amp;ev);

    printf("[DAEMON] Running. Press Ctrl+C (SIGINT) to trigger graceful shutdown...\n");

    int running = 1;
    while (running) {
        struct epoll_event events[4];
        int nfds = epoll_wait(epfd, events, 4, -1);
        for (int i = 0; i &lt; nfds; i++) {
            if (events[i].data.fd == sfd) {
                struct signalfd_siginfo fdsi;
                ssize_t s = read(sfd, &amp;fdsi, sizeof(fdsi));
                if (s == sizeof(fdsi)) {
                    printf("\n[SIGNAL] Caught signal %u (sender PID=%d)!\n",
                           fdsi.ssi_signo, fdsi.ssi_pid);
                    printf("[CLEANUP] Flushing file buffers and freeing resources safely...\n");
                    running = 0; // Terminate loop cleanly!
                }
            }
        }
    }

    close(sfd);
    close(epfd);
    printf("[DAEMON] Shutdown sequence completed cleanly. Exiting with code 0.\n");
    return 0;
}</code></pre>

        <h2 id="sec-31-quiz">31.6 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: Why is calling <code>printf()</code> inside an asynchronous signal handler dangerous?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) <code>printf()</code> is slow.</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) <code>printf()</code> acquires internal mutex locks on <code>stdout</code>; if interrupted while holding that lock, calling it again results in a permanent deadlock. (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) The terminal does not support signals.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>printf</code> is not async-signal-safe because it relies on non-reentrant stdio buffer locks.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What Linux system call converts incoming signals into readable events on a file descriptor?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) <code>eventfd()</code></label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) <code>signalfd()</code> (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) <code>timerfd()</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>signalfd()</code> provides a descriptor interface for consuming signals synchronously via <code>read()</code> and <code>epoll</code>.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: What clock source should be used for real-time periodic timers to prevent time jumps when NTP syncs the system clock?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) <code>CLOCK_REALTIME</code></label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) <code>CLOCK_MONOTONIC</code> (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) <code>CLOCK_PROCESS_CPUTIME_ID</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>CLOCK_MONOTONIC</code> advances strictly monotonically at constant speed and is immune to discontinuous system wall-clock adjustments.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: Before reading signals from <code>signalfd</code>, what must be done with <code>sigprocmask()</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) Unblock all signals.</label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) Block the signals so the default OS handlers do not intercept them before signalfd. (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) Set signals to SIG_DFL.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Blocking the signals prevents the kernel from executing default actions (e.g. terminating the process on SIGINT) so the signal is queued for <code>signalfd</code>.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: Which of the following data types is guaranteed by the C standard to be read and written atomically in a signal handler?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>volatile sig_atomic_t</code> (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>double</code></label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>char *</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>sig_atomic_t</code> is guaranteed by ISO C to be modified as an atomic entity even in the presence of asynchronous signals.
            </div>
          </div>
        </div>'''

# -------------------------------------------------------------
# MODULE 32: Network Sockets Programming
# -------------------------------------------------------------
MOD32_ARTICLE = '''        <div class="chapter-hero">
          <span class="chapter-module-tag">Part 6 • Linux System Programming</span>
          <h1 class="chapter-h1">Module 32: Network Sockets Programming</h1>
          <p class="chapter-lead">Master production network programming in C. Deep dive into the TCP state machine, socket options (SO_REUSEADDR, TCP_NODELAY), endianness and binary wire protocols (network byte order), framing over streaming TCP sockets, and struct packing hazards across heterogeneous platforms.</p>
          <div class="chapter-meta-bar">
            <div class="meta-item">⏱️ Duration: 60m</div>
            <div class="meta-item">🎯 Standard: POSIX Sockets / RFC 793 (TCP) / RFC 9293</div>
            <div class="meta-item">⚡ Target: Networked Embedded Systems &amp; Servers</div>
          </div>
        </div>

        <h2 id="sec-32-1">32.1 TCP Byte Streams vs. Packet Boundaries</h2>
        <p>
          The most prevalent defect in beginner socket programming is assuming that a single <code>write()</code> on the sender corresponds to a single <code>read()</code> on the receiver.
        </p>
        <div class="warning-box" style="border-left:4px solid #EF4444;background:rgba(239,68,68,0.08);padding:1rem;margin:1rem 0;border-radius:4px;">
          <strong style="color:#EF4444;">TCP is a Streaming Protocol:</strong> TCP has <strong>no concept of message boundaries!</strong> If a client calls <code>write(sock, buf, 100)</code> three times, the receiver’s <code>read()</code> might return 50 bytes on the first call, 200 bytes on the second, and 50 bytes on the third. Sockets must implement application-level framing!
        </div>

        <div class="diagram-box" style="background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1.25rem;font-family:monospace;font-size:0.85rem;line-height:1.5;color:#E2E8F0;margin:1.5rem 0;">
          <div style="color:#38BDF8;font-weight:bold;margin-bottom:0.5rem;">Length-Prefixed Binary Wire Framing</div>
          <pre style="margin:0;color:#94A3B8;">
Binary Stream on the Wire:
┌──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Magic Code (2 Bytes) │ Payload Len (2 Bytes)│ Payload Data (N Bytes│ CRC16 (2 Bytes)      │
│ 0xAA 0x55            │ 0x00 0x08 (8 bytes)  │ [ 8 Bytes of Telemetry ] │ 0x4B 0x12        │
└──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
 Receiver reads EXACTLY 4 bytes first -&gt; Decodes Length -&gt; Loops until N+2 bytes read!
          </pre>
        </div>

        <h2 id="sec-32-2">32.2 Endianness & Binary Wire Serialization</h2>
        <p>
          Network protocols standardize on <strong>Big-Endian</strong> (Network Byte Order: most significant byte transmitted first). Modern client processors (x86, ARM Cortex-M) are predominantly <strong>Little-Endian</strong>.
        </p>
        <p>
          Never cast raw memory structs directly onto network sockets without endianness conversion:
        </p>
        <ul>
          <li><code>htons()</code> / <code>ntohs()</code>: Host-to-Network / Network-to-Host for 16-bit integers.</li>
          <li><code>htonl()</code> / <code>ntohl()</code>: Host-to-Network / Network-to-Host for 32-bit integers.</li>
        </ul>

        <h3>Struct Packing Hazards</h3>
        <p>
          Different compiler ABIs insert padding bytes to align struct members to 4-byte or 8-byte boundaries. If transmitting a struct over the wire, packing must be enforced:
        </p>
        <pre><code class="language-c">#pragma pack(push, 1)
typedef struct {
    uint16_t magic;      /* 2 bytes */
    uint16_t length;     /* 2 bytes */
    uint32_t timestamp;  /* 4 bytes (Packed: zero padding bytes!) */
} __attribute__((packed)) wire_header_t;
#pragma pack(pop)</code></pre>

        <h2 id="sec-32-3">32.3 Socket Options: SO_REUSEADDR & TCP_NODELAY</h2>
        <p>
          Two essential socket options are required for production network servers:
        </p>
        <ol>
          <li><code>SO_REUSEADDR</code>: Bypasses the <code>TIME_WAIT</code> state, allowing immediate re-binding to a local port when a daemon is restarted.</li>
          <li><code>TCP_NODELAY</code>: Disables <strong>Nagle’s Algorithm</strong>. Nagle’s algorithm buffers small outgoing packets to minimize header overhead; in real-time telemetry or robotics control, Nagle’s delay introduces up to 200 ms of intolerable latency!</li>
        </ol>

        <h2 id="sec-32-4">32.4 Industrial Case Study: Medical Infusion Pump Packet Fragmentation</h2>
        <div class="case-study-card" style="border:1px solid #334155;border-radius:8px;padding:1.5rem;background:rgba(30,41,59,0.7);margin:1.5rem 0;">
          <h4 style="margin-top:0;color:#F59E0B;">The Fragmented Dosage Telemetry Failure</h4>
          <p>
            A hospital bedside infusion pump transmitting real-time insulin infusion rates to a central nurses' station displayed sporadic "Corrupt Dosage Rate: 0.0 mg/h" alarms.
          </p>
          <p>
            <strong>Investigation:</strong> The pump transmitted a 24-byte telemetry structure every 100 milliseconds. The hospital network switch experienced MTU fragmentation during a backup transfer.
          </p>
          <p>
            The receiver code executed:
          </p>
          <pre><code class="language-c">/* Faulty Assumption: 1 read() == 1 full packet */
ssize_t bytes = read(sock_fd, &amp;telemetry, sizeof(telemetry));
if (bytes &gt; 0) {
    display_dosage(telemetry.dosage_rate); /* Reads half-filled struct! */
}</code></pre>
          <p>
            When network congestion caused TCP to split the 24-byte message into two 12-byte fragments, the receiver parsed the first 12 bytes immediately, reading uninitialized zeroes for the dosage rate!
          </p>
          <p>
            <strong>The Fix:</strong> The protocol was refactored with a 4-byte header containing a length field, and the receiver was rewritten with an exact-length framing loop.
          </p>
        </div>

        <h2 id="sec-32-5">32.5 Hands-On Lab: Framing Client & Server in C</h2>
        <p>
          In this lab, you will construct a production-grade binary protocol parser that sends and receives length-prefixed binary telemetry frames over a TCP socket.
        </p>

        <h3>Robust Framing Helper (<code>wire_protocol.c</code>)</h3>
        <pre><code class="language-c">#include &lt;stdio.h&gt;
#include &lt;stdint.h&gt;
#include &lt;unistd.h&gt;
#include &lt;arpa/inet.h&gt;

#define MAGIC_HEADER 0xA55A

#pragma pack(push, 1)
typedef struct {
    uint16_t magic;
    uint16_t length;
} frame_header_t;
#pragma pack(pop)

/* Helper: Read exact number of bytes regardless of TCP fragmentation */
int read_exact(int fd, void *buf, size_t len) {
    char *ptr = (char *)buf;
    size_t remaining = len;
    while (remaining &gt; 0) {
        ssize_t n = read(fd, ptr, remaining);
        if (n &lt;= 0) return -1; // Connection closed or error
        ptr += n;
        remaining -= (size_t)n;
    }
    return 0; // Exactly len bytes read!
}

/* Helper: Send framed binary message */
int send_framed_packet(int fd, const void *payload, uint16_t payload_len) {
    frame_header_t hdr;
    hdr.magic = htons(MAGIC_HEADER);
    hdr.length = htons(payload_len);

    if (write(fd, &amp;hdr, sizeof(hdr)) != sizeof(hdr)) return -1;
    if (write(fd, payload, payload_len) != payload_len) return -1;
    return 0;
}

/* Helper: Receive framed binary message */
int receive_framed_packet(int fd, void *out_buf, uint16_t max_buf_len, uint16_t *out_payload_len) {
    frame_header_t hdr;
    if (read_exact(fd, &amp;hdr, sizeof(hdr)) != 0) return -1;

    uint16_t magic = ntohs(hdr.magic);
    uint16_t len = ntohs(hdr.length);

    if (magic != MAGIC_HEADER || len &gt; max_buf_len) return -2; // Corrupt frame

    if (read_exact(fd, out_buf, len) != 0) return -3;
    *out_payload_len = len;
    return 0; // Success!
}</code></pre>

        <h2 id="sec-32-quiz">32.6 Knowledge Verification &amp; Self-Study Quiz</h2>
        <div class="quiz-container" style="background:var(--bg-card);border:1px solid var(--border-strong);border-radius:12px;padding:1.5rem;margin:1.5rem 0;">
          <h3 style="margin-top:0;color:#38BDF8;">Self-Assessment Check</h3>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q1: What network byte order is standard across internet protocols such as TCP/IP?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q1" id="q1a"> <label for="q1a">A) Little-Endian</label></li>
              <li><input type="radio" name="q1" id="q1b"> <label for="q1b">B) Big-Endian (Correct)</label></li>
              <li><input type="radio" name="q1" id="q1c"> <label for="q1c">C) Mixed-Endian</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Internet standards dictate Big-Endian (most significant byte first) as the canonical Network Byte Order.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q2: What is the primary purpose of disabling Nagle’s algorithm via <code>TCP_NODELAY</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q2" id="q2a"> <label for="q2a">A) To eliminate write latency by transmitting small packets immediately without buffering. (Correct)</label></li>
              <li><input type="radio" name="q2" id="q2b"> <label for="q2b">B) To encrypt data payloads.</label></li>
              <li><input type="radio" name="q2" id="q2c"> <label for="q2c">C) To prevent packet drops.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> Nagle buffers small segments to combine them into single MTUs; disabling it via <code>TCP_NODELAY</code> provides sub-millisecond latency for real-time telemetry.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q3: Why can a single <code>read()</code> on a TCP stream socket return fewer bytes than were sent in a single <code>write()</code>?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q3" id="q3a"> <label for="q3a">A) TCP is a stream protocol without packet boundaries; IP fragmentation and buffer chunks deliver partial streams. (Correct)</label></li>
              <li><input type="radio" name="q3" id="q3b"> <label for="q3b">B) The router modifies the byte count.</label></li>
              <li><input type="radio" name="q3" id="q3c"> <label for="q3c">C) TCP truncates payloads above 128 bytes.</label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> TCP guarantees ordered, reliable byte stream delivery, but preserves no delineation of application message boundaries.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:1.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q4: What socket option allows an application to immediately bind to a port that is currently in the <code>TIME_WAIT</code> state?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q4" id="q4a"> <label for="q4a">A) <code>SO_BROADCAST</code></label></li>
              <li><input type="radio" name="q4" id="q4b"> <label for="q4b">B) <code>SO_REUSEADDR</code> (Correct)</label></li>
              <li><input type="radio" name="q4" id="q4c"> <label for="q4c">C) <code>SO_KEEPALIVE</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>SO_REUSEADDR</code> allows daemons to restart and re-bind their listening port immediately without waiting 60-120 seconds for <code>TIME_WAIT</code> sockets to expire.
            </div>
          </div>

          <div class="quiz-question" style="margin-bottom:0.5rem;">
            <p style="font-weight:600;margin-bottom:0.5rem;">Q5: What compiler attribute eliminates automatic padding bytes between struct members?</p>
            <ul style="list-style:none;padding-left:0;font-size:0.9rem;line-height:1.8;">
              <li><input type="radio" name="q5" id="q5a"> <label for="q5a">A) <code>__attribute__((aligned(4)))</code></label></li>
              <li><input type="radio" name="q5" id="q5b"> <label for="q5b">B) <code>__attribute__((packed))</code> (Correct)</label></li>
              <li><input type="radio" name="q5" id="q5c"> <label for="q5c">C) <code>__attribute__((pure))</code></label></li>
            </ul>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;padding:0.5rem;background:rgba(56,189,248,0.06);border-radius:6px;">
              <strong>Explanation:</strong> <code>__attribute__((packed))</code> instructs GCC to pack struct members with minimum byte alignment, preventing architecture-specific padding from altering wire serialization layouts.
            </div>
          </div>
        </div>'''
